#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// topology.hpp - turns a static list of "who's rank i" into a live ring
//
// CONCEPT: Ring bootstrap
//
//   Every rank needs exactly two live connections once setup is done:
//     send_sock_ → rank (my_rank + 1) % world_size   ("next" in the ring)
//     recv_sock_ ← rank (my_rank - 1 + world_size) % world_size ("prev")
//
//   To avoid a deadlock where everyone tries to connect() before anyone
//   calls listen(), each rank FIRST starts listening (in a background
//   thread, since listen_one() blocks until accepted) and THEN connects
//   outward to its successor. TcpSocket::connect() also retries for a
//   few seconds, which absorbs the remaining race where a peer hasn't
//   reached its listen() call yet.
//
//   For now this is entirely static: you hand it a fixed list of
//   "host:port per rank" (e.g. from a config file or CLI args). A real
//   process launcher (Phase 3) will generate this list automatically
//   instead of you typing IPs by hand.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <string>
#include <thread>
#include <stdexcept>
#include <memory>
#include <string>

#include "socket.hpp"

namespace stakmesh {
namespace comm {

struct PeerAddr {
    std::string host;
    int port;
};

// The two live connections a rank needs to participate in ring collectives.
struct RingLinks {
    int rank;
    int world_size;
    TcpSocket send_sock;  // → next rank
    TcpSocket recv_sock;  // ← prev rank
};

// `peers` must have exactly `world_size` entries, indexed by rank, giving
// the (host, port) each rank listens on. `my_rank` is this process's rank.
// `io_timeout_ms` bounds every send/recv AFTER the ring is up (see the
// CONCEPT note in socket.hpp) - 0 disables it (blocks forever, matching
// pre-Phase-4b behavior).
inline RingLinks establish_ring(int my_rank, const std::vector<PeerAddr>& peers,
                                 int io_timeout_ms = 30000, int connect_timeout_ms = 10000,
                                 int accept_timeout_ms = 60000) {
    const int world_size = static_cast<int>(peers.size());
    if (world_size < 2) {
        throw std::invalid_argument("establish_ring: need at least 2 ranks");
    }
    if (my_rank < 0 || my_rank >= world_size) {
        throw std::invalid_argument("establish_ring: rank out of range");
    }

    const int next_rank = (my_rank + 1) % world_size;
    const int my_port = peers[my_rank].port;

    // Start listening for the predecessor's connection on a background
    // thread - listen_one() blocks until someone connects, and we also
    // need to connect() outward at the same time.
    //
    // Two failure-handling subtleties here, both found by actually
    // crashing on an unreachable peer, not by inspection:
    //
    //   1. An exception escaping a std::thread's top-level function is
    //      ALWAYS fatal (std::terminate, unconditionally) - so listen_one()
    //      failing must be caught INSIDE the lambda. The failure still
    //      surfaces below via `recv_sock` staying invalid.
    //
    //   2. If connect() throws (peer unreachable), this function would
    //      previously return via an exception while `listener` was still
    //      joinable - a joinable std::thread's destructor ALSO calls
    //      std::terminate(), unconditionally, even during unwinding. We
    //      can't just add a `listener.join()` on this path either: the
    //      listener may still be waiting in accept() (up to
    //      accept_timeout_ms), so joining would stall the error report.
    //      detach() is correct here: this is a failure path, not the hot
    //      path.
    //
    //   3. A detached thread outlives this function, so it must NOT touch
    //      this function's locals: the old version captured `recv_sock` by
    //      reference, and on the connect-failure path the thread could later
    //      write into a stack slot that no longer existed. The thread owns a
    //      shared_ptr to its own state instead, so it stays valid however
    //      long the thread lives. accept_timeout_ms also guarantees the
    //      thread ends on its own rather than waiting in accept() forever.
    struct ListenState {
        TcpSocket sock;
        std::string error;
    };
    auto listen_state = std::make_shared<ListenState>();
    std::thread listener([listen_state, my_port, io_timeout_ms, accept_timeout_ms]() {
        try {
            listen_state->sock = TcpSocket::listen_one(my_port, io_timeout_ms, accept_timeout_ms);
        } catch (const std::exception& e) {
            listen_state->error = e.what();   // surfaced below via the thrown SocketError
        } catch (...) {
            listen_state->error = "unknown error";
        }
    });

    TcpSocket send_sock;
    try {
        // Connect outward to our successor. Retries internally, so it's
        // fine if that rank hasn't called listen() yet.
        send_sock = TcpSocket::connect(peers[next_rank].host, peers[next_rank].port,
                                        connect_timeout_ms, /*retry_delay_ms=*/100,
                                        io_timeout_ms);
    } catch (...) {
        listener.detach();
        throw;
    }

    listener.join();
    if (!listen_state->sock.valid()) {
        throw SocketError("establish_ring: failed to accept predecessor connection (rank " +
                           std::to_string((my_rank - 1 + world_size) % world_size) + "): " +
                           listen_state->error);
    }

    RingLinks links{my_rank, world_size, std::move(send_sock), std::move(listen_state->sock)};
    return links;
}

}  // namespace comm
}  // namespace stakmesh