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
                                 int io_timeout_ms = 30000, int connect_timeout_ms = 10000) {
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
    //      listener may be blocked in accept() with no timeout (only
    //      established connections get io_timeout_ms - a socket that's
    //      still just listening has no peer yet to time out on), so
    //      joining could hang exactly as badly as the bug being fixed.
    //      detach() is correct here: this is a failure path, not the hot
    //      path, and the thread dies with the process either way.
    TcpSocket recv_sock;
    std::thread listener([&]() {
        try {
            recv_sock = TcpSocket::listen_one(my_port, io_timeout_ms);
        } catch (...) {
            // swallowed deliberately - checked via recv_sock.valid() below
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
    if (!recv_sock.valid()) {
        throw SocketError("establish_ring: failed to accept predecessor connection");
    }

    RingLinks links{my_rank, world_size, std::move(send_sock), std::move(recv_sock)};
    return links;
}

}  // namespace comm
}  // namespace stakmesh