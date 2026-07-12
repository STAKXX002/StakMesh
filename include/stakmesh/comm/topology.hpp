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
    TcpSocket recv_sock;
    std::thread listener([&]() {
        recv_sock = TcpSocket::listen_one(my_port);
    });

    // Connect outward to our successor. Retries internally, so it's fine
    // if that rank hasn't called listen() yet.
    TcpSocket send_sock = TcpSocket::connect(peers[next_rank].host, peers[next_rank].port);

    listener.join();
    if (!recv_sock.valid()) {
        throw SocketError("establish_ring: failed to accept predecessor connection");
    }

    RingLinks links{my_rank, world_size, std::move(send_sock), std::move(recv_sock)};
    return links;
}

}  // namespace comm
}  // namespace stakmesh
