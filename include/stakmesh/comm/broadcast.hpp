#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// broadcast.hpp - send one rank's buffer to every other rank
//
// CONCEPT: Why do we need this on top of all-reduce?
//
//   ring_all_reduce() combines gradients that already differ across ranks
//   into an agreed-upon value. But there's an earlier problem: StakML's
//   Linear layer initializes W and b with Tensor::xavier(), which is
//   random. If every rank constructs its own model independently, they
//   start from DIFFERENT random weights - and sync_gradients() only keeps
//   already-synced weights in sync going forward, it can't undo a
//   divergent starting point.
//
//   The fix used by every real framework: construct the model on every
//   rank (cheap, no data needed yet), then have rank 0's weights
//   BROADCAST to everyone else once, before training starts. From that
//   point on, sync_gradients() keeps everyone in lockstep.
//
//   Implementation: a simple ring pipeline. Root sends to its successor;
//   every rank that receives forwards to its own successor, EXCEPT the
//   last rank in the chain (root's predecessor), which would otherwise
//   pointlessly forward back to a root that isn't listening for it. Total
//   latency is O(world_size) hops, moving each rank's buffer at most once
//   - broadcasting a model's weights is a startup cost, not a hot-loop
//   operation, so this doesn't need the same bandwidth-optimal chunking
//   ring_all_reduce uses.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>

#include "socket.hpp"

namespace stakmesh {
namespace comm {

// Sends `data[0..n)` from `root` to every other rank. All ranks (including
// root) must call this concurrently with the same n and root.
inline void ring_broadcast(float* data, size_t n, int rank, int world_size, int root,
                            const TcpSocket& send_sock, const TcpSocket& recv_sock) {
    if (world_size == 1) return;

    const size_t bytes = n * sizeof(float);
    const int last_hop = ((root - 1) % world_size + world_size) % world_size;

    if (rank == root) {
        send_sock.send_all(data, bytes);
    } else {
        recv_sock.recv_all(data, bytes);
        if (rank != last_hop) {
            send_sock.send_all(data, bytes);
        }
    }
}

}  // namespace comm
}  // namespace stakmesh
