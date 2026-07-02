#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// ring_allreduce.hpp - bandwidth-optimal ring all-reduce over raw TCP sockets
//
// CONCEPT: What is ring all-reduce, and why this shape?
//
//   Every rank has a local buffer of N floats (for us: a parameter's
//   gradient). We want every rank to end up with the elementwise SUM (or
//   AVERAGE) of all ranks' buffers - that's what "all-reduce" means.
//
//   The naive approach - everyone sends their whole buffer to rank 0, which
//   sums and broadcasts back - makes rank 0 a bandwidth bottleneck: it
//   moves 2×(W-1)×N floats while everyone else only moves N. Ring
//   all-reduce spreads the work evenly: split each buffer into W chunks,
//   arrange ranks in a ring, and do it in two phases:
//
//   Phase 1 - Reduce-Scatter (W-1 steps):
//     Each step, every rank sends one chunk to its successor and receives
//     one chunk from its predecessor, ADDING the received chunk into its
//     own copy of that chunk. After W-1 steps, each rank holds the FULLY
//     REDUCED sum for exactly one chunk (a different chunk per rank).
//
//   Phase 2 - All-Gather (W-1 steps):
//     Same ring, same chunk rotation, but this time each rank just
//     forwards the fully-reduced chunk it received onward instead of
//     adding - like passing a completed puzzle piece around the ring
//     until everyone has all W pieces.
//
//   Total data moved per rank: 2×(W-1)/W × N ≈ 2N floats, regardless of
//   how many ranks are in the ring. That's why this is the algorithm
//   NCCL/Horovod/PyTorch DDP all use under the hood - we're building the
//   same primitive, just over plain TCP instead of NCCL/InfiniBand.
//
//   This file knows nothing about StakML or Tensor - it operates on a
//   plain `float*` and a length. The StakML-specific glue (which buffers
//   to reduce, when to call this relative to backward()/step()) lives in
//   dist/distributed_context.hpp.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <cstddef>
#include <algorithm>

#include "socket.hpp"

namespace stakmesh {
namespace comm {

enum class ReduceOp { Sum, Average };

// Splits `n` elements into `world_size` contiguous chunks as evenly as
// possible. Remainder elements are distributed one-per-chunk starting from
// chunk 0, so chunk sizes differ by at most 1 - this must be computed
// IDENTICALLY on every rank (no communication needed) since both sides of
// each send/recv rely on agreeing about chunk sizes without a length header.
inline void compute_chunk_bounds(size_t n, int world_size,
                                  std::vector<size_t>& starts,
                                  std::vector<size_t>& sizes) {
    starts.assign(world_size, 0);
    sizes.assign(world_size, 0);

    size_t base = n / static_cast<size_t>(world_size);
    size_t remainder = n % static_cast<size_t>(world_size);

    size_t offset = 0;
    for (int i = 0; i < world_size; ++i) {
        size_t this_size = base + (static_cast<size_t>(i) < remainder ? 1 : 0);
        starts[i] = offset;
        sizes[i] = this_size;
        offset += this_size;
    }
}

// Performs an in-place ring all-reduce on `data[0..n)`.
//
//   rank, world_size  - this process's position in the ring
//   send_sock         - connection to (rank+1) % world_size
//   recv_sock         - connection from (rank-1+world_size) % world_size
//
// All ranks must call this concurrently with the same `n` and `op`.
// After this returns, `data` holds the reduced result on every rank.
inline void ring_all_reduce(float* data, size_t n, int rank, int world_size,
                             const TcpSocket& send_sock, const TcpSocket& recv_sock,
                             ReduceOp op = ReduceOp::Average) {
    if (world_size == 1) return;  // nothing to reduce against

    std::vector<size_t> starts, sizes;
    compute_chunk_bounds(n, world_size, starts, sizes);

    // Scratch buffer sized for the largest chunk - reused every step so we
    // don't allocate inside the hot loop.
    size_t max_chunk = 0;
    for (size_t s : sizes) max_chunk = std::max(max_chunk, s);
    std::vector<float> recv_buf(max_chunk);

    // ── Phase 1: Reduce-Scatter ──────────────────────────────────────────
    // At step s, rank `r` sends the chunk it currently owns for index
    // (r - s + W) % W, and receives+accumulates the chunk for index
    // (r - s - 1 + W) % W. After W-1 steps, rank r's chunk index
    // (r + 1) % W holds the true sum across all ranks.
    for (int step = 0; step < world_size - 1; ++step) {
        int send_chunk = ((rank - step) % world_size + world_size) % world_size;
        int recv_chunk = ((rank - step - 1) % world_size + world_size) % world_size;

        send_sock.send_all(data + starts[send_chunk], sizes[send_chunk] * sizeof(float));
        recv_sock.recv_all(recv_buf.data(), sizes[recv_chunk] * sizeof(float));

        float* dst = data + starts[recv_chunk];
        for (size_t i = 0; i < sizes[recv_chunk]; ++i) {
            dst[i] += recv_buf[i];
        }
    }

    // ── Phase 2: All-Gather ──────────────────────────────────────────────
    // Same rotation, but this time we just forward the fully-reduced chunk
    // instead of adding - every rank ends up with every chunk.
    for (int step = 0; step < world_size - 1; ++step) {
        int send_chunk = ((rank - step + 1) % world_size + world_size) % world_size;
        int recv_chunk = ((rank - step) % world_size + world_size) % world_size;

        send_sock.send_all(data + starts[send_chunk], sizes[send_chunk] * sizeof(float));
        recv_sock.recv_all(data + starts[recv_chunk], sizes[recv_chunk] * sizeof(float));
    }

    // ── Optional averaging ───────────────────────────────────────────────
    if (op == ReduceOp::Average) {
        float inv_w = 1.0f / static_cast<float>(world_size);
        for (size_t i = 0; i < n; ++i) data[i] *= inv_w;
    }
}

}  // namespace comm
}  // namespace stakmesh
