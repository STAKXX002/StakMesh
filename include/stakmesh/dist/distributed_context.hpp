#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// distributed_context.hpp - the ONLY file that knows about both StakML and
// the comm/ layer. Everything above (socket.hpp, topology.hpp,
// ring_allreduce.hpp) is StakML-agnostic and only moves raw floats; this
// file is the adapter that makes StakML training loops distributed.
//
// CONCEPT: Why does this slot in so cleanly?
//
//   StakML's optim::Optimizer::step() already does this, per-parameter:
//
//       for (auto& p : parameters_) {
//           if (!p->grad_) continue;
//           float* param_ptr = p->raw_ptr();
//           float* grad_ptr  = p->grad_->raw_ptr();
//           size_t n = p->num_elements();
//           ... update param_ptr using grad_ptr ...
//       }
//
//   Multi-node data parallelism only needs ONE thing inserted between
//   `backward()` finishing and `step()` running: replace each rank's local
//   gradient with the AVERAGE gradient across all ranks. Every rank then
//   calls the exact same, unmodified optim::Optimizer::step() on its own
//   copy of the weights and - because the input (averaged gradients) is
//   now identical everywhere - every rank's weights stay identical too,
//   with zero changes to Tensor, ops, nn, or optim.
//
//   Your training loop looks like this, unchanged by anything below:
//
//       loss->backward();
//       ctx.sync_gradients(model.parameters());   // <-- the only new line
//       opt.step();
//       opt.zero_grad();
//
// CONCEPT: Why flatten instead of one ring_all_reduce() call per parameter?
//
//   Network cost has two parts: latency (fixed cost per round trip) and
//   bandwidth (cost per byte). For small buffers -- and a bias vector of
//   10 floats is about as small as it gets -- latency dominates almost
//   completely. Calling ring_all_reduce() once per parameter tensor (6
//   calls for a 3-layer MLP) means paying that fixed round-trip cost 6
//   times per batch, sequentially, even though the actual data barely
//   takes any time to move once the connection is talking.
//
//   The fix every real framework uses (PyTorch DDP calls this "gradient
//   bucketing"): copy every parameter's gradient into ONE contiguous
//   scratch buffer, all-reduce THAT buffer in a single call, then copy the
//   results back out to each parameter's own .grad_. Same total bytes
//   moved, but the round-trip count drops from one-per-tensor to a fixed
//   small number regardless of how many layers the model has.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <memory>
#include <algorithm>

#include <stakml/tensor.hpp>

#include "../comm/socket.hpp"
#include "../comm/topology.hpp"
#include "../comm/ring_allreduce.hpp"
#include "../comm/broadcast.hpp"

namespace stakmesh {
namespace dist {

class DistributedContext {
public:
    // Establishes the ring for this process. Call once at startup, before
    // the training loop, on every rank simultaneously.
    static DistributedContext init(int rank, const std::vector<comm::PeerAddr>& peers) {
        DistributedContext ctx;
        ctx.links_ = comm::establish_ring(rank, peers);
        return ctx;
    }

    int rank() const { return links_.rank; }
    int world_size() const { return links_.world_size; }

    // Broadcasts every parameter's VALUE (not gradient) from `root` to all
    // other ranks. Call this ONCE, right after constructing the model and
    // BEFORE the training loop starts - it's what makes every rank's
    // Tensor::xavier()-randomized weights converge to root's weights before
    // a single gradient has been computed. Without this, sync_gradients()
    // alone would keep ranks' weights drifting in lockstep from DIFFERENT
    // starting points, which is not the same as training one shared model.
    //
    // Only called once per run, so the round-trip savings here don't matter
    // much -- flattened anyway, mostly for symmetry with sync_gradients().
    void broadcast_parameters(const std::vector<std::shared_ptr<stakml::Tensor>>& parameters,
                               int root = 0) {
        size_t total = 0;
        for (const auto& p : parameters) total += p->num_elements();
        if (total == 0) return;

        flat_buffer_.resize(total);

        // Only root's values are read as input -- every other rank's buffer
        // is about to be overwritten by ring_broadcast() regardless, so
        // there's nothing to gather from a non-root rank first.
        if (links_.rank == root) {
            size_t offset = 0;
            for (const auto& p : parameters) {
                size_t n = p->num_elements();
                std::copy(p->raw_ptr(), p->raw_ptr() + n, flat_buffer_.begin() + offset);
                offset += n;
            }
        }

        comm::ring_broadcast(flat_buffer_.data(), total, links_.rank, links_.world_size, root,
                              links_.send_sock, links_.recv_sock);

        size_t offset = 0;
        for (const auto& p : parameters) {
            size_t n = p->num_elements();
            std::copy(flat_buffer_.begin() + offset, flat_buffer_.begin() + offset + n, p->raw_ptr());
            offset += n;
        }
    }

    // Ring-all-reduces the .grad_ buffer of every parameter that has one, in
    // a SINGLE round trip regardless of parameter count. Parameters without
    // a gradient (frozen, or simply not yet touched by backward()) are
    // skipped when computing the flattened layout, matching
    // optim::Optimizer's own skip behavior.
    void sync_gradients(const std::vector<std::shared_ptr<stakml::Tensor>>& parameters,
                         comm::ReduceOp op = comm::ReduceOp::Average) {
        size_t total = 0;
        for (const auto& p : parameters) {
            if (p->grad_) total += p->grad_->num_elements();
        }
        if (total == 0) return;

        flat_buffer_.resize(total);

        // ── Gather: every parameter's gradient into one contiguous buffer ──
        size_t offset = 0;
        for (const auto& p : parameters) {
            if (!p->grad_) continue;
            size_t n = p->grad_->num_elements();
            std::copy(p->grad_->raw_ptr(), p->grad_->raw_ptr() + n, flat_buffer_.begin() + offset);
            offset += n;
        }

        // ── The only network round trip(s) in this whole function ─────────
        comm::ring_all_reduce(flat_buffer_.data(), total, links_.rank, links_.world_size,
                               links_.send_sock, links_.recv_sock, op);

        // ── Scatter: reduced values back into each parameter's .grad_ ─────
        offset = 0;
        for (const auto& p : parameters) {
            if (!p->grad_) continue;
            size_t n = p->grad_->num_elements();
            std::copy(flat_buffer_.begin() + offset, flat_buffer_.begin() + offset + n,
                      p->grad_->raw_ptr());
            offset += n;
        }
    }

private:
    DistributedContext() = default;
    comm::RingLinks links_;

    // Reused across calls. std::vector::resize() is a no-op if the size
    // already matches, so for a fixed model architecture this allocates
    // exactly once, on the very first call, and never again.
    std::vector<float> flat_buffer_;
};

}  // namespace dist
}  // namespace stakmesh