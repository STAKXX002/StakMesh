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
//   That's literally what sync_gradients() below does: iterate
//   model.parameters(), and for every parameter that has a gradient,
//   ring_all_reduce() its grad_ buffer in place.
//
//   Your training loop becomes:
//
//       loss->backward();
//       ctx.sync_gradients(model.parameters());   // <-- the only new line
//       opt.step();
//       opt.zero_grad();
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <memory>

#include <stakml/tensor.hpp>

#include "../comm/socket.hpp"
#include "../comm/topology.hpp"
#include "../comm/ring_allreduce.hpp"

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

    // Ring-all-reduces the .grad_ buffer of every parameter that has one.
    // Parameters with requires_grad_ == false (or that simply haven't been
    // touched by backward() yet) are skipped, matching optim::Optimizer's
    // own skip behavior - so this is safe to call even on partially-frozen
    // models.
    void sync_gradients(const std::vector<std::shared_ptr<stakml::Tensor>>& parameters,
                         comm::ReduceOp op = comm::ReduceOp::Average) {
        for (const auto& p : parameters) {
            if (!p->grad_) continue;

            float* grad_ptr = p->grad_->raw_ptr();
            size_t n = p->grad_->num_elements();

            comm::ring_all_reduce(grad_ptr, n, links_.rank, links_.world_size,
                                   links_.send_sock, links_.recv_sock, op);
        }
    }

private:
    DistributedContext() = default;
    comm::RingLinks links_;
};

}  // namespace dist
}  // namespace stakmesh
