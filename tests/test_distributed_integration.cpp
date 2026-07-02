// test_distributed_integration.cpp
//
// End-to-end check that DistributedContext actually does what Phase 2 needs:
// two simulated ranks, each running a real StakML nn::Linear with the SAME
// initial weights but DIFFERENT input data (so their local gradients differ),
// call sync_gradients(), then opt.step(). If the plumbing is correct, both
// ranks must land on IDENTICAL weights afterward - that's the entire point
// of data-parallel training: every replica sees a different data shard but
// stays in sync because gradients (not data) are what get averaged.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/optim.hpp>

#include "../include/stakmesh/dist/distributed_context.hpp"

#include <thread>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace stakml;

int main() {
    const std::vector<stakmesh::comm::PeerAddr> peers = {
        {"127.0.0.1", 16000}, {"127.0.0.1", 16001}
    };

    // Reference layer purely to get one set of initial weights that both
    // simulated ranks will start from - mimics every real rank loading the
    // same checkpoint/seed before training begins.
    nn::Linear ref(3, 2);

    std::vector<float> w_rank0, w_rank1, b_rank0, b_rank1;
    bool ok = true;
    std::string err0, err1;

    auto worker = [&](int rank) {
        try {
            nn::Linear fc(3, 2);
            std::copy(ref.W->raw_ptr(), ref.W->raw_ptr() + ref.W->num_elements(), fc.W->raw_ptr());
            std::copy(ref.b->raw_ptr(), ref.b->raw_ptr() + ref.b->num_elements(), fc.b->raw_ptr());

            optim::SGD opt(fc.parameters(), 0.1f);

            // Different "data shard" per rank -> different local gradients.
            std::vector<float> xdata = (rank == 0) ? std::vector<float>{1.0f, 0.0f, 0.0f}
                                                     : std::vector<float>{0.0f, 1.0f, 1.0f};
            auto x = std::make_shared<Tensor>(Tensor({1, 3}, xdata));
            Tensor out = fc.forward(x);
            out.backward();  // populates fc.W->grad_ / fc.b->grad_, different per rank

            auto ctx = stakmesh::dist::DistributedContext::init(rank, peers);
            ctx.sync_gradients(fc.parameters());  // <-- the line under test

            opt.step();

            auto& w_out = (rank == 0) ? w_rank0 : w_rank1;
            auto& b_out = (rank == 0) ? b_rank0 : b_rank1;
            w_out.assign(fc.W->raw_ptr(), fc.W->raw_ptr() + fc.W->num_elements());
            b_out.assign(fc.b->raw_ptr(), fc.b->raw_ptr() + fc.b->num_elements());
        } catch (const std::exception& e) {
            ok = false;
            (rank == 0 ? err0 : err1) = e.what();
        }
    };

    std::thread t0(worker, 0), t1(worker, 1);
    t0.join();
    t1.join();

    if (!ok) {
        std::cerr << "rank0 error: " << err0 << "\nrank1 error: " << err1 << "\n";
        return 1;
    }

    auto all_close = [](const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::fabs(a[i] - b[i]) > 1e-4f) return false;
        return true;
    };

    bool weights_match = all_close(w_rank0, w_rank1);
    bool biases_match = all_close(b_rank0, b_rank1);

    std::cout << (weights_match ? "[PASS] " : "[FAIL] ")
              << "post-sync weights identical across ranks despite different input data\n";
    std::cout << (biases_match ? "[PASS] " : "[FAIL] ")
              << "post-sync biases identical across ranks\n";

    return (weights_match && biases_match) ? 0 : 1;
}
