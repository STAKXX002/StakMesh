// test_distributed_training_loop.cpp
//
// Closer to what Phase 2's real training loop will do than the single-step
// integration test: two ranks construct nn::Linear INDEPENDENTLY (so they
// start with genuinely different random weights, unlike the earlier test
// which cheated by copying), call broadcast_parameters() once, then run
// several training steps back-to-back on the SAME established ring
// connection - each step with different synthetic "data shard" input, each
// step calling sync_gradients() + opt.step().
//
// This is the real risk surface Phase 2 depends on: does the ring stay
// usable across MANY sequential collective calls on the same sockets, not
// just one? If chunk-size bookkeeping or socket framing had any hidden
// per-call state bug, it would show up as growing divergence over steps,
// not on step 1.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/optim.hpp>

#include "../include/stakmesh/dist/distributed_context.hpp"

#include <thread>
#include <iostream>
#include <cmath>
#include <vector>

using namespace stakml;

static const int kSteps = 25;

int main() {
    const std::vector<stakmesh::comm::PeerAddr> peers = {
        {"127.0.0.1", 16100}, {"127.0.0.1", 16101}
    };

    std::vector<std::vector<float>> w_history[2];  // per-rank, per-step snapshots
    bool ok = true;
    std::string err0, err1;

    auto worker = [&](int rank) {
        try {
            // Independently random init -- NOT copied from a shared reference
            // this time. If broadcast_parameters() doesn't work, this test
            // fails immediately at step 0.
            nn::Linear fc(4, 3);
            optim::SGD opt(fc.parameters(), 0.05f);

            auto ctx = stakmesh::dist::DistributedContext::init(rank, peers);
            ctx.broadcast_parameters(fc.parameters(), /*root=*/0);

            for (int step = 0; step < kSteps; ++step) {
                // Different synthetic "shard" per rank per step -- deterministic
                // so both ranks' expected combined behavior is reproducible.
                std::vector<float> xdata(4);
                for (int i = 0; i < 4; ++i) {
                    xdata[i] = std::sin(static_cast<float>(step * 4 + i + rank * 100)) * 0.5f;
                }
                auto x = std::make_shared<Tensor>(Tensor({1, 4}, xdata));

                opt.zero_grad();
                Tensor out = fc.forward(x);
                out.backward();

                ctx.sync_gradients(fc.parameters());
                opt.step();

                std::vector<float> snapshot(fc.W->raw_ptr(), fc.W->raw_ptr() + fc.W->num_elements());
                w_history[rank].push_back(std::move(snapshot));
            }
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

    int first_divergence = -1;
    for (int step = 0; step < kSteps; ++step) {
        const auto& a = w_history[0][step];
        const auto& b = w_history[1][step];
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a[i] - b[i]) > 1e-4f) {
                first_divergence = step;
                break;
            }
        }
        if (first_divergence != -1) break;
    }

    if (first_divergence == -1) {
        std::cout << "[PASS] weights stayed identical across both ranks for all "
                  << kSteps << " consecutive training steps\n";
        return 0;
    } else {
        std::cout << "[FAIL] ranks diverged at step " << first_divergence << "\n";
        return 1;
    }
}
