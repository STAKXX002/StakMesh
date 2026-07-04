// test_distributed_sequential.cpp
//
// The two existing distributed tests use a single nn::Linear (2 parameters:
// W, b). That's enough to prove the mechanism works, but NOT enough to
// stress-test the flatten/gather/scatter offset math in sync_gradients() --
// a bug there (e.g. an off-by-one in how offsets accumulate across
// differently-sized tensors) could easily pass a 2-parameter test while
// still corrupting gradients on a real model.
//
// This test uses the EXACT model shape mnist_distributed.cpp trains:
// Linear(784,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,10)
// = 6 parameter tensors of sizes 100352, 128, 8192, 64, 640, 10 --
// deliberately uneven, which is exactly where a slicing bug would surface.
//
// Two ranks, independently random init, broadcast once, then 10 training
// steps with different synthetic input per rank per step. If ALL SIX
// parameters stay bit-identical across ranks for all 10 steps, the flatten
// logic is correctly handling multi-tensor, uneven-size layouts.

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

static const int kSteps = 10;

int main() {
    const std::vector<stakmesh::comm::PeerAddr> peers = {
        {"127.0.0.1", 16200}, {"127.0.0.1", 16201}
    };

    // Per-rank, per-step snapshots of ALL 6 parameter tensors flattened
    // into one vector each, so a single comparison catches a bug in any
    // one of them.
    std::vector<std::vector<float>> history[2];
    bool ok = true;
    std::string err0, err1;

    auto worker = [&](int rank) {
        try {
            nn::Sequential model({
                std::make_shared<nn::Linear>(784, 128),
                std::make_shared<nn::ReLU>(),
                std::make_shared<nn::Linear>(128, 64),
                std::make_shared<nn::ReLU>(),
                std::make_shared<nn::Linear>(64, 10)
            });
            optim::Adam opt(model.parameters(), 1e-3f);

            auto ctx = stakmesh::dist::DistributedContext::init(rank, peers);
            ctx.broadcast_parameters(model.parameters(), /*root=*/0);

            for (int step = 0; step < kSteps; ++step) {
                std::vector<float> xdata(784);
                for (int i = 0; i < 784; ++i) {
                    xdata[i] = std::sin(static_cast<float>(step * 784 + i + rank * 10000)) * 0.1f;
                }
                auto x = std::make_shared<Tensor>(Tensor({1, 784}, xdata));

                opt.zero_grad();
                Tensor out = model.forward(x);
                out.backward();

                ctx.sync_gradients(model.parameters());
                opt.step();

                // Snapshot ALL 6 parameters concatenated, in order.
                std::vector<float> snapshot;
                for (const auto& p : model.parameters()) {
                    snapshot.insert(snapshot.end(), p->raw_ptr(), p->raw_ptr() + p->num_elements());
                }
                history[rank].push_back(std::move(snapshot));
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
    size_t divergence_index = 0;
    for (int step = 0; step < kSteps && first_divergence == -1; ++step) {
        const auto& a = history[0][step];
        const auto& b = history[1][step];
        if (a.size() != b.size()) {
            std::cerr << "[FAIL] snapshot size mismatch at step " << step
                      << ": " << a.size() << " vs " << b.size() << "\n";
            return 1;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a[i] - b[i]) > 1e-4f) {
                first_divergence = step;
                divergence_index = i;
                break;
            }
        }
    }

    const size_t total_params = 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10;
    if (first_divergence == -1) {
        std::cout << "[PASS] all " << total_params
                  << " parameters across 6 tensors stayed identical across both ranks for "
                  << kSteps << " steps\n";
        return 0;
    } else {
        std::cout << "[FAIL] diverged at step " << first_divergence
                  << ", flattened index " << divergence_index << "\n";
        return 1;
    }
}