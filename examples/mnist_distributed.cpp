// mnist_distributed.cpp - Phase 2: real data-parallel training across your
// two laptops.
//
// Same model, same MNIST files, same training pattern as StakML's own
// examples/mnist_mlp.cpp - the only additions are the three distributed
// concerns every data-parallel framework has to solve:
//
//   1. STARTING FROM THE SAME WEIGHTS
//      broadcast_parameters() sends rank 0's randomly-initialized weights
//      to every other rank once, before training starts.
//
//   2. EACH RANK SEEING DIFFERENT DATA
//      The training set is sharded contiguously by rank: with 2 ranks,
//      rank 0 trains on the first half, rank 1 on the second half, each
//      epoch. That's the entire point of data parallelism - more data
//      gets processed per epoch without either machine seeing more than
//      its own shard.
//
//   3. STAYING IN SYNC DESPITE SEEING DIFFERENT DATA
//      sync_gradients() ring-all-reduces every parameter's gradient right
//      before opt.step(), every single batch. Because both ranks then
//      apply the identical (averaged) gradient, their weights stay
//      identical step over step even though their inputs never match.
//
// USAGE (run once per machine, with a different --rank each time):
//
//   ./mnist_distributed --rank 0 --peers 192.168.1.10:29500,192.168.1.11:29500
//   ./mnist_distributed --rank 1 --peers 192.168.1.10:29500,192.168.1.11:29500
//
// Both machines need the MNIST IDX files at --data-dir (default: ../data),
// reachable on the same LAN on the chosen ports.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/loss.hpp>
#include <stakml/optim.hpp>
#include <stakml/dataset.hpp>

#include "../include/stakmesh/dist/distributed_context.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace stakml;

// ── tiny CLI parsing - Phase 3 will replace this with a real config loader ──
struct Args {
    int rank = -1;
    std::vector<stakmesh::comm::PeerAddr> peers;
    std::string data_dir = "../data";
    size_t epochs = 5;
    size_t batch_size = 128;
};

static std::vector<stakmesh::comm::PeerAddr> parse_peers(const std::string& s) {
    std::vector<stakmesh::comm::PeerAddr> peers;
    std::stringstream ss(s);
    std::string entry;
    while (std::getline(ss, entry, ',')) {
        auto colon = entry.find(':');
        if (colon == std::string::npos)
            throw std::runtime_error("--peers entry missing ':' - expected host:port, got: " + entry);
        std::string host = entry.substr(0, colon);
        int port = std::stoi(entry.substr(colon + 1));
        peers.push_back({host, port});
    }
    return peers;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--rank") a.rank = std::stoi(next());
        else if (flag == "--peers") a.peers = parse_peers(next());
        else if (flag == "--data-dir") a.data_dir = next();
        else if (flag == "--epochs") a.epochs = std::stoul(next());
        else if (flag == "--batch-size") a.batch_size = std::stoul(next());
        else throw std::runtime_error("unknown flag: " + flag);
    }
    if (a.rank < 0) throw std::runtime_error("--rank is required (0-indexed)");
    if (a.peers.size() < 2) throw std::runtime_error("--peers must list at least 2 host:port entries");
    if (a.rank >= static_cast<int>(a.peers.size()))
        throw std::runtime_error("--rank out of range for --peers list");
    return a;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: mnist_distributed --rank <N> --peers host:port,host:port[,...] "
                  << "[--data-dir ../data] [--epochs 5] [--batch-size 128]\n";
        return 1;
    }

    const int world_size = static_cast<int>(args.peers.size());
    const bool is_root = (args.rank == 0);

    if (is_root) {
        std::cout << "══════════════════════════════════════════\n";
        std::cout << "  StakMesh - Distributed MNIST Training (" << world_size << " ranks)\n";
        std::cout << "══════════════════════════════════════════\n\n";
    }

    // ── 1. Establish the ring BEFORE touching data - if a peer is
    //    unreachable, fail fast instead of loading MNIST first. ──
    auto ctx = stakmesh::dist::DistributedContext::init(args.rank, args.peers);

    // ── 2. Load full datasets on every rank, then take this rank's shard
    //    of the TRAINING set only. Every rank evaluates on the full test
    //    set independently (cheap, and avoids needing a cross-rank metric
    //    reduction for something that's not in the training hot path). ──
    dataset::MNIST train_data, test_data;
    try {
        train_data = dataset::MNIST::load(args.data_dir + "/train-images-idx3-ubyte",
                                           args.data_dir + "/train-labels-idx1-ubyte");
        test_data = dataset::MNIST::load(args.data_dir + "/t10k-images-idx3-ubyte",
                                          args.data_dir + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& e) {
        std::cerr << "[rank " << args.rank << "] Failed to load dataset: " << e.what() << "\n";
        return 1;
    }

    const size_t shard_size = train_data.num_samples / static_cast<size_t>(world_size);
    const size_t shard_start = static_cast<size_t>(args.rank) * shard_size;
    // Last rank absorbs any remainder so no samples get silently dropped.
    const size_t shard_end = (args.rank == world_size - 1) ? train_data.num_samples
                                                             : shard_start + shard_size;
    const size_t local_samples = shard_end - shard_start;

    if (is_root) {
        std::cout << "Loaded " << train_data.num_samples << " training images, sharded across "
                  << world_size << " ranks (~" << shard_size << " samples/rank).\n";
        std::cout << "Loaded " << test_data.num_samples << " test images (evaluated on every rank).\n\n";
    }

    // ── 3. Model - identical architecture to StakML's mnist_mlp.cpp. ──
    nn::Sequential model({
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 64),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(64, 10)
    });
    optim::Adam opt(model.parameters(), 1e-3f);

    // ── 4. THE critical distributed step: force every rank onto the same
    //    starting weights before a single gradient is computed. ──
    ctx.broadcast_parameters(model.parameters(), /*root=*/0);

    const size_t train_batches = local_samples / args.batch_size;
    const size_t test_batches = test_data.num_samples / args.batch_size;

    for (size_t epoch = 0; epoch < args.epochs; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // ─── Training phase, on this rank's shard only ───────────────────
        float total_loss = 0.0f;
        int train_correct = 0;

        for (size_t b = 0; b < train_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{args.batch_size, 784});
            std::vector<int> Y_batch(args.batch_size);

            float* x_ptr = X_batch->raw_ptr();
            const float* dataset_ptr = train_data.images.raw_ptr();
            size_t offset = shard_start + b * args.batch_size;

            std::copy(dataset_ptr + offset * 784, dataset_ptr + (offset + args.batch_size) * 784, x_ptr);
            for (size_t i = 0; i < args.batch_size; ++i) Y_batch[i] = train_data.labels[offset + i];

            opt.zero_grad();
            auto logits = model.forward(X_batch);
            auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));

            total_loss += ops::nll_loss(log_probs, Y_batch);
            train_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * args.batch_size);

            log_probs.backward();

            // ── THE other critical distributed step: every rank's gradient
            //    gets averaged with every other rank's before the update. ──
            ctx.sync_gradients(model.parameters());

            opt.step();
        }

        // ─── Test phase - full test set, every rank, no gradient ─────────
        int test_correct = 0;
        for (size_t b = 0; b < test_batches; ++b) {
            auto X_batch = std::make_shared<Tensor>(std::vector<size_t>{args.batch_size, 784});
            std::vector<int> Y_batch(args.batch_size);

            float* x_ptr = X_batch->raw_ptr();
            const float* dataset_ptr = test_data.images.raw_ptr();
            size_t offset = b * args.batch_size;

            std::copy(dataset_ptr + offset * 784, dataset_ptr + (offset + args.batch_size) * 784, x_ptr);
            for (size_t i = 0; i < args.batch_size; ++i) Y_batch[i] = test_data.labels[offset + i];

            auto logits = model.forward(X_batch);
            test_correct += static_cast<int>(ops::accuracy(logits, Y_batch) * args.batch_size);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        float avg_loss = train_batches > 0 ? total_loss / train_batches : 0.0f;
        float train_acc = train_batches > 0
            ? static_cast<float>(train_correct) / (train_batches * args.batch_size) * 100.0f : 0.0f;
        float test_acc = static_cast<float>(test_correct) / (test_batches * args.batch_size) * 100.0f;

        std::cout << "[rank " << args.rank << "] Epoch " << epoch + 1 << "/" << args.epochs
                  << " | Local Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Local Train Acc: " << std::setprecision(2) << train_acc << "%"
                  << " | Test Acc: " << std::setprecision(2) << test_acc << "%"
                  << " | Time: " << std::setprecision(2) << elapsed.count() << "s\n";
    }

    if (is_root) std::cout << "\nRun complete!\n";
    return 0;
}
