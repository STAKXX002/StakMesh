// mnist_distributed.cpp - Phase 2/3: real data-parallel training across your
// two devices, with automatic rank detection.
//
// Same model, same MNIST files, same training pattern as StakML's own
// examples/mnist_mlp.cpp - the only additions are the distributed concerns
// every data-parallel framework has to solve:
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
//   4. KNOWING WHICH RANK YOU ARE (Phase 3)
//      --config points at a file listing every machine in the cluster
//      (see configs/two_laptop_cluster.txt). Each process resolves every
//      entry's host and checks which one matches an IP it actually owns
//      -- that's its rank, detected automatically. Run the EXACT SAME
//      command on every machine; no more hand-typing --rank and risking
//      running the wrong one on the wrong laptop.
//
//   5. SURVIVING A CRASH (Phase 4)
//      Every rank saves its own weights + Adam momentum + epoch number to
//      <checkpoint-dir>/rank<N>_latest.bin after each epoch (configurable
//      via --checkpoint-every). Pass --resume (same flag on every machine,
//      same as --config) to pick back up from there instead of starting
//      over - each rank resumes its own file, no cross-machine copying.
//
// USAGE - config mode (recommended, same command on every machine):
//
//   ./mnist_distributed --config ../configs/two_laptop_cluster.txt
//
// USAGE - manual mode (still supported, e.g. for quick one-off tests):
//
//   ./mnist_distributed --rank 0 --peers 192.168.1.10:29500,192.168.1.11:29500
//   ./mnist_distributed --rank 1 --peers 192.168.1.10:29500,192.168.1.11:29500
//
// --rank can be combined with --config too, to override auto-detection if
// you ever need to (e.g. running two ranks on one machine for testing).
//
// Both machines need the MNIST IDX files at --data-dir (default: ../data),
// reachable on the same LAN (or Tailscale/VPN) on the chosen ports.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/loss.hpp>
#include <stakml/optim.hpp>
#include <stakml/dataset.hpp>

#include "../include/stakmesh/dist/distributed_context.hpp"
#include "../include/stakmesh/dist/checkpoint.hpp"
#include "../include/stakmesh/comm/cluster_config.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <filesystem>

using namespace stakml;

struct Args {
    int rank = -1;                                  // -1 == "auto-detect from --config"
    std::vector<stakmesh::comm::PeerAddr> peers;
    std::string data_dir = "../data";
    size_t epochs = 5;
    size_t batch_size = 128;
    std::string checkpoint_dir = "checkpoints";
    size_t checkpoint_every = 1;    // save every N epochs; 0 disables checkpointing
    bool resume = false;            // load <checkpoint_dir>/rank<N>_latest.bin before training
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
    std::string config_path;
    bool have_peers = false;

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--rank") a.rank = std::stoi(next());
        else if (flag == "--peers") { a.peers = parse_peers(next()); have_peers = true; }
        else if (flag == "--config") config_path = next();
        else if (flag == "--data-dir") a.data_dir = next();
        else if (flag == "--epochs") a.epochs = std::stoul(next());
        else if (flag == "--batch-size") a.batch_size = std::stoul(next());
        else if (flag == "--checkpoint-dir") a.checkpoint_dir = next();
        else if (flag == "--checkpoint-every") a.checkpoint_every = std::stoul(next());
        else if (flag == "--resume") a.resume = true;   // no value - a flag, not an option
        else throw std::runtime_error("unknown flag: " + flag);
    }

    if (!config_path.empty() && have_peers) {
        throw std::runtime_error("pass either --config or --peers, not both");
    }
    if (config_path.empty() && !have_peers) {
        throw std::runtime_error("either --config <file> or --peers <list> is required");
    }

    if (!config_path.empty()) {
        a.peers = stakmesh::comm::parse_cluster_config(config_path);
        if (a.rank < 0) {
            // Auto-detect: throws with a clear message if this machine
            // matches zero or more than one entry, rather than guessing.
            a.rank = stakmesh::comm::detect_local_rank(a.peers);
        }
    }

    if (a.rank < 0) throw std::runtime_error("--rank is required when using --peers (not --config)");
    if (a.peers.size() < 2) throw std::runtime_error("cluster must have at least 2 entries");
    if (a.rank >= static_cast<int>(a.peers.size()))
        throw std::runtime_error("--rank out of range for the peer/cluster list");

    return a;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: mnist_distributed --config <file> [--rank <N>]\n"
                  << "   or: mnist_distributed --rank <N> --peers host:port,host:port[,...] "
                  << "[--data-dir ../data] [--epochs 5] [--batch-size 128]\n"
                  << "Checkpointing: [--checkpoint-dir checkpoints] [--checkpoint-every 1] [--resume]\n"
                  << "  --resume loads <checkpoint-dir>/rank<N>_latest.bin (same flag, every machine -\n"
                  << "  each rank resumes its own file, matching the auto-rank-detection philosophy).\n";
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
    stakmesh::dist::DistributedContext ctx = [&]() {
        try {
            return stakmesh::dist::DistributedContext::init(args.rank, args.peers);
        } catch (const std::exception& e) {
            std::cerr << "[rank " << args.rank << "] Failed to establish the ring: " << e.what() << "\n";
            std::exit(1);
        }
    }();

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

    // ── 4. Either resume from this rank's own last checkpoint, or start
    //    fresh with the usual broadcast-from-rank-0 (Phase 2's approach).
    //    These are mutually exclusive: a resumed checkpoint already holds
    //    real, trained (and, across ranks, identical-by-construction)
    //    weights - broadcasting rank 0's weights over top of that would
    //    silently discard everyone else's resumed state and every rank's
    //    Adam momentum, replacing it with whatever rank 0 happened to load
    //    (or a fresh random init, if rank 0 isn't resuming). ──
    size_t start_epoch = 0;
    const std::string checkpoint_path =
        args.checkpoint_dir + "/rank" + std::to_string(args.rank) + "_latest.bin";

    if (args.resume) {
        try {
            auto info = stakmesh::dist::load_checkpoint(checkpoint_path, model.parameters(), opt);
            start_epoch = static_cast<size_t>(info.epoch) + 1;
            std::cout << "[rank " << args.rank << "] Resumed from " << checkpoint_path
                      << " (completed epoch " << info.epoch << ", continuing at epoch "
                      << start_epoch + 1 << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "[rank " << args.rank << "] --resume failed: " << e.what() << "\n";
            return 1;
        }
    } else {
        ctx.broadcast_parameters(model.parameters(), /*root=*/0);
    }

    if (args.checkpoint_every > 0) {
        std::filesystem::create_directories(args.checkpoint_dir);
    }

    if (start_epoch >= args.epochs) {
        std::cout << "[rank " << args.rank << "] Checkpoint is already at epoch "
                  << start_epoch << ", nothing to do for --epochs " << args.epochs << ".\n";
        return 0;
    }

    const size_t train_batches = local_samples / args.batch_size;
    const size_t test_batches = test_data.num_samples / args.batch_size;

    for (size_t epoch = start_epoch; epoch < args.epochs; ++epoch) {
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

        // ─── Checkpoint (this rank's own file, overwritten each save -
        //     "latest" is the only thing --resume ever looks for) ───────
        if (args.checkpoint_every > 0 && (epoch + 1) % args.checkpoint_every == 0) {
            try {
                stakmesh::dist::save_checkpoint(checkpoint_path, model.parameters(), opt,
                                                 static_cast<int>(epoch), args.rank);
            } catch (const std::exception& e) {
                // Don't abort a training run over a checkpoint write failure
                // (e.g. disk full) - just make sure it's loud, not silent.
                std::cerr << "[rank " << args.rank << "] WARNING: checkpoint save failed: "
                          << e.what() << "\n";
            }
        }
    }

    if (is_root) std::cout << "\nRun complete!\n";
    return 0;
}