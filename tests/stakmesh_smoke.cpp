// stakmesh_smoke.cpp - one-binary end-to-end check that a StakMesh ring works.
//
// Runs the REAL code path (separate processes, kernel TCP, establish_ring,
// ring_all_reduce, ring_broadcast, sync_gradients) with NO dataset needed.
// Takes the same --config / --peers / --rank flags as mnist_distributed, so
// the identical binary can be run two ways and the results compared:
//
//   Loopback, one machine (spawns every rank for you):
//       scripts/loopback_test.sh
//
//   Real cluster (same command on each machine, e.g. over Tailscale):
//       ./stakmesh_smoke --config configs/two_laptop_cluster.txt
//
// Stages (each rank prints PASS/FAIL per stage; exit code 0 only if all pass):
//   1. ring       - establish_ring handshake
//   2. broadcast  - root's buffer reaches everyone (several sizes)
//   3. allreduce  - Sum/Average vs. closed-form expected values, including
//                   n < world_size, odd sizes, and non-divisible sizes
//   4. large      - big buffer all-reduce (catches send/send deadlock) + MB/s
//   5. training   - tiny MLP on synthetic sharded data: weights must stay
//                   BIT-IDENTICAL across ranks and the loss must fall
//
// Exit codes: 0 pass, 1 check failed, 2 bad args, 3 comm/socket error.
// --die-at-step K makes the LAST rank _exit() abruptly mid-training, to
// verify that survivors fail with a clean error instead of hanging/SIGPIPE.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/loss.hpp>
#include <stakml/optim.hpp>

#include "../include/stakmesh/dist/distributed_context.hpp"
#include "../include/stakmesh/comm/cluster_config.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace stakml;
using stakmesh::comm::PeerAddr;
using stakmesh::comm::ReduceOp;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    int rank = -1;
    std::vector<PeerAddr> peers;
    int io_timeout_ms = 15000;
    size_t large_floats = 4u * 1024 * 1024;   // 16 MB
    int train_steps = 150;
    int die_at_step = -1;                      // -1 == disabled
};

std::vector<PeerAddr> parse_peers(const std::string& s) {
    std::vector<PeerAddr> peers;
    std::stringstream ss(s);
    std::string entry;
    while (std::getline(ss, entry, ',')) {
        auto colon = entry.rfind(':');
        if (colon == std::string::npos)
            throw std::runtime_error("--peers entry needs host:port, got: " + entry);
        peers.push_back({entry.substr(0, colon), std::stoi(entry.substr(colon + 1))});
    }
    return peers;
}

Args parse_args(int argc, char** argv) {
    Args a;
    std::string config;
    bool have_peers = false;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + f);
            return argv[++i];
        };
        if (f == "--rank") a.rank = std::stoi(next());
        else if (f == "--peers") { a.peers = parse_peers(next()); have_peers = true; }
        else if (f == "--config") config = next();
        else if (f == "--io-timeout-ms") a.io_timeout_ms = std::stoi(next());
        else if (f == "--large-floats") a.large_floats = std::stoul(next());
        else if (f == "--train-steps") a.train_steps = std::stoi(next());
        else if (f == "--die-at-step") a.die_at_step = std::stoi(next());
        else throw std::runtime_error("unknown flag: " + f);
    }
    if (!config.empty() == have_peers)
        throw std::runtime_error("pass exactly one of --config or --peers");
    if (!config.empty()) {
        a.peers = stakmesh::comm::parse_cluster_config(config);
        if (a.rank < 0) a.rank = stakmesh::comm::detect_local_rank(a.peers);
    }
    if (a.peers.size() < 2) throw std::runtime_error("need at least 2 peers");
    if (a.rank < 0 || a.rank >= static_cast<int>(a.peers.size()))
        throw std::runtime_error("--rank missing or out of range");
    return a;
}

// Tiny deterministic PRNG so every rank/run generates the same "data".
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    float uniform() {  // [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((s >> 40) & 0xFFFFFF) / 8388608.0f - 1.0f;
    }
};

bool g_all_ok = true;
int g_rank = 0;

void report(const char* stage, bool ok, const std::string& detail = "") {
    std::cout << "[rank " << g_rank << "] " << (ok ? "PASS" : "FAIL") << "  " << std::left
              << std::setw(10) << stage << " " << detail << "\n";
    if (!ok) g_all_ok = false;
}

// Deterministic per-rank input: small multiples of 0.5, so sums are EXACT in
// float for any realistic world size.
float input_value(int rank, size_t i) {
    return static_cast<float>(rank + 1) * static_cast<float>(i % 13 + 1) * 0.5f;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n"
                  << "Usage: stakmesh_smoke (--config <file> | --peers h:p,h:p[,...]) [--rank N]\n"
                  << "       [--io-timeout-ms 15000] [--large-floats 4194304] [--train-steps 150]\n"
                  << "       [--die-at-step K]\n";
        return 2;
    }
    g_rank = args.rank;
    const int W = static_cast<int>(args.peers.size());
    const int R = args.rank;

    try {
        // ── 1. ring ──────────────────────────────────────────────────────
        auto t0 = Clock::now();
        auto ctx = stakmesh::dist::DistributedContext::init(R, args.peers, args.io_timeout_ms);
        double ring_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        report("ring", ctx.rank() == R && ctx.world_size() == W,
               "world=" + std::to_string(W) + " setup=" + std::to_string(static_cast<int>(ring_ms)) + "ms");

        // ── 2. broadcast (every root, several sizes) ─────────────────────
        {
            bool ok = true;
            for (int root = 0; root < W; ++root) {
                for (size_t n : {size_t(1), size_t(7), size_t(1000), size_t(65537)}) {
                    std::vector<float> buf(n, -1.0f);
                    if (R == root) for (size_t i = 0; i < n; ++i) buf[i] = input_value(root, i);
                    ctx.broadcast(buf.data(), n, root);
                    for (size_t i = 0; i < n; ++i)
                        if (buf[i] != input_value(root, i)) { ok = false; break; }
                }
            }
            report("broadcast", ok, "roots=0.." + std::to_string(W - 1));
        }

        // ── 3. all-reduce correctness ────────────────────────────────────
        {
            bool ok = true;
            std::string bad;
            const double rank_sum = static_cast<double>(W) * (W + 1) / 2.0;
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(7), size_t(64),
                              size_t(1000), size_t(100003)}) {
                for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Average}) {
                    std::vector<float> buf(n);
                    for (size_t i = 0; i < n; ++i) buf[i] = input_value(R, i);
                    ctx.all_reduce(buf.data(), n, op);
                    for (size_t i = 0; i < n; ++i) {
                        double expect = rank_sum * static_cast<double>(i % 13 + 1) * 0.5;
                        if (op == ReduceOp::Average) expect /= W;
                        if (std::fabs(buf[i] - expect) > 1e-5 * std::max(1.0, std::fabs(expect))) {
                            if (ok) bad = "n=" + std::to_string(n) + " i=" + std::to_string(i) +
                                          " got " + std::to_string(buf[i]) + " want " + std::to_string(expect);
                            ok = false;
                            break;
                        }
                    }
                }
            }
            report("allreduce", ok, ok ? "Sum+Average, 7 sizes incl. n<world" : bad);
        }

        // ── 4. large buffer: deadlock check + throughput ─────────────────
        {
            const size_t n = args.large_floats;
            std::vector<float> buf(n);
            for (size_t i = 0; i < n; ++i) buf[i] = input_value(R, i);
            auto s = Clock::now();
            ctx.all_reduce(buf.data(), n, ReduceOp::Sum);
            double sec = std::chrono::duration<double>(Clock::now() - s).count();
            const double rank_sum = static_cast<double>(W) * (W + 1) / 2.0;
            bool ok = true;
            for (size_t i = 0; i < n; i += 4099) {
                double expect = rank_sum * static_cast<double>(i % 13 + 1) * 0.5;
                if (std::fabs(buf[i] - expect) > 1e-5 * std::max(1.0, expect)) { ok = false; break; }
            }
            // Per-rank traffic of ring all-reduce: 2*(W-1)/W * N * 4 bytes, each direction.
            double mb = 2.0 * (W - 1) / W * static_cast<double>(n) * 4.0 / (1024.0 * 1024.0);
            std::ostringstream d;
            d << std::fixed << std::setprecision(1) << (n * 4.0 / (1024 * 1024)) << " MB buffer in "
              << std::setprecision(3) << sec << " s (" << std::setprecision(1) << mb / sec << " MB/s/rank)";
            report("large", ok, d.str());
        }

        // ── 5. tiny sharded training run ─────────────────────────────────
        {
            constexpr int kIn = 16, kClasses = 4, kBatch = 32;
            // Fixed "teacher": label = argmax_k (w_k . x). Same on every rank.
            std::vector<float> teacher(kClasses * kIn);
            { Lcg g(12345); for (auto& v : teacher) v = g.uniform(); }
            auto label_of = [&](const float* x) {
                int best = 0; float bv = -1e30f;
                for (int k = 0; k < kClasses; ++k) {
                    float s = 0; for (int j = 0; j < kIn; ++j) s += teacher[k * kIn + j] * x[j];
                    if (s > bv) { bv = s; best = k; }
                }
                return best;
            };
            auto make_batch = [&](Lcg& g, std::shared_ptr<Tensor>& X, std::vector<int>& Y) {
                X = std::make_shared<Tensor>(std::vector<size_t>{size_t(kBatch), size_t(kIn)});
                Y.assign(kBatch, 0);
                float* p = X->raw_ptr();
                for (int i = 0; i < kBatch; ++i) {
                    for (int j = 0; j < kIn; ++j) p[i * kIn + j] = g.uniform();
                    Y[i] = label_of(p + i * kIn);
                }
            };

            nn::Sequential model({
                std::make_shared<nn::Linear>(kIn, 32), std::make_shared<nn::ReLU>(),
                std::make_shared<nn::Linear>(32, kClasses)});
            optim::Adam opt(model.parameters(), 1e-2f);

            // Each rank starts from DIFFERENT random weights (StakML seeds from
            // random_device); the broadcast is what makes them one model.
            ctx.broadcast_parameters(model.parameters(), 0);

            Lcg shard(1000 + R);   // different data per rank: the point of data parallelism
            float first_loss = 0, last_loss = 0;
            const int window = 10;
            for (int step = 0; step < args.train_steps; ++step) {
                if (args.die_at_step >= 0 && step == args.die_at_step && R == W - 1) {
                    std::cout << "[rank " << R << "] --die-at-step: exiting abruptly now\n";
                    std::_Exit(0);   // no destructors, no clean close handshake from our code
                }
                std::shared_ptr<Tensor> X; std::vector<int> Y;
                make_batch(shard, X, Y);
                opt.zero_grad();
                auto logits = model.forward(X);
                auto log_probs = ops::log_softmax(std::make_shared<Tensor>(logits));
                float loss = ops::nll_loss(log_probs, Y);
                log_probs.backward();
                ctx.sync_gradients(model.parameters());
                opt.step();
                if (step < window) first_loss += loss / window;
                if (step >= args.train_steps - window) last_loss += loss / window;
            }

            // (a) bit-identical weights across ranks: root broadcasts its copy,
            //     everyone compares bytes against their own.
            std::vector<float> mine;
            for (auto& p : model.parameters())
                mine.insert(mine.end(), p->raw_ptr(), p->raw_ptr() + p->num_elements());
            std::vector<float> root_copy = mine;
            ctx.broadcast(root_copy.data(), root_copy.size(), 0);
            bool identical = std::memcmp(mine.data(), root_copy.data(), mine.size() * sizeof(float)) == 0;

            // (b) held-out accuracy on a data set every rank generates identically.
            Lcg held(999999);
            int correct = 0, total = 0;
            for (int b = 0; b < 8; ++b) {
                std::shared_ptr<Tensor> X; std::vector<int> Y;
                make_batch(held, X, Y);
                auto logits = model.forward(X);
                correct += static_cast<int>(ops::accuracy(logits, Y) * kBatch + 0.5f);
                total += kBatch;
            }
            float acc = 100.0f * correct / total;

            report("weights", identical, std::to_string(mine.size()) + " params, bit-identical on all ranks");
            std::ostringstream d;
            d << std::fixed << std::setprecision(3) << "loss " << first_loss << " -> " << last_loss
              << ", held-out acc " << std::setprecision(1) << acc << "% (chance 25%)";
            report("training", last_loss < 0.8f * first_loss && acc > 60.0f, d.str());
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << R << "] COMM ERROR: " << e.what() << "\n";
        return 3;
    }

    std::cout << "[rank " << R << "] SMOKE RESULT: " << (g_all_ok ? "PASS" : "FAIL") << "\n";
    return g_all_ok ? 0 : 1;
}
