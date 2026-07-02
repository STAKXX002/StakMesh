// test_ring_allreduce.cpp
//
// Spins up `world_size` ranks as threads on localhost (127.0.0.1), each with
// its own port, wires them into a ring via establish_ring(), gives each rank
// a distinct synthetic buffer, and checks that ring_all_reduce() produces the
// correct elementwise sum/average on every rank.
//
// Deliberately uses a buffer size NOT evenly divisible by world_size, since
// that's exactly where off-by-one chunk-bound bugs hide.

#include "../include/stakmesh/comm/socket.hpp"
#include "../include/stakmesh/comm/topology.hpp"
#include "../include/stakmesh/comm/ring_allreduce.hpp"

#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <atomic>

using namespace stakmesh::comm;

static bool run_case(int world_size, size_t n, ReduceOp op, int base_port) {
    std::vector<PeerAddr> peers;
    for (int i = 0; i < world_size; ++i) {
        peers.push_back({"127.0.0.1", base_port + i});
    }

    // reference[i] = sum over all ranks of that rank's value at index i
    std::vector<float> reference(n, 0.0f);
    std::vector<std::vector<float>> initial(world_size, std::vector<float>(n));
    for (int r = 0; r < world_size; ++r) {
        for (size_t i = 0; i < n; ++i) {
            float v = static_cast<float>(r * 100 + static_cast<int>(i));
            initial[r][i] = v;
            reference[i] += v;
        }
    }
    if (op == ReduceOp::Average) {
        for (auto& v : reference) v /= static_cast<float>(world_size);
    }

    std::vector<std::vector<float>> results(world_size);
    std::vector<std::string> errors(world_size);
    std::atomic<bool> all_ok{true};

    std::vector<std::thread> workers;
    for (int r = 0; r < world_size; ++r) {
        workers.emplace_back([&, r]() {
            try {
                RingLinks links = establish_ring(r, peers);
                std::vector<float> buf = initial[r];
                ring_all_reduce(buf.data(), buf.size(), r, world_size,
                                 links.send_sock, links.recv_sock, op);
                results[r] = buf;
            } catch (const std::exception& e) {
                errors[r] = e.what();
                all_ok = false;
            }
        });
    }
    for (auto& t : workers) t.join();

    if (!all_ok) {
        for (int r = 0; r < world_size; ++r) {
            if (!errors[r].empty()) {
                std::cerr << "  rank " << r << " threw: " << errors[r] << "\n";
            }
        }
        return false;
    }

    bool match = true;
    for (int r = 0; r < world_size; ++r) {
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(results[r][i] - reference[i]) > 1e-3f) {
                std::cerr << "  MISMATCH rank " << r << " idx " << i
                          << " got " << results[r][i] << " want " << reference[i] << "\n";
                match = false;
            }
        }
    }
    return match;
}

int main() {
    struct Case { int world_size; size_t n; ReduceOp op; int base_port; const char* name; };
    std::vector<Case> cases = {
        {2, 10, ReduceOp::Sum,     15000, "2 ranks, n=10 (even), sum"},
        {2, 13, ReduceOp::Average, 15010, "2 ranks, n=13 (uneven), average -- matches your laptop pair"},
        {3, 10, ReduceOp::Sum,     15020, "3 ranks, n=10 (uneven, remainder=1), sum"},
        {4, 7,  ReduceOp::Average, 15030, "4 ranks, n=7 (n < world_size*2, tight), average"},
        {5, 1,  ReduceOp::Sum,     15040, "5 ranks, n=1 (degenerate, single element), sum"},
    };

    int passed = 0;
    for (auto& c : cases) {
        bool ok = run_case(c.world_size, c.n, c.op, c.base_port);
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << c.name << "\n";
        if (ok) ++passed;
    }

    std::cout << passed << "/" << cases.size() << " cases passed\n";
    return passed == static_cast<int>(cases.size()) ? 0 : 1;
}
