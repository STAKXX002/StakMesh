// test_broadcast.cpp
//
// Verifies ring_broadcast(): root's buffer ends up identical on every rank,
// non-root ranks' original (different) data gets overwritten correctly.

#include "../include/stakmesh/comm/socket.hpp"
#include "../include/stakmesh/comm/topology.hpp"
#include "../include/stakmesh/comm/broadcast.hpp"

#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <atomic>

using namespace stakmesh::comm;

static bool run_case(int world_size, size_t n, int root, int base_port) {
    std::vector<PeerAddr> peers;
    for (int i = 0; i < world_size; ++i) peers.push_back({"127.0.0.1", base_port + i});

    std::vector<float> root_data(n);
    for (size_t i = 0; i < n; ++i) root_data[i] = static_cast<float>(i) * 1.5f;

    std::vector<std::vector<float>> results(world_size);
    std::vector<std::string> errors(world_size);
    std::atomic<bool> all_ok{true};

    std::vector<std::thread> workers;
    for (int r = 0; r < world_size; ++r) {
        workers.emplace_back([&, r]() {
            try {
                RingLinks links = establish_ring(r, peers);
                // Each non-root rank starts with DIFFERENT junk data, to prove
                // broadcast actually overwrites it rather than coincidentally
                // matching.
                std::vector<float> buf(n);
                for (size_t i = 0; i < n; ++i)
                    buf[i] = (r == root) ? root_data[i] : static_cast<float>(r * 1000 + i);

                ring_broadcast(buf.data(), n, r, world_size, root, links.send_sock, links.recv_sock);
                results[r] = buf;
            } catch (const std::exception& e) {
                errors[r] = e.what();
                all_ok = false;
            }
        });
    }
    for (auto& t : workers) t.join();

    if (!all_ok) {
        for (int r = 0; r < world_size; ++r)
            if (!errors[r].empty()) std::cerr << "  rank " << r << " threw: " << errors[r] << "\n";
        return false;
    }

    bool match = true;
    for (int r = 0; r < world_size; ++r) {
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(results[r][i] - root_data[i]) > 1e-4f) {
                std::cerr << "  MISMATCH rank " << r << " idx " << i
                          << " got " << results[r][i] << " want " << root_data[i] << "\n";
                match = false;
            }
        }
    }
    return match;
}

int main() {
    struct Case { int world_size; size_t n; int root; int base_port; const char* name; };
    std::vector<Case> cases = {
        {2, 16, 0, 15100, "2 ranks, root=0 -- matches your laptop pair"},
        {2, 16, 1, 15110, "2 ranks, root=1 (non-zero root)"},
        {3, 9,  0, 15120, "3 ranks, root=0"},
        {4, 5,  2, 15130, "4 ranks, root=2 (mid-ring root)"},
    };

    int passed = 0;
    for (auto& c : cases) {
        bool ok = run_case(c.world_size, c.n, c.root, c.base_port);
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << c.name << "\n";
        if (ok) ++passed;
    }
    std::cout << passed << "/" << cases.size() << " cases passed\n";
    return passed == static_cast<int>(cases.size()) ? 0 : 1;
}
