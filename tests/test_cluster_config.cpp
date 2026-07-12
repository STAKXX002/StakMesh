// test_cluster_config.cpp
//
// Verifies parse_cluster_config() rejects malformed input correctly, and
// detect_local_rank() correctly identifies which config entry is "this
// machine" -- including the failure modes (no match, ambiguous match)
// where guessing wrong would mean silently connecting to the wrong peer.

#include "../include/stakmesh/comm/cluster_config.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdio>

using namespace stakmesh::comm;

static int passed = 0, total = 0;

static void check(bool cond, const std::string& name) {
    ++total;
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (cond) ++passed;
}

// /tmp doesn't exist on Windows - std::filesystem::temp_directory_path()
// resolves to the right place on every platform.
static std::string temp_path(const std::string& filename) {
    return (std::filesystem::temp_directory_path() / filename).string();
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

int main() {
    // ── Well-formed parsing ──────────────────────────────────────────────
    write_file(temp_path("stakmesh_cfg_good.txt"),
               "# comment line\n"
               "0 hosta 29500\n"
               "\n"
               "1 hostb 29501  # inline comment\n");
    {
        auto peers = parse_cluster_config(temp_path("stakmesh_cfg_good.txt"));
        check(peers.size() == 2 && peers[0].host == "hosta" && peers[0].port == 29500 &&
                  peers[1].host == "hostb" && peers[1].port == 29501,
              "parses well-formed config with comments/blank lines correctly");
    }

    // ── Malformed: duplicate rank ────────────────────────────────────────
    write_file(temp_path("stakmesh_cfg_dup.txt"), "0 hosta 1\n0 hostb 2\n");
    {
        bool threw = false;
        try { parse_cluster_config(temp_path("stakmesh_cfg_dup.txt")); }
        catch (const std::exception&) { threw = true; }
        check(threw, "rejects duplicate rank");
    }

    // ── Malformed: gap in rank sequence ───────────────────────────────────
    write_file(temp_path("stakmesh_cfg_gap.txt"), "0 hosta 1\n2 hostb 2\n");
    {
        bool threw = false;
        try { parse_cluster_config(temp_path("stakmesh_cfg_gap.txt")); }
        catch (const std::exception&) { threw = true; }
        check(threw, "rejects gap in rank sequence (0, 2 with no 1)");
    }

    // ── Malformed: empty file ────────────────────────────────────────────
    write_file(temp_path("stakmesh_cfg_empty.txt"), "# just a comment\n\n");
    {
        bool threw = false;
        try { parse_cluster_config(temp_path("stakmesh_cfg_empty.txt")); }
        catch (const std::exception&) { threw = true; }
        check(threw, "rejects a config with no valid entries");
    }

    // ── Missing file ─────────────────────────────────────────────────────
    {
        bool threw = false;
        try { parse_cluster_config(temp_path("stakmesh_cfg_does_not_exist.txt")); }
        catch (const std::exception&) { threw = true; }
        check(threw, "rejects a nonexistent file path");
    }

    // ── Rank auto-detection: find THIS sandbox's real local IP first ────
    auto local_ips = local_ipv4_addresses();
    check(!local_ips.empty(), "local_ipv4_addresses() finds at least one real address");

    if (!local_ips.empty()) {
        const std::string& my_ip = local_ips[0];

        // Correct match: rank 1's host IS this machine's real IP, rank 0 is
        // a TEST-NET-3 decoy address (RFC 5737, guaranteed non-routable/
        // non-local, so it can never accidentally match).
        write_file(temp_path("stakmesh_cfg_detect.txt"),
                   "0 203.0.113.5 29500\n1 " + my_ip + " 29501\n");
        {
            auto peers = parse_cluster_config(temp_path("stakmesh_cfg_detect.txt"));
            int rank = detect_local_rank(peers);
            check(rank == 1, "detect_local_rank finds the correct matching entry (rank 1)");
        }

        // No match: every entry is a decoy -- should throw, not silently
        // default to some rank.
        write_file(temp_path("stakmesh_cfg_nomatch.txt"),
                   "0 203.0.113.5 29500\n1 203.0.113.6 29501\n");
        {
            auto peers = parse_cluster_config(temp_path("stakmesh_cfg_nomatch.txt"));
            bool threw = false;
            try { detect_local_rank(peers); }
            catch (const std::exception&) { threw = true; }
            check(threw, "detect_local_rank throws when no entry matches this machine");
        }

        // Ambiguous: TWO entries both resolve to this machine's real IP --
        // should throw rather than silently picking the first one.
        write_file(temp_path("stakmesh_cfg_ambiguous.txt"),
                   "0 " + my_ip + " 29500\n1 " + my_ip + " 29501\n");
        {
            auto peers = parse_cluster_config(temp_path("stakmesh_cfg_ambiguous.txt"));
            bool threw = false;
            try { detect_local_rank(peers); }
            catch (const std::exception&) { threw = true; }
            check(threw, "detect_local_rank throws when this machine matches multiple ranks");
        }
    }

    std::cout << passed << "/" << total << " cases passed\n";
    return (passed == total) ? 0 : 1;
}