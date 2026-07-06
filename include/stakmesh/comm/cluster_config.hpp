#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cluster_config.hpp - parse a static "rank host port" file and figure out
// which rank THIS process is, automatically.
//
// CONCEPT: what problem this solves
//
//   Up to now, starting a training run meant hand-typing the full peer
//   list AND remembering which --rank belongs on which machine -- easy to
//   typo, and easy to run the wrong rank on the wrong laptop by mistake.
//
//   The fix: write the cluster topology down ONCE, in a config file (see
//   configs/two_laptop_cluster.txt), and run the EXACT SAME command on
//   every machine. Each process resolves every entry's host to an IP
//   address, checks which one matches an IP THIS machine actually owns
//   (via local_addresses.hpp), and that's its rank. No --rank needed
//   unless you want to override auto-detection for some reason (testing,
//   running two ranks on one machine, etc.).
//
// FORMAT (one line per rank):
//
//   0 stakxx002-linux 29500
//   1 stakxx002-win   29500
//
//   Blank lines and anything after a '#' are ignored. Host can be a raw IP
//   or any resolvable hostname -- including Tailscale MagicDNS names,
//   which is what makes this robust to Tailscale IPs changing (rare, but
//   possible) or to switching networks, since the hostname stays stable
//   even if the underlying IP doesn't.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "topology.hpp"         // PeerAddr
#include "local_addresses.hpp"
#include "socket.hpp"           // getaddrinfo/inet_ntop, already platform-guarded there

namespace stakmesh {
namespace comm {

// Parses the config file into an ordered peer list where index == rank.
// Throws on any malformed entry, duplicate rank, gap in the rank sequence,
// or empty file -- fail loudly at startup rather than silently connecting
// to the wrong machine.
inline std::vector<PeerAddr> parse_cluster_config(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("parse_cluster_config: cannot open " + path);

    std::vector<std::pair<int, PeerAddr>> entries;
    std::string line;
    int max_rank = -1;

    while (std::getline(file, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        std::istringstream iss(line);
        int rank;
        std::string host;
        int port;
        if (!(iss >> rank >> host >> port)) continue;  // blank/comment-only line

        if (rank < 0) {
            throw std::runtime_error("parse_cluster_config: negative rank in " + path);
        }
        entries.emplace_back(rank, PeerAddr{host, port});
        max_rank = std::max(max_rank, rank);
    }

    if (entries.empty()) {
        throw std::runtime_error("parse_cluster_config: no valid entries found in " + path);
    }

    std::vector<PeerAddr> peers_by_rank(static_cast<size_t>(max_rank) + 1, PeerAddr{"", -1});
    for (auto& entry : entries) {
        int rank = entry.first;
        const PeerAddr& addr = entry.second;
        if (peers_by_rank[static_cast<size_t>(rank)].port != -1) {
            throw std::runtime_error("parse_cluster_config: duplicate rank " + std::to_string(rank) +
                                      " in " + path);
        }
        peers_by_rank[static_cast<size_t>(rank)] = addr;
    }
    for (size_t r = 0; r < peers_by_rank.size(); ++r) {
        if (peers_by_rank[r].port == -1) {
            throw std::runtime_error("parse_cluster_config: missing entry for rank " +
                                      std::to_string(r) + " in " + path +
                                      " (ranks must be contiguous starting at 0)");
        }
    }

    return peers_by_rank;
}

// Resolves `host` (raw IP or hostname, e.g. a Tailscale MagicDNS name) to
// its IPv4 address(es). Returns an empty list rather than throwing if
// resolution fails -- an unresolvable entry just won't match anything in
// detect_local_rank(), which reports that clearly on its own.
inline std::vector<std::string> resolve_hostname(const std::string& host) {
    std::vector<std::string> ips;

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0) return ips;

    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        auto* sa = reinterpret_cast<sockaddr_in*>(p->ai_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &(sa->sin_addr), buf, INET_ADDRSTRLEN)) {
            ips.emplace_back(buf);
        }
    }
    freeaddrinfo(res);
    return ips;
}

// Figures out which entry in `peers` is THIS machine, by resolving every
// entry's host and checking for a match against this machine's own local
// addresses. Throws if zero entries match (this machine isn't in the
// cluster config, or the host didn't resolve) or if MORE than one matches
// (genuinely ambiguous -- e.g. testing two ranks against localhost on one
// machine) so the caller can fall back to an explicit --rank instead of
// silently guessing wrong.
inline int detect_local_rank(const std::vector<PeerAddr>& peers) {
    auto local_ips = local_ipv4_addresses();

    int found_rank = -1;
    for (size_t r = 0; r < peers.size(); ++r) {
        auto host_ips = resolve_hostname(peers[r].host);
        bool matches = std::any_of(host_ips.begin(), host_ips.end(), [&](const std::string& ip) {
            return std::find(local_ips.begin(), local_ips.end(), ip) != local_ips.end();
        });

        if (matches) {
            if (found_rank != -1) {
                throw std::runtime_error(
                    "detect_local_rank: this machine matches BOTH rank " +
                    std::to_string(found_rank) + " and rank " + std::to_string(r) +
                    " -- ambiguous, pass --rank explicitly to disambiguate");
            }
            found_rank = static_cast<int>(r);
        }
    }

    if (found_rank == -1) {
        throw std::runtime_error(
            "detect_local_rank: none of this machine's IP addresses match any entry "
            "in the cluster config -- check the config file's hosts, or pass --rank explicitly");
    }
    return found_rank;
}

}  // namespace comm
}  // namespace stakmesh