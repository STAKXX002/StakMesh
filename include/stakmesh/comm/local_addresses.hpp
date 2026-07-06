#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// local_addresses.hpp - "what IP addresses does THIS machine have?"
//
// CONCEPT: why this is needed
//
//   Phase 3's whole point is removing the need to manually pass --rank on
//   each machine. The trick: a cluster config lists {rank, host, port} for
//   EVERY machine in the cluster, and each process can figure out "which
//   one am I" by checking which config entry's host resolves to an IP
//   address this machine actually owns. That requires being able to list
//   this machine's own addresses -- which, like sockets, is a genuinely
//   different API on POSIX (getifaddrs) vs Windows (GetAdaptersAddresses/
//   IP Helper API), hence the platform split below.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>

#include "socket.hpp"  // brings in the right platform networking headers + ensure_winsock_initialized()

#ifdef _WIN32
    #include <iphlpapi.h>
    #pragma comment(lib, "iphlpapi.lib")
#else
    #include <ifaddrs.h>
#endif

namespace stakmesh {
namespace comm {

// Returns every non-loopback IPv4 address configured on this machine,
// across ALL interfaces -- WiFi, Ethernet, and virtual adapters like
// Tailscale's (which is exactly what lets a Tailscale hostname/IP in the
// cluster config match correctly). Order is not significant.
inline std::vector<std::string> local_ipv4_addresses() {
    std::vector<std::string> addrs;

#ifdef _WIN32
    ensure_winsock_initialized();  // inet_ntop lives in ws2_32, be defensive

    ULONG buf_len = 15000;
    std::vector<char> buffer(buf_len);
    auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());

    ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
    ULONG result = GetAdaptersAddresses(AF_INET, flags, nullptr, adapters, &buf_len);
    if (result == ERROR_BUFFER_OVERFLOW) {
        buffer.resize(buf_len);
        adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
        result = GetAdaptersAddresses(AF_INET, flags, nullptr, adapters, &buf_len);
    }
    if (result != NO_ERROR) return addrs;  // best-effort: empty list rather than throwing

    for (auto* adapter = adapters; adapter != nullptr; adapter = adapter->Next) {
        for (auto* ua = adapter->FirstUnicastAddress; ua != nullptr; ua = ua->Next) {
            auto* sa = reinterpret_cast<sockaddr_in*>(ua->Address.lpSockaddr);
            char ip_str[INET_ADDRSTRLEN];
            if (inet_ntop(AF_INET, &(sa->sin_addr), ip_str, INET_ADDRSTRLEN)) {
                std::string ip(ip_str);
                if (ip != "127.0.0.1") addrs.push_back(ip);
            }
        }
    }
#else
    ifaddrs* ifap = nullptr;
    if (getifaddrs(&ifap) != 0) return addrs;

    for (ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        auto* sa = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        char ip_str[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &(sa->sin_addr), ip_str, INET_ADDRSTRLEN)) {
            std::string ip(ip_str);
            if (ip != "127.0.0.1") addrs.push_back(ip);
        }
    }
    freeifaddrs(ifap);
#endif

    return addrs;
}

}  // namespace comm
}  // namespace stakmesh