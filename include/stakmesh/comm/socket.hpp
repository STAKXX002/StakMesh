#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// socket.hpp - minimal RAII TCP socket wrapper (POSIX + native Windows)
//
// CONCEPT: Why hand-roll this instead of using a networking library?
//
//   Every collective op in StakMesh (ring all-reduce, broadcast, the future
//   launcher control channel) ultimately boils down to "send these bytes to
//   that rank" / "receive exactly N bytes from that rank". TCP gives us a
//   reliable, ordered byte stream but NOT message boundaries - a single
//   send() on one side can arrive as multiple recv()s on the other, or
//   several sends can coalesce into one recv(). send_all/recv_all exist to
//   paper over that: they loop until the full buffer is transferred or the
//   connection dies.
//
//   This file has zero knowledge of StakML or Tensors - it only moves raw
//   bytes. Keeping it dumb means it's reusable for control messages,
//   checkpoints, or anything else later phases need, not just gradients.
//
// CONCEPT: What's different on Windows?
//
//   Berkeley sockets (POSIX) and Winsock2 (Windows) are ~95% the same API
//   by design - Winsock was deliberately modeled on BSD sockets - but the
//   remaining 5% is exactly the kind of thing that silently breaks a build:
//     - Socket handles are `SOCKET` (unsigned), not a plain `int`; the
//       "invalid" sentinel is `INVALID_SOCKET`, not a negative number.
//     - You close a socket with `closesocket()`, not `close()` - `close()`
//       exists on Windows but means something else entirely.
//     - The whole Winsock subsystem needs one-time `WSAStartup()` before
//       ANY socket call, and `WSACleanup()` at the end - POSIX has no
//       equivalent, sockets just work.
//     - Error codes come from `WSAGetLastError()`, not `errno`.
//     - `send`/`recv` take an `int` length on Windows, not `size_t` - this
//       matters if a single chunk ever exceeded ~2GB, so send_all/recv_all
//       cap each underlying call defensively either way.
//
//   Everything else (getaddrinfo, sockaddr_in, htons, TCP_NODELAY, the
//   actual connect/bind/listen/accept/send/recv call shapes) is identical,
//   which is why this stays one file with #ifdef branches rather than two
//   parallel implementations.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <thread>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")

    using SocketHandle = SOCKET;
    constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
    using SendRecvLen = int;  // Winsock send/recv take int, not size_t
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>

    using SocketHandle = int;
    constexpr SocketHandle kInvalidSocket = -1;
    using SendRecvLen = size_t;
#endif

namespace stakmesh {
namespace comm {

class SocketError : public std::runtime_error {
public:
    explicit SocketError(const std::string& msg) : std::runtime_error(msg) {}
};

#ifdef _WIN32
// One-time WSAStartup, lazily triggered by the first socket operation.
// Meyer's singleton -> thread-safe init, WSACleanup runs at process exit.
inline void ensure_winsock_initialized() {
    struct WinsockGuard {
        WinsockGuard() {
            WSADATA wsa_data;
            if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
                throw SocketError("WSAStartup failed");
            }
        }
        ~WinsockGuard() { WSACleanup(); }
    };
    static WinsockGuard guard;
    (void)guard;
}

inline std::string last_socket_error() {
    return "WSA error " + std::to_string(WSAGetLastError());
}
#else
inline void ensure_winsock_initialized() {}  // no-op on POSIX

inline std::string last_socket_error() {
    return std::strerror(errno);
}
#endif

// ── TcpSocket ────────────────────────────────────────────────────────────────
// Move-only. Owns exactly one socket handle. Closes it on destruction.
class TcpSocket {
public:
    TcpSocket() = default;
    explicit TcpSocket(SocketHandle handle) : handle_(handle) {}

    ~TcpSocket() { close_if_open(); }

    // Non-copyable, movable.
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    TcpSocket(TcpSocket&& other) noexcept : handle_(other.handle_) {
        other.handle_ = kInvalidSocket;
    }
    TcpSocket& operator=(TcpSocket&& other) noexcept {
        if (this != &other) {
            close_if_open();
            handle_ = other.handle_;
            other.handle_ = kInvalidSocket;
        }
        return *this;
    }

    bool valid() const { return handle_ != kInvalidSocket; }
    SocketHandle handle() const { return handle_; }

    // ── Client side: connect to (host, port), retrying for up to
    //    `timeout_ms` total. Retries matter for ring bootstrap: rank r+1
    //    may not have called listen() yet when rank r tries to connect. ──
    static TcpSocket connect(const std::string& host, int port,
                              int timeout_ms = 10000, int retry_delay_ms = 100) {
        ensure_winsock_initialized();

        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        addrinfo* res = nullptr;
        std::string port_str = std::to_string(port);
        int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
        if (rc != 0) {
            throw SocketError("connect: getaddrinfo failed for " + host + ":" +
                               port_str + " - " + gai_strerror(rc));
        }

        auto deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(timeout_ms);
        std::string last_error;

        while (std::chrono::steady_clock::now() < deadline) {
            for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
                SocketHandle h = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
                if (h == kInvalidSocket) continue;

                if (::connect(h, p->ai_addr, static_cast<int>(p->ai_addrlen)) == 0) {
                    freeaddrinfo(res);
                    TcpSocket sock(h);
                    sock.set_nodelay();
                    return sock;
                }
                last_error = last_socket_error();
#ifdef _WIN32
                ::closesocket(h);
#else
                ::close(h);
#endif
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }

        freeaddrinfo(res);
        throw SocketError("connect: timed out reaching " + host + ":" + port_str +
                           " (last error: " + last_error + ")");
    }

    // ── Server side: bind to `port` on all interfaces, listen, and block
    //    until exactly one peer connects. Returns the accepted connection. ──
    static TcpSocket listen_one(int port) {
        ensure_winsock_initialized();

        SocketHandle listen_h = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_h == kInvalidSocket) throw SocketError("listen_one: socket() failed");

#ifdef _WIN32
        // SO_REUSEADDR on Windows has looser (arguably unsafe) semantics
        // than POSIX -- it can let another socket silently steal a bound
        // address. SO_EXCLUSIVEADDRUSE is the closer safety match here.
        BOOL yes = TRUE;
        ::setsockopt(listen_h, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                     reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
        int yes = 1;
        ::setsockopt(listen_h, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(listen_h, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close_handle(listen_h);
            throw SocketError("listen_one: bind() failed on port " +
                               std::to_string(port) + " - " + last_socket_error());
        }
        if (::listen(listen_h, 1) != 0) {
            close_handle(listen_h);
            throw SocketError("listen_one: listen() failed - " + last_socket_error());
        }

        SocketHandle conn_h = ::accept(listen_h, nullptr, nullptr);
        close_handle(listen_h);  // don't need the listening socket once connected
        if (conn_h == kInvalidSocket) {
            throw SocketError("listen_one: accept() failed - " + last_socket_error());
        }

        TcpSocket sock(conn_h);
        sock.set_nodelay();
        return sock;
    }

    // ── send_all / recv_all: loop until `len` bytes are fully transferred. ──
    void send_all(const void* data, size_t len) const {
        const char* p = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < len) {
            SendRecvLen chunk = clamp_chunk(len - sent);
            auto n = ::send(handle_, p + sent, chunk, 0);
            if (n <= 0) {
                throw SocketError("send_all: connection error/closed after " +
                                   std::to_string(sent) + "/" + std::to_string(len) +
                                   " bytes - " + last_socket_error());
            }
            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(void* data, size_t len) const {
        char* p = static_cast<char*>(data);
        size_t received = 0;
        while (received < len) {
            SendRecvLen chunk = clamp_chunk(len - received);
            auto n = ::recv(handle_, p + received, chunk, 0);
            if (n <= 0) {
                throw SocketError("recv_all: connection error/closed after " +
                                   std::to_string(received) + "/" + std::to_string(len) +
                                   " bytes - " + last_socket_error());
            }
            received += static_cast<size_t>(n);
        }
    }

private:
    SocketHandle handle_ = kInvalidSocket;

    // Winsock's send/recv take a signed int length -- cap each underlying
    // call so a (currently unrealistic, but not impossible for a big
    // model's flattened gradient) multi-gigabyte chunk can't overflow it.
    // send_all/recv_all's loop handles the rest transparently either way.
    static SendRecvLen clamp_chunk(size_t remaining) {
#ifdef _WIN32
        constexpr size_t kMaxChunk = 64 * 1024 * 1024;  // 64MB per syscall
        return static_cast<SendRecvLen>(std::min(remaining, kMaxChunk));
#else
        return remaining;
#endif
    }

    void set_nodelay() {
#ifdef _WIN32
        BOOL yes = TRUE;
        ::setsockopt(handle_, IPPROTO_TCP, TCP_NODELAY,
                     reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
        int yes = 1;
        ::setsockopt(handle_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
#endif
    }

    static void close_handle(SocketHandle h) {
#ifdef _WIN32
        ::closesocket(h);
#else
        ::close(h);
#endif
    }

    void close_if_open() {
        if (handle_ != kInvalidSocket) {
            close_handle(handle_);
            handle_ = kInvalidSocket;
        }
    }
};

}  // namespace comm
}  // namespace stakmesh