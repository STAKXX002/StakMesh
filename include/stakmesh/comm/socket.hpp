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
//
// CONCEPT: Why does every socket get an I/O timeout, not just connect()?
//
//   connect() already retried/timed out on its own - that only covers ring
//   BOOTSTRAP, before any training happens. Once the ring is up, send_all/
//   recv_all are called every single batch for the rest of the run, and
//   until Phase 4b they had NO timeout at all: if a peer died or the
//   network dropped mid-training, the surviving rank would block in
//   recv() forever - no error, no way to know something's wrong, just a
//   process that looks alive but will never make progress again.
//
//   SO_RCVTIMEO/SO_SNDTIMEO turn that into a bounded wait: past
//   `io_timeout_ms` (default 30s - generous relative to the ~0.3s/batch
//   worst case measured on bad WiFi, but still a real bound), send_all/
//   recv_all throw a SocketError that says "timed out", distinct from
//   "peer closed the connection cleanly" or another kind of error - useful
//   for telling "the network is being slow" apart from "that machine is
//   actually gone" when you're staring at a crashed training run later.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cerrno>

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
    #include <poll.h>
    #include <fcntl.h>

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

inline bool last_error_was_timeout() {
    return WSAGetLastError() == WSAETIMEDOUT;
}

inline bool last_error_was_interrupt() {
    return WSAGetLastError() == WSAEINTR;
}
#else
inline void ensure_winsock_initialized() {}  // no-op on POSIX

inline std::string last_socket_error() {
    return std::strerror(errno);
}

inline bool last_error_was_timeout() {
    return errno == EAGAIN || errno == EWOULDBLOCK;
}

inline bool last_error_was_interrupt() {
    return errno == EINTR;
}
#endif

// send() flags. On Linux, writing to a socket whose peer already closed
// raises SIGPIPE by default, which silently KILLS the process (exit 141, no
// message) instead of surfacing as an error we can report. MSG_NOSIGNAL turns
// that into a normal EPIPE failure -> SocketError. (Windows has no SIGPIPE;
// macOS lacks MSG_NOSIGNAL and uses SO_NOSIGPIPE, set per-socket below.)
#if defined(MSG_NOSIGNAL) && !defined(_WIN32)
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

// Waits until `h` is readable (e.g. a pending connection on a listening
// socket) or `timeout_ms` elapses. Returns true if readable, false on timeout.
inline bool wait_readable(SocketHandle h, int timeout_ms) {
#ifdef _WIN32
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(h, &rfds);
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    return ::select(0, &rfds, nullptr, nullptr, &tv) > 0;
#else
    pollfd pfd{};
    pfd.fd = h;
    pfd.events = POLLIN;
    int rc;
    do { rc = ::poll(&pfd, 1, timeout_ms); } while (rc < 0 && errno == EINTR);
    return rc > 0;
#endif
}

// One connect() attempt bounded by `timeout_ms`. A plain blocking connect()
// to a host that silently DROPS packets (a firewall rule - e.g. Windows
// treating a network as "Public" - or a peer that is simply off) ignores our
// retry deadline and blocks for the OS default (~2 minutes on Linux, ~20 s on
// Windows). Non-blocking connect + a bounded wait keeps the deadline honest.
// On failure returns false and fills `err`.
inline bool connect_bounded(SocketHandle h, const sockaddr* addr, int addrlen,
                             int timeout_ms, std::string& err) {
#ifdef _WIN32
    u_long nonblocking = 1;
    ::ioctlsocket(h, FIONBIO, &nonblocking);
    if (::connect(h, addr, addrlen) != 0) {
        if (WSAGetLastError() != WSAEWOULDBLOCK) { err = last_socket_error(); return false; }
        fd_set wfds, efds;
        FD_ZERO(&wfds); FD_ZERO(&efds);
        FD_SET(h, &wfds); FD_SET(h, &efds);   // Windows reports a failed connect via exceptfds
        timeval tv{};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        int rc = ::select(0, nullptr, &wfds, &efds, &tv);
        if (rc == 0) { err = "connect attempt timed out (no response - firewall drop or host down?)"; return false; }
        int so_err = 0, len = sizeof(so_err);
        ::getsockopt(h, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so_err), &len);
        if (rc < 0 || so_err != 0 || FD_ISSET(h, &efds)) {
            err = "WSA error " + std::to_string(so_err != 0 ? so_err : WSAGetLastError());
            return false;
        }
    }
    nonblocking = 0;
    ::ioctlsocket(h, FIONBIO, &nonblocking);
    return true;
#else
    int flags = ::fcntl(h, F_GETFL, 0);
    ::fcntl(h, F_SETFL, flags | O_NONBLOCK);
    if (::connect(h, addr, static_cast<socklen_t>(addrlen)) != 0) {
        if (errno != EINPROGRESS && errno != EINTR) { err = last_socket_error(); return false; }
        pollfd pfd{};
        pfd.fd = h;
        pfd.events = POLLOUT;
        int rc;
        do { rc = ::poll(&pfd, 1, timeout_ms); } while (rc < 0 && errno == EINTR);
        if (rc == 0) { err = "connect attempt timed out (no response - firewall drop or host down?)"; return false; }
        int so_err = 0;
        socklen_t len = sizeof(so_err);
        ::getsockopt(h, SOL_SOCKET, SO_ERROR, &so_err, &len);
        if (rc < 0 || so_err != 0) { err = std::strerror(so_err != 0 ? so_err : errno); return false; }
    }
    ::fcntl(h, F_SETFL, flags);   // back to blocking for send_all/recv_all
    return true;
#endif
}

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
                              int timeout_ms = 10000, int retry_delay_ms = 100,
                              int io_timeout_ms = 30000) {
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

                // Per-attempt cap: never wait past the overall deadline, and
                // re-try a silent (dropped-SYN) target every few seconds
                // rather than sitting on one attempt.
                auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now()).count();
                int attempt_ms = static_cast<int>(std::max<long long>(1, std::min<long long>(remaining_ms, 3000)));
                std::string attempt_err;
                if (connect_bounded(h, p->ai_addr, static_cast<int>(p->ai_addrlen),
                                     attempt_ms, attempt_err)) {
                    freeaddrinfo(res);
                    TcpSocket sock(h);
                    sock.set_nodelay();
                    sock.set_no_sigpipe();
                    sock.set_io_timeout(io_timeout_ms);
                    return sock;
                }
                last_error = attempt_err;
                close_handle(h);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }

        freeaddrinfo(res);
        throw SocketError("connect: timed out reaching " + host + ":" + port_str +
                           " (last error: " + last_error + ")");
    }

    // ── Server side: bind to `port` on all interfaces, listen, and block
    //    until exactly one peer connects. Returns the accepted connection. ──
    //    `accept_timeout_ms` bounds the wait for that peer (0 = wait forever,
    //    the original behavior): a rank whose predecessor never shows up
    //    should fail with an error, not sit in accept() indefinitely. ──
    static TcpSocket listen_one(int port, int io_timeout_ms = 30000, int accept_timeout_ms = 0) {
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

        if (accept_timeout_ms > 0 && !wait_readable(listen_h, accept_timeout_ms)) {
            close_handle(listen_h);
            throw SocketError("listen_one: no peer connected to port " + std::to_string(port) +
                               " within " + std::to_string(accept_timeout_ms) +
                               " ms (is the previous rank running? firewall blocking inbound?)");
        }

        SocketHandle conn_h = ::accept(listen_h, nullptr, nullptr);
        close_handle(listen_h);  // don't need the listening socket once connected
        if (conn_h == kInvalidSocket) {
            throw SocketError("listen_one: accept() failed - " + last_socket_error());
        }

        TcpSocket sock(conn_h);
        sock.set_nodelay();
        sock.set_no_sigpipe();
        sock.set_io_timeout(io_timeout_ms);
        return sock;
    }

    // ── send_all / recv_all: loop until `len` bytes are fully transferred,
    //    or the socket's I/O timeout (set in connect()/listen_one()) fires. ──
    void send_all(const void* data, size_t len) const {
        const char* p = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < len) {
            SendRecvLen chunk = clamp_chunk(len - sent);
            auto n = ::send(handle_, p + sent, chunk, kSendFlags);
            if (n < 0 && last_error_was_interrupt()) continue;   // signal landed mid-call: retry
            if (n < 0 && last_error_was_timeout()) {
                throw SocketError("send_all: timed out after " + std::to_string(sent) + "/" +
                                   std::to_string(len) + " bytes - peer may be dead, or the "
                                   "network is down/too slow (see io_timeout_ms)");
            }
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
            if (n < 0 && last_error_was_interrupt()) continue;   // signal landed mid-call: retry
            if (n == 0) {
                throw SocketError("recv_all: peer closed the connection cleanly after " +
                                   std::to_string(received) + "/" + std::to_string(len) +
                                   " bytes");
            }
            if (n < 0 && last_error_was_timeout()) {
                throw SocketError("recv_all: timed out waiting for data after " +
                                   std::to_string(received) + "/" + std::to_string(len) +
                                   " bytes - peer may be dead, or the network is down/too "
                                   "slow (see io_timeout_ms)");
            }
            if (n < 0) {
                throw SocketError("recv_all: connection error after " +
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

    void set_no_sigpipe() {
#ifdef SO_NOSIGPIPE   // macOS/BSD: per-socket equivalent of MSG_NOSIGNAL
        int yes = 1;
        ::setsockopt(handle_, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    }

    // timeout_ms <= 0 means "no timeout" (block forever, the old default
    // behavior) - an intentional escape hatch, not just an oversight.
    void set_io_timeout(int timeout_ms) {
        if (timeout_ms <= 0) return;
#ifdef _WIN32
        DWORD tv = static_cast<DWORD>(timeout_ms);
        ::setsockopt(handle_, SOL_SOCKET, SO_RCVTIMEO,
                     reinterpret_cast<const char*>(&tv), sizeof(tv));
        ::setsockopt(handle_, SOL_SOCKET, SO_SNDTIMEO,
                     reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
        struct timeval tv{};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ::setsockopt(handle_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(handle_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
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