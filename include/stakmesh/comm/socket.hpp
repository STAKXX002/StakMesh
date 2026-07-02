#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// socket.hpp - minimal RAII TCP socket wrapper (POSIX)
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
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <thread>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

namespace stakmesh {
namespace comm {

class SocketError : public std::runtime_error {
public:
    explicit SocketError(const std::string& msg) : std::runtime_error(msg) {}
};

// ── TcpSocket ────────────────────────────────────────────────────────────────
// Move-only. Owns exactly one fd. Closes it on destruction.
class TcpSocket {
public:
    TcpSocket() = default;
    explicit TcpSocket(int fd) : fd_(fd) {}

    ~TcpSocket() { close_if_open(); }

    // Non-copyable, movable.
    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    TcpSocket(TcpSocket&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    TcpSocket& operator=(TcpSocket&& other) noexcept {
        if (this != &other) {
            close_if_open();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    bool valid() const { return fd_ >= 0; }
    int fd() const { return fd_; }

    // ── Client side: connect to (host, port), retrying for up to
    //    `timeout_ms` total. Retries matter for ring bootstrap: rank r+1
    //    may not have called listen() yet when rank r tries to connect. ──
    static TcpSocket connect(const std::string& host, int port,
                              int timeout_ms = 10000, int retry_delay_ms = 100) {
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
                int fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
                if (fd < 0) continue;

                if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    TcpSocket sock(fd);
                    sock.set_nodelay();
                    return sock;
                }
                last_error = std::strerror(errno);
                ::close(fd);
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
        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) throw SocketError("listen_one: socket() failed");

        int yes = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(listen_fd);
            throw SocketError("listen_one: bind() failed on port " +
                               std::to_string(port) + " - " + std::strerror(errno));
        }
        if (::listen(listen_fd, 1) < 0) {
            ::close(listen_fd);
            throw SocketError("listen_one: listen() failed");
        }

        int conn_fd = ::accept(listen_fd, nullptr, nullptr);
        ::close(listen_fd);  // don't need the listening socket once connected
        if (conn_fd < 0) {
            throw SocketError("listen_one: accept() failed - " + std::string(std::strerror(errno)));
        }

        TcpSocket sock(conn_fd);
        sock.set_nodelay();
        return sock;
    }

    // ── send_all / recv_all: loop until `len` bytes are fully transferred. ──
    void send_all(const void* data, size_t len) const {
        const char* p = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = ::send(fd_, p + sent, len - sent, 0);
            if (n <= 0) {
                throw SocketError("send_all: connection error/closed after " +
                                   std::to_string(sent) + "/" + std::to_string(len) + " bytes");
            }
            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(void* data, size_t len) const {
        char* p = static_cast<char*>(data);
        size_t received = 0;
        while (received < len) {
            ssize_t n = ::recv(fd_, p + received, len - received, 0);
            if (n <= 0) {
                throw SocketError("recv_all: connection error/closed after " +
                                   std::to_string(received) + "/" + std::to_string(len) + " bytes");
            }
            received += static_cast<size_t>(n);
        }
    }

private:
    int fd_ = -1;

    void set_nodelay() {
        int yes = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    }

    void close_if_open() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }
};

}  // namespace comm
}  // namespace stakmesh
