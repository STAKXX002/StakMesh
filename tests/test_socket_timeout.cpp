// test_socket_timeout.cpp
//
// Phase 4b: send_all/recv_all used to have NO timeout - a dead or silent
// peer meant blocking in recv() forever. This test proves the fix actually
// bounds the wait, and that a genuine timeout is reported distinctly from
// a peer closing the connection cleanly (two different failure modes that
// look identical without the distinction - one means "still might be
// alive, just slow/unreachable", the other means "definitely gone").

#include "../include/stakmesh/comm/socket.hpp"
#include "../include/stakmesh/comm/topology.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

using namespace stakmesh::comm;
using Clock = std::chrono::steady_clock;

static bool g_all_ok = true;

static void check(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "[PASS] " << what << "\n";
    } else {
        std::cout << "[FAIL] " << what << "\n";
        g_all_ok = false;
    }
}

// A silent peer: accepts the connection, then sends nothing and holds the
// socket open well past the client's timeout. This is what a hung/dead
// process or a dropped network path looks like from the other side - the
// TCP connection is still technically "up", nothing ever arrives.
static void test_recv_times_out_on_silent_peer() {
    const int port = 18100;
    const int io_timeout_ms = 500;

    // (no synchronization needed - connect() already retries internally,
    // which absorbs the race of the client trying before the server has
    // bound its listening socket yet)
    std::thread server([&]() {
        TcpSocket s = TcpSocket::listen_one(port, io_timeout_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(io_timeout_ms * 4));
        // socket closes here when `s` goes out of scope
    });

    TcpSocket client = TcpSocket::connect("127.0.0.1", port, 5000, 50, io_timeout_ms);

    char buf[16];
    auto start = Clock::now();
    bool threw = false;
    std::string what;
    try {
        client.recv_all(buf, sizeof(buf));
    } catch (const SocketError& e) {
        threw = true;
        what = e.what();
    }
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();

    check(threw, "recv_all on a silent peer throws (doesn't hang forever)");
    // Generous bounds: must wait roughly io_timeout_ms, not return instantly
    // and not wait for the full 4x sleep the server thread does.
    check(elapsed_ms >= io_timeout_ms - 100 && elapsed_ms < io_timeout_ms * 3,
          "recv_all's wait is bounded by io_timeout_ms (~" + std::to_string(io_timeout_ms) +
          "ms), took " + std::to_string(elapsed_ms) + "ms");
    check(what.find("timed out") != std::string::npos,
          "error message says 'timed out' specifically, not a generic error: \"" + what + "\"");

    server.join();
}

// A peer that closes cleanly (no crash, just done) should be reported
// distinctly from a timeout, AND should return near-instantly - a clean
// FIN arrives whenever it arrives, it doesn't wait for io_timeout_ms.
static void test_recv_reports_clean_close_distinctly() {
    const int port = 18101;
    const int io_timeout_ms = 5000;  // deliberately long - close should still be fast

    // (no synchronization needed - connect() already retries internally,
    // which absorbs the race of the client trying before the server has
    // bound its listening socket yet)
    std::thread server([&]() {
        TcpSocket s = TcpSocket::listen_one(port, io_timeout_ms);
        // immediately falls out of scope -> closes the connection cleanly
    });

    TcpSocket client = TcpSocket::connect("127.0.0.1", port, 5000, 50, io_timeout_ms);

    char buf[16];
    auto start = Clock::now();
    bool threw = false;
    std::string what;
    try {
        client.recv_all(buf, sizeof(buf));
    } catch (const SocketError& e) {
        threw = true;
        what = e.what();
    }
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();

    check(threw, "recv_all on a cleanly-closed peer throws");
    check(elapsed_ms < 1000,
          "clean close is reported almost immediately (" + std::to_string(elapsed_ms) +
          "ms), not held up until io_timeout_ms (" + std::to_string(io_timeout_ms) + "ms)");
    check(what.find("closed the connection cleanly") != std::string::npos,
          "error message distinguishes clean close from a timeout: \"" + what + "\"");

    server.join();
}

// io_timeout_ms <= 0 must preserve the old block-forever behavior exactly
// (an intentional escape hatch, e.g. for debugging under a slow debugger).
// This test can't prove "waits forever" directly without hanging the test
// suite, so it proves the weaker but sufficient property: with timeout
// disabled, a wait that WOULD have timed out at 200ms does not.
static void test_zero_timeout_disables_it() {
    const int port = 18102;

    // (no synchronization needed - connect() already retries internally,
    // which absorbs the race of the client trying before the server has
    // bound its listening socket yet)
    std::thread server([&]() {
        TcpSocket s = TcpSocket::listen_one(port, /*io_timeout_ms=*/0);
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        uint8_t byte = 42;
        s.send_all(&byte, 1);
    });

    TcpSocket client = TcpSocket::connect("127.0.0.1", port, 5000, 50, /*io_timeout_ms=*/0);

    uint8_t buf = 0;
    bool threw = false;
    try {
        client.recv_all(&buf, 1);
    } catch (const SocketError&) {
        threw = true;
    }

    check(!threw, "io_timeout_ms=0 disables the timeout - a 600ms-delayed reply still arrives fine");
    check(buf == 42, "and the actual byte sent is the byte received");

    server.join();
}

// Regression test for a real crash found while testing this feature:
// establish_ring() used to call std::terminate() (not throw a catchable
// exception) whenever the outgoing connect() failed, because a background
// std::thread was still joinable when the function exited via exception -
// a joinable thread's destructor calling terminate() during unwinding is
// standard (if brutal) C++ behavior. Confirmed via gdb backtrace before
// fixing: frame right at establish_ring calling std::terminate directly.
static void test_unreachable_peer_fails_cleanly_not_terminate() {
    // Rank 0's "next" peer (rank 1) is a port nothing is listening on -
    // connect() will exhaust its retries and throw. A short
    // connect_timeout_ms keeps this test fast; the crash this guards
    // against doesn't depend on how long the wait was, only on what
    // happens once connect() actually throws.
    const std::vector<PeerAddr> peers = {
        {"127.0.0.1", 18110},  // rank 0 - this process, listens but nothing
                                //          connects to it before we bail out
        {"127.0.0.1", 18199},  // rank 1 - nobody home, connect() will fail
    };

    bool threw_cleanly = false;
    std::string what;
    try {
        establish_ring(/*my_rank=*/0, peers, /*io_timeout_ms=*/1000, /*connect_timeout_ms=*/500);
    } catch (const std::exception& e) {
        threw_cleanly = true;
        what = e.what();
    }
    // If this test hangs or the process aborts instead of reaching either
    // branch below, that IS the bug (or a regression of it) - ctest will
    // report it as a crash/timeout, which is exactly the failure mode
    // being guarded against.
    check(threw_cleanly, "establish_ring on an unreachable peer throws a normal "
                          "C++ exception (not std::terminate/abort)");
    check(what.find("timed out") != std::string::npos,
          "and the exception message is the expected connect-timeout error: \"" + what + "\"");
}

int main() {
    test_recv_times_out_on_silent_peer();
    test_recv_reports_clean_close_distinctly();
    test_zero_timeout_disables_it();
    test_unreachable_peer_fails_cleanly_not_terminate();

    if (g_all_ok) {
        std::cout << "\nAll socket timeout tests passed.\n";
        return 0;
    } else {
        std::cout << "\nSome socket timeout tests FAILED.\n";
        return 1;
    }
}
