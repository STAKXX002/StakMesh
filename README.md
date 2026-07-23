# StakMesh

StakMesh is a CPU only distributed training system, built from scratch over
raw TCP sockets, and demonstrated on a real two node cluster of two laptops.

StakMesh is the distributed systems layer for
[StakML](https://github.com/STAKXX002/StakML), a tensor and autograd library
also built from scratch. StakML answers the question of how to compute a
gradient. StakMesh answers the question of how multiple machines agree on
one.

## Purpose

Frameworks such as PyTorch DDP, Horovod, and JAX pmap hide a common primitive
behind a single API call: ring all reduce. This project implements that
primitive directly, using raw POSIX sockets, a hand rolled wire protocol, and
the reduce scatter and all gather algorithm, then connects it to a real
autograd engine so that gradients computed on two separate physical machines
become identical after every training step.

The system runs entirely on CPUs. The core engineering problems addressed
here are networking, concurrency, and coordination.

## Architecture

```
comm/                               StakML agnostic. Moves raw float* over TCP. Reusable for gradients, checkpoints, and control messages.
  socket.hpp                        RAII TCP wrapper: connect, listen, send_all, recv_all
  topology.hpp                      Static peer list converted into a live ring (send_sock, recv_sock)
  ring_allreduce.hpp                Reduce scatter and all gather over the ring
  broadcast.hpp                     Distributes one rank's buffer to every rank (used for initial weights)
  local_addresses.hpp               This machine's own IPv4 addresses (POSIX and Windows)
  cluster_config.hpp                Parses a "rank host port" file and auto detects the local rank

dist/                               The layer that knows about StakML.
  distributed_context.hpp           Converts model.parameters() into a broadcast (once) and an all reduce (every step)
```

The design mirrors how StakML added its CUDA backend. The new layer plugs in
without modifying `Tensor`, `ops`, `nn`, or `optim`. A full distributed
training loop requires only two new lines compared to a single machine
version:

```cpp
auto ctx = stakmesh::dist::DistributedContext::init(my_rank, peers);
ctx.broadcast_parameters(model.parameters());   // once, before training starts

// standard training loop
loss->backward();
ctx.sync_gradients(model.parameters());   // the only new line per step
opt.step();
opt.zero_grad();
```

Every rank starts from identical weights after the broadcast, computes
gradients on its own data shard, and calls `sync_gradients()` to average
those gradients across all ranks in place. Every rank then runs the same
unmodified `opt.step()`, so weights remain identical across the cluster
without any distributed aware code inside StakML itself.

## Project status

- [x] **Phase 1, collective communication library.** Ring all reduce over
      raw TCP, verified against a reference implementation across 2, 3, 4,
      and 5 simulated ranks, including uneven chunk sizes and degenerate
      cases (`tests/test_ring_allreduce.cpp`). Ring broadcast for
      distributing initial weights, verified the same way, including a
      non zero, mid ring root (`tests/test_broadcast.cpp`).
- [x] **`DistributedContext` verified end to end** against real StakML
      `nn::Linear` and `optim::SGD`. Two ranks with different random initial
      weights converge to bit identical weights after one
      `broadcast_parameters()` call, and remain identical across 25
      consecutive `sync_gradients()` and `step()` calls on the same
      connection, matching the shape of a real training loop
      (`tests/test_distributed_integration.cpp`,
      `tests/test_distributed_training_loop.cpp`).
- [x] **Phase 2, real data parallel training script**
      (`examples/mnist_distributed.cpp`). Reuses StakML's own MNIST loader
      and model architecture, shards the training set contiguously by rank,
      broadcasts initial weights, and syncs gradients every batch. Compiles
      and links cleanly against real StakML headers.
- [x] **Phase 3, cluster configuration and automatic rank detection.** The
      cluster topology is written once, in
      `configs/two_laptop_cluster.txt`, and the same command runs on every
      machine. Each process resolves every entry's host and detects its own
      rank by matching against its own IP addresses
      (`tests/test_cluster_config.cpp`, 9 of 9 tests passing, including both
      failure modes of no match and ambiguous match). `--rank` and `--peers`
      remain available for manual overrides. Remote launch from one
      terminal (`scripts/launch_cluster.sh`) starts every rank, local
      directly and remote over SSH, from a single command, with tagged and
      color coded live output. Ctrl+C stops all ranks. See "Launching from
      one terminal" below.
- [x] **Phase 4a, checkpointing.** Each rank saves its own weights, Adam
      moment state (m_, v_, t_), and epoch number locally every N epochs,
      and `--resume` continues training from that point. Shape and
      optimizer type mismatches are checked before any data is touched and
      raise a clear error rather than loading invalid state. Verified at
      the object level (save into one model and optimizer, load into a
      fresh one, take one more step, and confirm the result matches
      continuing the original, `tests/test_checkpoint.cpp`, 13 of 13 tests
      passing) and end to end over a real two rank TCP run (train two
      epochs, restart both processes, resume, and confirm both ranks stay
      in sync exactly as they would in an uninterrupted run). See
      "Checkpointing and resuming" below.
- [x] **Phase 4b, fault detection.** `send_all` and `recv_all` now carry a
      configurable I/O timeout, thirty seconds by default, applied through
      `SO_RCVTIMEO` and `SO_SNDTIMEO` once the ring is established. A dead
      or silent peer now fails with a clear timeout error instead of
      hanging indefinitely, and this is distinguished from a peer closing
      the connection cleanly (`tests/test_socket_timeout.cpp`, 10 of 10
      tests passing).

      This work also surfaced and fixed a real crash. An unreachable peer
      during the initial ring bootstrap previously called
      `std::terminate()` directly, confirmed through a gdb backtrace,
      instead of raising a catchable exception. The cause was a background
      `std::thread` that remained joinable when `establish_ring()` exited
      through an exception, and a joinable thread whose destructor runs
      during exception unwinding calls `terminate()` by design in standard
      C++. The fix catches exceptions inside the thread, so nothing escapes
      the thread boundary, and calls `detach()` on the failure path instead
      of `join()`, since the listening side has no timeout on `accept()`
      itself, only on already established connections.
- [ ] **Phase 5, bandwidth aware optimizations.** Gradient compression and
      quantization, motivated by the real constraint of home WiFi and
      Ethernet bandwidth.
- [ ] **Phase 6, stretch goal, inference serving layer.** Request batching
      across the cluster once training is complete.

## Building

Requires CMake 3.28 or later, a C++17 compiler, and pthreads. StakML is
pulled automatically through `FetchContent`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

When developing StakML and StakMesh together, point at a local checkout
instead of fetching from GitHub each time:

```bash
cmake .. -DSTAKMESH_STAKML_SOURCE_DIR=/path/to/your/StakML
```

## Running the distributed training example

The standard MNIST IDX files are required (`train-images-idx3-ubyte`,
`train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`,
`t10k-labels-idx1-ubyte`), placed at the same relative path on both
machines. The default path is `../data` next to the build directory, and
this can be overridden with `--data-dir`.

### Recommended: configuration mode (Phase 3)

Edit `configs/two_laptop_cluster.txt` with the real addresses to use,
whether Tailscale IPs, hostnames, or LAN IPs, then run the same command on
every machine:

```bash
./mnist_distributed --config ../configs/two_laptop_cluster.txt
```

Each process determines its own rank automatically by checking which
configuration entry matches an IP address it owns. If no entry matches, or
more than one does, for example when testing on a single machine, the
process fails immediately with a clear error. Pass `--rank <N>` explicitly
to override automatic detection in that situation.

### Manual mode

```bash
# On laptop A:
./mnist_distributed --rank 0 --peers 192.168.1.10:29500,192.168.1.11:29500

# On laptop B:
./mnist_distributed --rank 1 --peers 192.168.1.10:29500,192.168.1.11:29500
```

### Requirements for either mode

Both machines must be able to reach each other on the chosen port, over the
same network or over Tailscale, with the port open on both sides. Check
the Windows firewall if the connection hangs. Each rank trains on its own
shard of the dataset and prints its own loss and accuracy per epoch. Since
gradients sync every batch, both ranks' models remain identical throughout
training even though neither rank sees the other's data.

## Checkpointing and resuming

```bash
# Every N epochs, every epoch by default, each rank saves its own weights
# and Adam state locally. This is on by default and needs no flag:
./mnist_distributed --config ../configs/two_laptop_cluster.local.txt --epochs 10

# If training stops for any reason after a few epochs, run the same
# command on every machine with --resume added:
./mnist_distributed --config ../configs/two_laptop_cluster.local.txt --epochs 10 --resume
```

Each rank writes to `<checkpoint-dir>/rank<N>_latest.bin`. The default
directory is `checkpoints/`, which can be overridden with
`--checkpoint-dir`, and the save frequency can be set with
`--checkpoint-every <N>`, where 0 disables saving. `--resume` loads that
same rank's own file, so no files need to be copied between machines. This
is consistent with the "same command on every machine" approach introduced
in Phase 3.

Checkpoints are stored per rank locally rather than as one shared file
because gradients sync every batch and every rank starts from the same
broadcast, so all ranks' weights and Adam momentum remain bit identical
throughout training. This means any rank's checkpoint is numerically
interchangeable with any other's, and each rank can restart independently
from its own local disk, which is useful groundwork for Phase 4b, where a
dead peer becomes a recoverable event.

If `--resume` is given a checkpoint that does not exist, or one saved by a
model of a different shape, or one saved with a different optimizer, it
fails with a specific error message and exit code 1, rather than loading
corrupted or partial state.

## Launching from one terminal

The whole cluster can be launched from a single terminal on one machine,
instead of running the same command manually on every machine:

```bash
cp scripts/cluster_nodes.conf.example scripts/cluster_nodes.conf
# edit cluster_nodes.conf: local rank and ssh target/command for remote ranks
./scripts/launch_cluster.sh -- --epochs 5 --batch-size 512
```

The local rank runs directly, and remote ranks run over SSH. Output from
every rank streams live into the same terminal, tagged with `[rank N]` and
color coded per rank. Ctrl+C stops every rank, local and remote. Any
arguments after `--` are appended to every rank's command, so the
configuration file does not need to be edited just to change `--epochs` or
`--batch-size` between runs.

This requires SSH access already working to any remote machine, meaning
that `ssh user@host` should connect without a password prompt. Set up a key
if it does not. See the comments in `cluster_nodes.conf.example` for
Windows specific setup, covering OpenSSH Server and the syntax difference
between cmd.exe and PowerShell.

### Adding a node without cloning or building there

Every push to `main` builds portable Release binaries, with
`-march=native` disabled through `STAKMESH_NATIVE_ARCH` in
`CMakeLists.txt`, for both machine types this project runs on, through
`.github/workflows/build-binaries.yml`:

- `mnist_distributed-linux-x86_64`
- `mnist_distributed-windows-x86_64`, built with MSYS2 and MinGW-w64, the
  same toolchain the Windows cluster machine uses locally

Download the artifacts from a workflow run, or from a tagged release once
one exists, then either copy the binary manually, or add a `deploy` line to
`cluster_nodes.conf` and pass `--deploy`:

```
1 deploy ./mnist_distributed.exe C:\path\to\StakMesh\build\mnist_distributed.exe
```

```bash
./scripts/launch_cluster.sh --deploy -- --epochs 5 --batch-size 512
```

This copies the binary to that rank's SSH target, using the host from its
matching `ssh` line, before any rank starts. This means a checkout, a
compiler, and a FetchContent pull are all unnecessary on the remote
machine. Combining this with Tailscale MagicDNS names, used in place of raw
IPs, in the cluster configuration file means that a change in a node's
Tailscale IP address requires no edits either.

## Benchmarks: batch size and network overhead

Early runs showed that approximately 97 percent of wall clock time was
spent in `sync_gradients()` rather than in computation. This was confirmed
and quantified by testing batch sizes of 128, 256, 512, and 1024 on the
same two laptop cluster, using MNIST, three epochs, and 30,000 samples per
rank:

| Batch size | Batches per epoch | Average epoch time | Time per batch |
|---|---|---|---|
| 128  | 234 | 75.2s | 0.321s |
| 256  | 117 | 39.6s | 0.339s |
| 512  | 58  | 19.4s | 0.335s |
| 1024 | 29  | 9.7s  | 0.334s |

Per batch time stays roughly constant, between 0.32 and 0.34 seconds,
across an eight fold range of batch sizes. This indicates the cost is
dominated by a near fixed round trip overhead per `sync_gradients()` call,
two sequential ring steps at a world size of two, rather than by the amount
of gradient data moved, approximately 437KB for this model. Larger batches
help mainly by amortizing that fixed cost over more samples, rather than by
reducing the amount of data moved per byte.

The fixed cost itself was traced to the network link. `tailscale ping`
between the two laptops, on the same WiFi network, showed approximately
101ms round trip time. A raw `ping` to the same IP, bypassing Tailscale
entirely, showed similar instability, between 12 and 185ms, with a standard
deviation around 60ms. This indicated the WiFi link itself was the cause,
rather than Tailscale or routing. Switching both machines to a phone
hotspot using raw LAN IPs, without Tailscale, reduced batch 128 epoch time
from 75.2s to approximately 29s, a steady state of about 26.5s, a
reduction of roughly 2.6 to 2.8 times, consistent with the same latency
bound model at a smaller fixed cost per round trip, approximately 160ms
reduced to 57 to 62ms. This remains above a healthy LAN's floor of a few
milliseconds, suggesting a further cause that has not yet been
investigated. One candidate is WiFi power save mode on one or both
laptops, checked with `iw dev <iface> get power_save`, and this is worth
checking before Phase 5.

## Repository layout

```
include/stakmesh/comm/   socket.hpp, topology.hpp, ring_allreduce.hpp, broadcast.hpp,
                         local_addresses.hpp, cluster_config.hpp
include/stakmesh/dist/   distributed_context.hpp
tests/                   correctness tests, comm only and full StakML integration
examples/                mnist_distributed.cpp, the real Phase 2 and Phase 3 training script
configs/                 cluster configuration for the two laptop setup, see Phase 3
CMakeLists.txt
```