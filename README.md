# StakMesh

CPU-only distributed training, built from scratch over raw TCP sockets across
a real two-node cluster (two laptops, no GPUs, no NCCL, no MPI).

StakMesh is the distributed-systems layer for [StakML](https://github.com/STAKXX002/StakML) -
a tensor/autograd library also built from scratch. Where StakML answers "how
do you compute a gradient," StakMesh answers "how do multiple machines agree
on one."

## Why this exists

Every framework you've used (PyTorch DDP, Horovod, JAX pmap) hides the same
core primitive behind a one-line API call: **ring all-reduce**. This project
builds that primitive from nothing - raw POSIX sockets, hand-rolled wire
protocol, the actual reduce-scatter/all-gather algorithm - and then wires it
into a real autograd engine so gradients computed on two separate physical
machines end up identical after every training step.

No GPUs required. The interesting problems here are networking, concurrency,
and coordination, not FLOPs.

## Architecture

```
comm/            StakML-agnostic. Moves raw float* over TCP. Reusable for
                  anything (gradients, checkpoints, control messages).
  socket.hpp        RAII TCP wrapper: connect/listen/send_all/recv_all
  topology.hpp       static peer list -> live ring (send_sock, recv_sock)
  ring_allreduce.hpp  reduce-scatter + all-gather over the ring
  broadcast.hpp       one rank's buffer -> every rank (used for initial weights)
  local_addresses.hpp  this machine's own IPv4 addresses (POSIX + Windows)
  cluster_config.hpp   parse "rank host port" file + auto-detect own rank

dist/             The ONLY layer that knows about StakML.
  distributed_context.hpp   model.parameters() -> broadcast (once) + all-reduce (every step)
```

The design mirrors how StakML itself added its CUDA backend: the new layer
plugs in without touching `Tensor`, `ops`, `nn`, or `optim` at all. A full
distributed training loop only gains two new lines over a single-machine one:

```cpp
auto ctx = stakmesh::dist::DistributedContext::init(my_rank, peers);
ctx.broadcast_parameters(model.parameters());   // once, before training starts

// ... standard training loop ...
loss->backward();
ctx.sync_gradients(model.parameters());   // <-- the only new line per step
opt.step();
opt.zero_grad();
```

Every rank starts from identical weights (broadcast), computes gradients on
its own data shard, `sync_gradients()` averages them across all ranks in
place, and every rank then runs the exact same unmodified `opt.step()` - so
weights stay identical across the cluster without a single distributed-aware
line inside StakML itself.

## Status: Phase 1 + core of Phase 2 complete

- [x] **Phase 1 - Collective communication library.** Ring all-reduce over
      raw TCP, verified correct against a reference implementation across
      2/3/4/5 simulated ranks, including uneven chunk sizes and degenerate
      cases (`tests/test_ring_allreduce.cpp`). Ring broadcast for
      distributing initial weights, verified the same way, including a
      non-zero/mid-ring root (`tests/test_broadcast.cpp`).
- [x] **`DistributedContext` verified end-to-end** against real StakML
      `nn::Linear` + `optim::SGD`: two ranks with DIFFERENT random initial
      weights converge to bit-identical weights after one
      `broadcast_parameters()` call, then STAY identical across 25
      consecutive `sync_gradients()` + `step()` calls on the same
      connection - the actual shape of a real training loop
      (`tests/test_distributed_integration.cpp`,
      `tests/test_distributed_training_loop.cpp`).
- [x] **Phase 2 - Real data-parallel training script**
      (`examples/mnist_distributed.cpp`): reuses StakML's own MNIST loader
      and model architecture, shards the training set contiguously by
      rank, broadcasts initial weights, syncs gradients every batch.
      Compiles and links cleanly against real StakML headers; not yet
      run against real MNIST files or a real second machine - that's the
      next step, on your actual hardware.
- [x] **Phase 3 - Cluster config + automatic rank detection.** Write the
      cluster topology down once (`configs/two_laptop_cluster.txt`), run
      the exact same command on every machine - each process resolves
      every entry's host and detects its own rank by matching against its
      own IP addresses (`tests/test_cluster_config.cpp`, 9/9 including both
      failure modes: no match, ambiguous match). `--rank`/`--peers` still
      work for quick manual overrides. **Remote launch from one terminal**
      (`scripts/launch_cluster.sh`) now starts every rank - local directly,
      remote over SSH - from a single command, tagged/colored live output,
      Ctrl+C stops all of them. See "Launching from one terminal" below.
- [ ] **Phase 4 - Fault tolerance & checkpointing.** Kill one node mid-run,
      resume from a checkpoint without restarting the other rank.
- [ ] **Phase 5 - Bandwidth-aware optimizations.** Gradient compression /
      quantization, since home WiFi/Ethernet is a real constraint here, not
      a theoretical one.
- [ ] **Phase 6 (stretch) - Inference serving layer.** Request batching
      across the cluster once training works.

## Building

Requires CMake ≥ 3.28, a C++17 compiler, and pthreads. StakML is pulled
automatically via `FetchContent`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

While developing StakML and StakMesh side by side, point at a local checkout
instead of re-fetching from GitHub every time:

```bash
cmake .. -DSTAKMESH_STAKML_SOURCE_DIR=/path/to/your/StakML
```

## Running the real distributed training example

You'll need the standard MNIST IDX files (`train-images-idx3-ubyte`,
`train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`)
at the same relative path on both machines (default: `../data` next to your
build directory - override with `--data-dir`).

### Recommended: config mode (Phase 3)

Edit `configs/two_laptop_cluster.txt` with your real addresses (Tailscale
IPs, hostnames, or LAN IPs - anything resolvable), then run the **exact
same command** on every machine:

```bash
./mnist_distributed --config ../configs/two_laptop_cluster.txt
```

Each process figures out its own rank automatically by checking which
config entry matches an IP it actually owns. If it can't find a match (or
finds more than one - e.g. testing on one machine), it fails fast with a
clear error rather than guessing - pass `--rank <N>` explicitly to
override auto-detection in that case.

### Manual mode (still supported)

```bash
# On laptop A:
./mnist_distributed --rank 0 --peers 192.168.1.10:29500,192.168.1.11:29500

# On laptop B:
./mnist_distributed --rank 1 --peers 192.168.1.10:29500,192.168.1.11:29500
```

### Either way

Both machines need to reach each other on the chosen port - same network
or Tailscale, and the port open on both sides (check the Windows firewall
if the connection hangs). Each rank trains on its shard of the dataset and
prints its own loss/accuracy per epoch; because gradients sync every
batch, both ranks' models stay identical throughout even though neither
sees the other's data.

## Launching from one terminal

Instead of manually running the same command on every machine, launch the
whole cluster from a single terminal on one machine:

```bash
cp scripts/cluster_nodes.conf.example scripts/cluster_nodes.conf
# edit cluster_nodes.conf: local rank + ssh target/command for remote ranks
./scripts/launch_cluster.sh -- --epochs 5 --batch-size 512
```

The local rank runs directly; remote ranks run over SSH. Output from every
rank streams live into the same terminal, tagged `[rank N]` and
color-coded per rank. Ctrl+C stops everything, local and remote. Anything
after `--` gets appended to every rank's command, so you don't have to
edit the config file just to change `--epochs`/`--batch-size` between runs.

Requires SSH access already working to any remote machine (`ssh user@host`
should just work, no password prompt ideally - set up a key if it doesn't).
See the comments in `cluster_nodes.conf.example` for Windows-specific setup
(OpenSSH Server, cmd.exe vs PowerShell syntax).

## Benchmarks: batch size & network overhead

Early runs showed **~97% of wall-clock time was spent in `sync_gradients()`,
not compute** - confirmed and quantified by testing batch sizes 128/256/512/1024
on the same 2-laptop cluster (MNIST, 3 epochs, 30k samples/rank):

| Batch size | Batches/epoch | Avg epoch time | Time/batch |
|---|---|---|---|
| 128  | 234 | 75.2s | 0.321s |
| 256  | 117 | 39.6s | 0.339s |
| 512  | 58  | 19.4s | 0.335s |
| 1024 | 29  | 9.7s  | 0.334s |

Per-batch time stays roughly constant (~0.32-0.34s) across an 8x range of
batch sizes - the cost is dominated by a near-fixed round-trip overhead per
`sync_gradients()` call (2 sequential ring steps at world_size=2), not by
the amount of gradient data moved (~437KB for this model). Bigger batches
help mainly by amortizing that fixed cost over more samples, not by moving
less data per byte.

Root-caused the fixed cost itself: `tailscale ping` between the two
laptops (same WiFi network) showed ~101ms RTT; a raw `ping` to the same IP
(bypassing Tailscale entirely) showed the same instability (12-185ms,
~60ms stddev) - so it wasn't a Tailscale/routing issue, it was the WiFi
link itself. Switching both machines to a phone hotspot with raw LAN IPs
(no Tailscale) dropped batch-128 epoch time from 75.2s to ~29s (~26.5s
steady-state) - about a 2.6-2.8x reduction, consistent with the same
latency-bound model at a smaller fixed cost per round trip (~160ms -> ~57-62ms).
Still above a healthy LAN's sub-5ms floor, so there's likely a further,
uninvestigated cause (candidate: WiFi power-save mode on one or both
laptops - `iw dev <iface> get power_save`) worth checking before Phase 5.

## Repo layout

```
include/stakmesh/comm/   socket.hpp, topology.hpp, ring_allreduce.hpp, broadcast.hpp,
                         local_addresses.hpp, cluster_config.hpp
include/stakmesh/dist/   distributed_context.hpp
tests/                   correctness tests (comm-only + full StakML integration)
examples/                mnist_distributed.cpp - real Phase 2/3 training script
configs/                 cluster config for the 2-laptop setup (see Phase 3)
CMakeLists.txt
```