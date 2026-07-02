# StakMesh

CPU-only distributed training, built from scratch over raw TCP sockets across
a real two-node cluster.

StakMesh is the distributed-systems layer for [StakML](https://github.com/STAKXX002/StakML) -
a tensor/autograd library built from scratch. Where StakML answers "how
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
comm/                         StakML-agnostic. Moves raw float* over TCP. Reusable for anything (gradients, checkpoints, control messages).
  socket.hpp                  RAII TCP wrapper: connect/listen/send_all/recv_all
  topology.hpp                static peer list -> live ring (send_sock, recv_sock)
  ring_allreduce.hpp          reduce-scatter + all-gather over the ring

dist/                         The ONLY layer that knows about StakML.
  distributed_context.hpp     model.parameters() -> ring_all_reduce on .grad_
```

The design mirrors how StakML itself added its CUDA backend: the new layer
plugs in without touching `Tensor`, `ops`, `nn`, or `optim` at all. Your
training loop only gains one line:

```cpp
loss->backward();
ctx.sync_gradients(model.parameters());   // <-- the only new line
opt.step();
opt.zero_grad();
```

Every rank computes gradients on its own data shard, `sync_gradients()`
averages them across all ranks in place, and every rank then runs the exact
same unmodified `opt.step()` - so weights stay identical across the cluster
without a single distributed-aware line inside StakML itself.

## Status: Phase 1 complete

- [x] **Phase 1 - Collective communication library.** Ring all-reduce over
      raw TCP, verified correct against a reference implementation across
      2/3/4/5 simulated ranks, including uneven chunk sizes and degenerate
      cases (`tests/test_ring_allreduce.cpp`).
- [x] **`DistributedContext` glue verified end-to-end** against real StakML
      `nn::Linear` + `optim::SGD`: two ranks with identical initial weights
      but different input data converge to bit-identical weights after one
      `sync_gradients()` + `step()` (`tests/test_distributed_integration.cpp`).
- [ ] **Phase 2 - Multi-GPU-style CPU data parallelism**, wired all the way
      through a real training loop (MNIST-scale) split across your two
      laptops instead of simulated threads.
- [ ] **Phase 3 - Process launcher.** A minimal `torchrun`-equivalent: spawn
      the training script on both machines, assign ranks, coordinate startup
      (currently you'd hand-write the peer list and launch each rank
      manually).
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

## Running across your two laptops

Not yet wired up as a launcher (that's Phase 3) - for now, `establish_ring()`
takes a static peer list of `{host, port}` per rank that you provide, e.g.:

```cpp
std::vector<stakmesh::comm::PeerAddr> peers = {
    {"192.168.1.10", 29500},  // rank 0 - laptop A
    {"192.168.1.11", 29500},  // rank 1 - laptop B
};
auto ctx = stakmesh::dist::DistributedContext::init(my_rank, peers);
```

Run the same binary on both machines with `my_rank` set to 0 and 1
respectively (CLI arg or env var - not yet standardized, part of Phase 2).
Both machines need to be reachable on the same LAN and the chosen port open.