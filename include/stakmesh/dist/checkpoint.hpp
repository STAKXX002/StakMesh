#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// checkpoint.hpp - save/restore training state to/from a local binary file.
//
// CONCEPT: Why per-rank, local files - not one shared checkpoint?
//
//   Because sync_gradients() averages gradients every batch and every rank
//   starts from the same broadcast_parameters() call, every rank's weights
//   (and, for Adam, its optimizer momentum) stay bit-identical throughout
//   training - it's deterministic, no per-rank randomness anywhere in the
//   update rule. So any rank's checkpoint is numerically interchangeable
//   with any other's.
//
//   That means the simplest reliable design is: each rank writes its OWN
//   checkpoint to its OWN local disk, on its OWN schedule. If a rank's
//   machine crashes, its checkpoint is already sitting right there for a
//   local restart - no need to fetch a file over the network from a
//   machine that might itself be the one that just died.
//
// CONCEPT: Why does loading validate shapes before touching any data?
//
//   A checkpoint saved for one model architecture, loaded into a
//   DIFFERENT architecture (you changed a layer size and forgot), would
//   otherwise silently misalign every byte after the mismatch - garbage
//   weights with no error. Every size is checked against the live model
//   BEFORE any float is copied in, so a mismatch is a loud, immediate
//   exception instead of silently-wrong numbers three epochs later.
//
// CONCEPT: Why refuse to load across different optimizer types?
//
//   Adam's m_/v_ moment buffers materially change training dynamics -
//   silently resetting them (or silently loading them into an optimizer
//   that doesn't have them) is exactly the kind of bug that doesn't crash,
//   it just makes results subtly wrong. Loading is all-or-nothing: either
//   the optimizer state matches what's in the file, or it refuses to load.
//
// FILE FORMAT (little-endian, matches x86/ARM - the only architectures
// this project targets):
//
//   [8 bytes]  magic "SMCKPT01"
//   [4 bytes]  rank that saved this file        (int32, diagnostic only)
//   [4 bytes]  epoch                            (int32)
//   [4 bytes]  parameter count                  (int32)
//   [8 bytes * param count]  each parameter's element count (uint64)
//   [4 bytes * total elements]  parameter values, per-parameter concatenated,
//                               same order as the `parameters` vector (float32)
//   [1 byte]   optimizer tag: 0 = none/SGD, 1 = Adam
//   -- if Adam --
//   [8 bytes]  step counter t_                  (uint64)
//   [4 bytes * total elements]  m_ (first moments),  per-parameter concatenated
//   [4 bytes * total elements]  v_ (second moments), per-parameter concatenated
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <stakml/tensor.hpp>
#include <stakml/optim.hpp>

namespace stakmesh {
namespace dist {

namespace checkpoint_detail {

constexpr char kMagic[8] = {'S', 'M', 'C', 'K', 'P', 'T', '0', '1'};

inline void write_raw(std::ofstream& f, const void* data, size_t bytes) {
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("checkpoint: write failed (disk full? permissions?)");
}

inline void read_raw(std::ifstream& f, void* data, size_t bytes) {
    f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("checkpoint: read failed - file truncated or corrupt");
}

template <typename T>
inline void write_pod(std::ofstream& f, const T& v) {
    write_raw(f, &v, sizeof(T));
}

template <typename T>
inline void read_pod(std::ifstream& f, T& v) {
    read_raw(f, &v, sizeof(T));
}

enum class OptimizerKind : uint8_t { None = 0, Adam = 1 };

}  // namespace checkpoint_detail

// Returned by load_checkpoint() so the caller knows where training left off.
struct CheckpointInfo {
    int epoch;        // resume training at epoch + 1
    int saved_rank;   // which rank originally wrote this file (diagnostic only)
};

// ── save ─────────────────────────────────────────────────────────────────
// `epoch` should be the last COMPLETED epoch (so loading resumes at epoch+1).
template <typename OptimizerT>
void save_checkpoint(const std::string& path,
                      const std::vector<std::shared_ptr<stakml::Tensor>>& parameters,
                      const OptimizerT& opt,
                      int epoch,
                      int rank) {
    using namespace checkpoint_detail;
    (void)opt;  // unused when OptimizerT has no extra state (e.g. SGD)

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("checkpoint: could not open '" + path + "' for writing");

    write_raw(f, kMagic, sizeof(kMagic));
    write_pod(f, static_cast<int32_t>(rank));
    write_pod(f, static_cast<int32_t>(epoch));
    write_pod(f, static_cast<int32_t>(parameters.size()));

    for (const auto& p : parameters) {
        write_pod(f, static_cast<uint64_t>(p->num_elements()));
    }
    for (const auto& p : parameters) {
        write_raw(f, p->raw_ptr(), p->num_elements() * sizeof(float));
    }

    constexpr bool is_adam = std::is_same_v<OptimizerT, stakml::optim::Adam>;
    write_pod(f, static_cast<uint8_t>(is_adam ? OptimizerKind::Adam : OptimizerKind::None));

    if constexpr (is_adam) {
        write_pod(f, static_cast<uint64_t>(opt.t_));
        for (const auto& m : opt.m_) write_raw(f, m.data(), m.size() * sizeof(float));
        for (const auto& v : opt.v_) write_raw(f, v.data(), v.size() * sizeof(float));
    }
}

// ── load ─────────────────────────────────────────────────────────────────
// Overwrites `parameters`' values and (for Adam) `opt`'s moment buffers
// in place. Throws on any shape or optimizer-type mismatch rather than
// loading partial/misaligned data.
template <typename OptimizerT>
CheckpointInfo load_checkpoint(const std::string& path,
                                const std::vector<std::shared_ptr<stakml::Tensor>>& parameters,
                                OptimizerT& opt) {
    using namespace checkpoint_detail;

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("checkpoint: could not open '" + path + "' for reading");

    char magic[8];
    read_raw(f, magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("checkpoint: '" + path +
                                  "' is not a valid StakMesh checkpoint (bad magic bytes)");
    }

    int32_t saved_rank = 0, epoch = 0, num_params = 0;
    read_pod(f, saved_rank);
    read_pod(f, epoch);
    read_pod(f, num_params);

    if (static_cast<size_t>(num_params) != parameters.size()) {
        throw std::runtime_error(
            "checkpoint: '" + path + "' has " + std::to_string(num_params) +
            " parameters, but the model has " + std::to_string(parameters.size()) +
            " - architecture mismatch, refusing to load");
    }

    std::vector<uint64_t> counts(static_cast<size_t>(num_params));
    for (int i = 0; i < num_params; ++i) {
        read_pod(f, counts[static_cast<size_t>(i)]);
        if (counts[static_cast<size_t>(i)] != parameters[static_cast<size_t>(i)]->num_elements()) {
            throw std::runtime_error(
                "checkpoint: parameter " + std::to_string(i) + " has " +
                std::to_string(counts[static_cast<size_t>(i)]) + " elements in the checkpoint but " +
                std::to_string(parameters[static_cast<size_t>(i)]->num_elements()) +
                " in the model - architecture mismatch, refusing to load");
        }
    }
    for (int i = 0; i < num_params; ++i) {
        auto& p = parameters[static_cast<size_t>(i)];
        read_raw(f, p->raw_ptr(), counts[static_cast<size_t>(i)] * sizeof(float));
    }

    uint8_t tag = 0;
    read_pod(f, tag);
    auto kind = static_cast<OptimizerKind>(tag);
    constexpr bool is_adam = std::is_same_v<OptimizerT, stakml::optim::Adam>;

    if (is_adam && kind != OptimizerKind::Adam) {
        throw std::runtime_error(
            "checkpoint: '" + path +
            "' has no Adam moment state (saved with a different optimizer), but you're "
            "loading into Adam - momentum would silently reset. Refusing to load.");
    }
    if (!is_adam && kind == OptimizerKind::Adam) {
        throw std::runtime_error(
            "checkpoint: '" + path +
            "' was saved WITH Adam moment state, but you're loading into a non-Adam "
            "optimizer - the saved momentum would be silently discarded. Refusing to load.");
    }

    if constexpr (is_adam) {
        uint64_t t = 0;
        read_pod(f, t);
        opt.t_ = static_cast<size_t>(t);
        for (auto& m : opt.m_) read_raw(f, m.data(), m.size() * sizeof(float));
        for (auto& v : opt.v_) read_raw(f, v.data(), v.size() * sizeof(float));
    }

    return CheckpointInfo{epoch, saved_rank};
}

}  // namespace dist
}  // namespace stakmesh