// test_checkpoint.cpp
//
// Standalone (no network, no second rank needed) test for
// stakmesh/dist/checkpoint.hpp. Covers:
//
//   1. SGD round-trip: save a trained model, load into a FRESH model with
//      different random init, verify parameters match exactly.
//   2. Adam round-trip: same, but also verify m_/v_ moment buffers and the
//      t_ step counter survive the round-trip - not just the weights.
//   3. Safety check: loading a checkpoint into a model with a DIFFERENT
//      architecture must throw, not silently misalign bytes.
//   4. Safety check: loading an Adam checkpoint into an SGD optimizer (or
//      vice versa) must throw, not silently drop/fabricate momentum state.

#include <stakml/tensor.hpp>
#include <stakml/ops.hpp>
#include <stakml/nn.hpp>
#include <stakml/optim.hpp>

#include "../include/stakmesh/dist/checkpoint.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace stakml;
using namespace stakmesh::dist;

static bool g_all_ok = true;

// /tmp doesn't exist on Windows - std::filesystem::temp_directory_path()
// resolves to the right place on every platform (TMPDIR/TEMP/TMP, or a
// sane default if none of those are set).
static std::string temp_path(const std::string& filename) {
    return (std::filesystem::temp_directory_path() / filename).string();
}

static void check(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "[PASS] " << what << "\n";
    } else {
        std::cout << "[FAIL] " << what << "\n";
        g_all_ok = false;
    }
}

// Runs a few fake "training steps" (random input, backward, optimizer step)
// so parameters (and, for Adam, moment buffers) move away from their
// initial values - a round-trip that only worked on freshly-initialized
// zeros/randoms wouldn't be a meaningful test.
template <typename OptimizerT>
static void train_a_bit(nn::Linear& model, OptimizerT& opt, int steps) {
    for (int s = 0; s < steps; ++s) {
        std::vector<float> xdata(model.in_features);
        for (size_t i = 0; i < xdata.size(); ++i) {
            xdata[i] = std::sin(static_cast<float>(s * 7 + i)) * 0.7f;
        }
        auto x = std::make_shared<Tensor>(Tensor({1, model.in_features}, xdata));
        opt.zero_grad();
        Tensor out = model.forward(x);
        out.backward();
        opt.step();
    }
}

static bool params_match(const std::vector<std::shared_ptr<Tensor>>& a,
                          const std::vector<std::shared_ptr<Tensor>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->num_elements() != b[i]->num_elements()) return false;
        const float* pa = a[i]->raw_ptr();
        const float* pb = b[i]->raw_ptr();
        for (size_t j = 0; j < a[i]->num_elements(); ++j) {
            if (pa[j] != pb[j]) return false;  // exact bit match expected - same floats round-tripped
        }
    }
    return true;
}

static void test_sgd_roundtrip() {
    const std::string path = temp_path("stakmesh_test_ckpt_sgd.bin");

    nn::Linear trained(5, 3);
    optim::SGD opt(trained.parameters(), 0.1f);
    train_a_bit(trained, opt, 10);

    save_checkpoint(path, trained.parameters(), opt, /*epoch=*/4, /*rank=*/0);

    nn::Linear fresh(5, 3);  // independently random-initialized - different values
    optim::SGD fresh_opt(fresh.parameters(), 0.1f);
    check(!params_match(trained.parameters(), fresh.parameters()),
          "sanity: fresh model differs from trained model before load");

    CheckpointInfo info = load_checkpoint(path, fresh.parameters(), fresh_opt);

    check(info.epoch == 4, "SGD: loaded epoch matches saved epoch (4)");
    check(info.saved_rank == 0, "SGD: loaded rank matches saved rank (0)");
    check(params_match(trained.parameters(), fresh.parameters()),
          "SGD: loaded parameters exactly match saved parameters");

    std::remove(path.c_str());
}

static void test_adam_roundtrip() {
    const std::string path = temp_path("stakmesh_test_ckpt_adam.bin");

    nn::Linear trained(5, 3);
    optim::Adam opt(trained.parameters(), 0.01f);
    train_a_bit(trained, opt, 10);

    check(opt.t_ == 10, "sanity: Adam step counter advanced during training (t_==10)");

    save_checkpoint(path, trained.parameters(), opt, /*epoch=*/7, /*rank=*/1);

    nn::Linear fresh(5, 3);
    optim::Adam fresh_opt(fresh.parameters(), 0.01f);
    check(fresh_opt.t_ == 0, "sanity: fresh Adam optimizer starts at t_==0");

    CheckpointInfo info = load_checkpoint(path, fresh.parameters(), fresh_opt);

    check(info.epoch == 7, "Adam: loaded epoch matches saved epoch (7)");
    check(params_match(trained.parameters(), fresh.parameters()),
          "Adam: loaded parameters exactly match saved parameters");
    check(fresh_opt.t_ == opt.t_, "Adam: loaded step counter t_ matches saved t_");

    bool moments_match = true;
    for (size_t i = 0; i < opt.m_.size(); ++i) {
        if (opt.m_[i] != fresh_opt.m_[i] || opt.v_[i] != fresh_opt.v_[i]) {
            moments_match = false;
            break;
        }
    }
    check(moments_match, "Adam: loaded m_/v_ moment buffers exactly match saved state");

    // Prove this actually matters: take one more optimizer step on both the
    // original and the resumed-from-checkpoint copy with identical input,
    // and confirm they land on the identical result - i.e. resuming really
    // continues the same trajectory, not just "close enough" weights.
    std::vector<float> xdata = {0.1f, 0.2f, -0.3f, 0.4f, -0.5f};
    auto x1 = std::make_shared<Tensor>(Tensor({1, 5}, xdata));
    auto x2 = std::make_shared<Tensor>(Tensor({1, 5}, xdata));

    opt.zero_grad();
    trained.forward(x1).backward();
    opt.step();

    fresh_opt.zero_grad();
    fresh.forward(x2).backward();
    fresh_opt.step();

    check(params_match(trained.parameters(), fresh.parameters()),
          "Adam: one more step after resuming matches one more step on the original "
          "(training trajectory truly continues, not just visually-close weights)");

    std::remove(path.c_str());
}

static void test_architecture_mismatch_throws() {
    const std::string path = temp_path("stakmesh_test_ckpt_shape.bin");

    nn::Linear small(5, 3);
    optim::SGD opt(small.parameters(), 0.1f);
    save_checkpoint(path, small.parameters(), opt, 0, 0);

    nn::Linear different_shape(5, 4);  // different out_features -> different param sizes
    optim::SGD other_opt(different_shape.parameters(), 0.1f);

    bool threw = false;
    try {
        load_checkpoint(path, different_shape.parameters(), other_opt);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "architecture mismatch: loading into a differently-shaped model throws");

    std::remove(path.c_str());
}

static void test_optimizer_mismatch_throws() {
    const std::string path = temp_path("stakmesh_test_ckpt_opt.bin");

    nn::Linear model(5, 3);
    optim::Adam adam_opt(model.parameters(), 0.01f);
    train_a_bit(model, adam_opt, 3);
    save_checkpoint(path, model.parameters(), adam_opt, 0, 0);

    nn::Linear other(5, 3);
    optim::SGD sgd_opt(other.parameters(), 0.1f);

    bool threw = false;
    try {
        load_checkpoint(path, other.parameters(), sgd_opt);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "optimizer mismatch: loading an Adam checkpoint into SGD throws "
                 "(would otherwise silently discard momentum)");

    std::remove(path.c_str());
}

int main() {
    test_sgd_roundtrip();
    test_adam_roundtrip();
    test_architecture_mismatch_throws();
    test_optimizer_mismatch_throws();

    if (g_all_ok) {
        std::cout << "\nAll checkpoint tests passed.\n";
        return 0;
    } else {
        std::cout << "\nSome checkpoint tests FAILED.\n";
        return 1;
    }
}