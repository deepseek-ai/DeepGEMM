#pragma once

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <torch/python.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <deep_gemm/common/types.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/layout.hpp"

namespace deep_gemm {

// The GEMM epilogue type: selects the device-side `epilogue::operators` implementation,
// carries its runtime state, and validates the GEMM output against it
// NOTES: every SM100 GEMM launches with exactly one epilogue class; they do not compose
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;

    virtual EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const { return {}; }

    virtual deep_jit::cuda::CompilerOptions compiler_options() const { return {}; }

    virtual void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    }

    // The BLAS linear-combination coefficient, when this epilogue is a plain store cuBLASLt
    // can compute directly; `nullopt` when cuBLASLt cannot handle it
    virtual std::optional<float> get_alpha() const { return std::nullopt; }

    // The SFD this epilogue quantizes into, if any (an empty problem zeroes it)
    virtual std::optional<torch::Tensor> output_sfd() const { return std::nullopt; }
};

class IdentityEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::Identity";
    }

    std::optional<float> get_alpha() const override { return 1.0f; }
};

class AlphaEpilogue final: public EpilogueClass {
    float alpha;

public:
    explicit AlphaEpilogue(const float& alpha): alpha(alpha) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::ScaleByAlpha";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int&) const override {
        return {.alpha = alpha};
    }

    std::optional<float> get_alpha() const override { return alpha; }
};

class FP8CastWithClampEpilogue final: public EpilogueClass {
    torch::Tensor sfd;
    std::optional<float> clamp_value;
    std::optional<torch::Tensor> clamped_count;

    static uint32_t* get_clamp_count_ready() {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> ready_by_stream;
        auto& ready = ready_by_stream[stream];
        if (not ready.defined()) {
            // Initialize each stream once, before graph capture.
            DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
            ready = torch::zeros({2}, torch::TensorOptions().device(stream.device()).dtype(torch::kUInt32));
        }
        return static_cast<uint32_t*>(ready.data_ptr());
    }

public:
    explicit FP8CastWithClampEpilogue(const torch::Tensor& sfd,
                                      const std::optional<float>& clamp_value = std::nullopt,
                                      const std::optional<torch::Tensor>& clamped_count = std::nullopt):
        sfd(sfd), clamp_value(clamp_value), clamped_count(clamped_count) {
        DG_HOST_ASSERT(not clamp_value.has_value() or clamp_value.value() > 0.0f);
        if (clamped_count.has_value())
            DG_HOST_ASSERT(clamped_count->scalar_type() == torch::kUInt32 and clamped_count->numel() == 1);
    }

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::CastToFP8WithClamp";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.data_ptr()),
                .sfd_stride = static_cast<uint32_t>(sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n),
                .clamp_value = clamp_value.value_or(0.0f),
                .clamped_count = clamped_count.has_value() ? static_cast<uint32_t*>(clamped_count->data_ptr()) : nullptr,
                .clamp_count_ready = clamped_count.has_value() ? get_clamp_count_ready() : nullptr};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn and
                       "FP8 quantization requires a direct E4M3 output");
        const auto [num_groups, m, n] = get_shape<3>(d);
        DG_HOST_ASSERT(n % 128 == 0 and sfd.device() == d.device());
        check_sf_layout(sfd, m, num_groups * n, 1, 32, std::nullopt, true, false, torch::kInt);
    }

    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
};

class BiasEpilogue final: public EpilogueClass {
    torch::Tensor bias;

public:
    explicit BiasEpilogue(const torch::Tensor& bias): bias(bias) {}

    std::string get_epilogue_operator_type() const override {
        return std::format("epilogue::operators::AddBias<{}>",
                           bias.scalar_type() == torch::kFloat ? "float" : "cutlass::bfloat16_t");
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int&, const int& n) const override {
        return {.shape_n = static_cast<uint32_t>(n), .bias = bias.data_ptr()};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        EpilogueClass::check(c, d);
        DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and "Bias GEMM requires SM100");
        DG_HOST_ASSERT(d.dim() == 2 and bias.dim() == 2 and bias.size(0) == 1 and bias.size(1) == d.size(1));
        DG_HOST_ASSERT(bias.device() == d.device() and bias.scalar_type() == d.scalar_type());
        DG_HOST_ASSERT(bias.is_contiguous() and "Bias must be contiguous");
    }
};

class BF16StochasticRoundingEpilogue final: public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToBF16";
    }

    // `cvt.rs` requires an architecture-specific target (e.g., `sm_100a`), not a family one
    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return {.arch = jit->device.get_arch(false)};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16 and
                       "Stochastic rounding requires a direct BF16 output");
    }
};

class FP8StochasticRoundingEpilogue final: public EpilogueClass {
    torch::Tensor sfd;

public:
    explicit FP8StochasticRoundingEpilogue(const torch::Tensor& sfd): sfd(sfd) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToFP8";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.data_ptr()),
                .sfd_stride = static_cast<uint32_t>(sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n)};
    }

    deep_jit::cuda::CompilerOptions compiler_options() const override {
        return {.arch = jit->device.get_arch(false)};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn and
                       "FP8 stochastic rounding requires a direct E4M3 output");
        const auto [num_groups, m, n] = get_shape<3>(d);
        DG_HOST_ASSERT(n % 128 == 0 and sfd.device() == d.device());
        check_sf_layout(sfd, m, num_groups * n, 1, 32, std::nullopt, true, false, torch::kInt);
    }

    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
};

// Resolve the epilogue-related API inputs into the single epilogue class of an SM100 GEMM,
// validated against the GEMM output: the user class, the standalone `alpha` (an API argument
// for cuBLASLt signature parity), bias, or the FP8 `(d, sfd)` output pair; all exclusive
static std::shared_ptr<EpilogueClass> resolve_epilogue_class(const std::shared_ptr<EpilogueClass>& epilogue_class,
                                                             const std::optional<torch::Tensor>& c,
                                                             const torch::Tensor& d,
                                                             const std::optional<float>& alpha = std::nullopt,
                                                             const std::optional<torch::Tensor>& sfd = std::nullopt) {
    const bool with_bias = c.has_value() and d.dim() == 2 and c->dim() == 2 and
                           c->size(0) == 1 and c->data_ptr() != d.data_ptr();
    DG_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() + with_bias <= 1 and
                   "The epilogue class, `alpha`, the FP8 output pair and bias are exclusive");

    std::shared_ptr<EpilogueClass> resolved = epilogue_class;
    if (with_bias)
        resolved = std::make_shared<BiasEpilogue>(c.value());
    else if (alpha.has_value())
        resolved = std::make_shared<AlphaEpilogue>(alpha.value());
    else if (sfd.has_value())
        resolved = std::make_shared<FP8CastWithClampEpilogue>(sfd.value());
    else if (resolved == nullptr)
        resolved = std::make_shared<IdentityEpilogue>();
    resolved->check(c, d);
    return resolved;
}

} // namespace deep_gemm
