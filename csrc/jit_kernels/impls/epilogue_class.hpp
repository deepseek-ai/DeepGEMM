#pragma once

#include <memory>
#include <optional>
#include <string>

#include <torch/python.h>

#include <deep_gemm/common/types.cuh>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"

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

class FP8QuantizationEpilogue final: public EpilogueClass {
    torch::Tensor sfd;

public:
    explicit FP8QuantizationEpilogue(const torch::Tensor& sfd): sfd(sfd) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::QuantizeToFP8";
    }

    EpilogueOperatorArgs make_epilogue_operator_args(const int& m, const int& n) const override {
        return {.sfd = static_cast<uint32_t*>(sfd.data_ptr()),
                .sfd_stride = static_cast<uint32_t>(sfd.stride(-1)),
                .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n)};
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn and
                       "FP8 quantization requires a direct E4M3 output");
    }

    std::optional<torch::Tensor> output_sfd() const override { return sfd; }
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

// Resolve the epilogue-related API inputs into the single epilogue class of an SM100 GEMM,
// validated against the GEMM output: the user class, the standalone `alpha` (an API argument
// for cuBLASLt signature parity), or the FP8 `(d, sfd)` output pair; all exclusive
static std::shared_ptr<EpilogueClass> resolve_epilogue_class(const std::shared_ptr<EpilogueClass>& epilogue_class,
                                                             const std::optional<torch::Tensor>& c,
                                                             const torch::Tensor& d,
                                                             const std::optional<float>& alpha = std::nullopt,
                                                             const std::optional<torch::Tensor>& sfd = std::nullopt) {
    DG_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() <= 1 and
                   "The epilogue class, `alpha` and the FP8 output pair are exclusive");
    std::shared_ptr<EpilogueClass> resolved = epilogue_class;
    if (alpha.has_value())
        resolved = std::make_shared<AlphaEpilogue>(alpha.value());
    if (sfd.has_value())
        resolved = std::make_shared<FP8QuantizationEpilogue>(sfd.value());
    if (resolved == nullptr)
        resolved = std::make_shared<IdentityEpilogue>();
    resolved->check(c, d);
    return resolved;
}

} // namespace deep_gemm
