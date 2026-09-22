#pragma once

#include <optional>
#include <string>

#include <torch/python.h>

#include <deep_gemm/common/types.cuh>

#include "../../utils/exception.hpp"

namespace deep_gemm {

static std::string get_default_epilogue_type(const std::optional<std::string>& epilogue_type) {
    return epilogue_type.value_or("epilogue::transform::EpilogueIdentity");
}

// The full host-determined epilogue input: the operator type to instantiate the kernel with,
// and the operator's runtime state to launch the kernel with
struct EpilogueInput {
    std::string type = "epilogue::transform::EpilogueIdentity";
    EpilogueArgs args;
};

// Construct the epilogue input, keeping the operator type and its state consistent
// NOTES: the selection only depends on which inputs are given, never on their values
static EpilogueInput make_epilogue_input(const int& m, const int& n,
                                         const std::optional<std::string>& epilogue_type = std::nullopt,
                                         const std::optional<float>& alpha = std::nullopt,
                                         const std::optional<torch::Tensor>& sfd = std::nullopt) {
    if (sfd.has_value()) {
        DG_HOST_ASSERT(not epilogue_type.has_value() and not alpha.has_value());
        return {"epilogue::transform::EpilogueDynamicScaledFP8",
                {.sfd = static_cast<uint32_t*>(sfd->data_ptr()),
                 .sfd_stride = static_cast<uint32_t>(sfd->stride(-1)),
                 .shape_m = static_cast<uint32_t>(m), .shape_n = static_cast<uint32_t>(n)}};
    }
    if (alpha.has_value()) {
        DG_HOST_ASSERT(not epilogue_type.has_value());
        return {"epilogue::transform::EpilogueWithAlpha", {.alpha = alpha.value()}};
    }
    return {get_default_epilogue_type(epilogue_type), {}};
}

} // namespace deep_gemm
