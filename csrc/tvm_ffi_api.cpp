#include <cstdint>
#include <optional>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/object.h>
#include <torch/torch.h>
#include <ATen/DLConvertor.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "utils/torch_compat.hpp"


using namespace deep_gemm;
using namespace tvm::ffi;

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------
void dg_init(std::string library_root_path, std::string cuda_home) {
#if DG_TENSORMAP_COMPATIBLE
    Compiler::prepare_init(library_root_path, cuda_home);
    KernelRuntime::prepare_init(cuda_home);
    IncludeParser::prepare_init(library_root_path);
#endif
}

int64_t dg_get_num_sms() { return device_runtime->get_num_sms(); }
void dg_set_num_sms(int64_t n) { device_runtime->set_num_sms(static_cast<int>(n)); }
// int64_t dg_get_compile_mode() { return device_runtime->get_compile_mode(); }
// void dg_set_compile_mode(int64_t n) { device_runtime->set_compile_mode(static_cast<int>(n)); }
int64_t dg_get_tc_util() { return device_runtime->get_tc_util(); }
void dg_set_tc_util(int64_t n) { device_runtime->set_tc_util(static_cast<int>(n)); }
bool dg_get_pdl() { return device_runtime->get_pdl(); }
void dg_set_pdl(bool v) { device_runtime->set_pdl(v); }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, dg_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_num_sms, dg_get_num_sms);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_num_sms, dg_set_num_sms);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_compile_mode, dg_get_compile_mode);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_compile_mode, dg_set_compile_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tc_util, dg_get_tc_util);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_tc_util, dg_set_tc_util);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_pdl, dg_get_pdl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_pdl, dg_set_pdl);

// ---------------------------------------------------------------------------
// Layout utilities
// ---------------------------------------------------------------------------
int64_t dg_get_tma_aligned_size(int64_t mn, int64_t element_size) {
    return get_tma_aligned_size(static_cast<int>(mn), static_cast<int>(element_size));
}

int64_t dg_get_mk_alignment_for_contiguous_layout() {
    return heuristics_runtime->get_mk_alignment_for_contiguous_layout();
}

void dg_set_mk_alignment_for_contiguous_layout(int64_t new_value) {
    heuristics_runtime->set_mk_alignment_for_contiguous_layout(static_cast<int>(new_value));
}

int64_t dg_get_theoretical_mk_alignment_for_contiguous_layout(Optional<int64_t> expected_m) {
    auto val = expected_m.has_value()? std::make_optional(static_cast<int>(expected_m.value())) : std::nullopt;
    return heuristics_runtime->get_theoretical_mk_alignment_for_contiguous_layout(val);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_tma_aligned_size, dg_get_tma_aligned_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mk_alignment_for_contiguous_layout, dg_get_mk_alignment_for_contiguous_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_mk_alignment_for_contiguous_layout, dg_set_mk_alignment_for_contiguous_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_theoretical_mk_alignment_for_contiguous_layout, dg_get_theoretical_mk_alignment_for_contiguous_layout);


// ---------------------------------------------------------------------------
// Layout kernels
// ---------------------------------------------------------------------------
#if DG_TENSORMAP_COMPATIBLE

tvm::ffi::Array<int64_t> dg_preprocess_sf(TensorView sf) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto [dim, ng, mn_pp, sf_k_pp, tma_mn, batched_sf] = preprocess_sf(sf_v);
    return {static_cast<int64_t>(dim),
            static_cast<int64_t>(ng),
            static_cast<int64_t>(mn_pp),
            static_cast<int64_t>(sf_k_pp),
            static_cast<int64_t>(tma_mn)};
}

Tensor dg_get_mn_major_tma_aligned_tensor(TensorView sf) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto result = get_mn_major_tma_aligned_tensor(sf_v);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_mn_major_tma_aligned_packed_ue8m0_tensor(TensorView sf) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto result = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_v);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        TensorView sf, TensorView ks_tensor, Array<int64_t> ks, int64_t gran_k) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto ks_tensor_v = convert_to_torch_tensor(ks_tensor);
    std::vector<int> ks_opt;
    ks_opt.reserve(ks.size());

    for (Array<int64_t>::iterator it = ks.begin(); it != ks.end(); ++it) {
        ks_opt.push_back(static_cast<int>(*it));
    }
    auto result = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf_v,
        ks_tensor_v,
        ks_opt,
        static_cast<int>(gran_k)
    );
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_transform_sf_into_required_layout(
        TensorView sf, int64_t mn, int64_t k,
        int64_t recipe_a, int64_t recipe_b, Optional<int64_t> recipe_c,
        Optional<int64_t> num_groups,
        Optional<bool> is_sfa,
        bool disable_ue8m0_cast) {
    auto sf_v = convert_to_torch_tensor(sf);
    auto is_sfa_val = is_sfa.has_value() ? std::make_optional(is_sfa.value()) : std::nullopt;
    auto ng = num_groups.has_value() ? std::make_optional(static_cast<int>(num_groups.value())) : std::nullopt;
    if(recipe_c.has_value()) {
        auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c.value()));
        auto result = layout::transform_sf_into_required_layout(
            sf_v, static_cast<int>(mn), static_cast<int>(k),
            recipe, ng, is_sfa_val, disable_ue8m0_cast);
        return Tensor::FromDLPack(at::toDLPack(result));
    } else {
        auto recipe = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b));
        auto result = layout::transform_sf_into_required_layout(
            sf_v, static_cast<int>(mn), static_cast<int>(k),
            recipe, ng, is_sfa_val, disable_ue8m0_cast);
        return Tensor::FromDLPack(at::toDLPack(result));
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(preprocess_sf, dg_preprocess_sf);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_tensor, dg_get_mn_major_tma_aligned_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor, dg_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(transform_sf_into_required_layout, dg_transform_sf_into_required_layout);

#endif  // DG_TENSORMAP_COMPATIBLE

// ---------------------------------------------------------------------------
// cuBLASLt GEMMs (always available)
// ---------------------------------------------------------------------------
void dg_cublaslt_gemm_nt(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_nt(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_nn(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_nn(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_tn(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_tn(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}
void dg_cublaslt_gemm_tt(TensorView a, TensorView b, TensorView d, Optional<TensorView> c) {
    auto c_val = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::cublaslt_gemm_tt(
        convert_to_torch_tensor(a),
        convert_to_torch_tensor(b),
        convert_to_torch_tensor(d),
        c_val
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nt, dg_cublaslt_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_nn, dg_cublaslt_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tn, dg_cublaslt_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_gemm_tt, dg_cublaslt_gemm_tt);

// ---------------------------------------------------------------------------
// FP8/FP4 GEMMs and BF16 GEMMs
// ---------------------------------------------------------------------------
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

void dg_fp8_fp4_gemm_nt(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_nt(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_nn(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_nn(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_tn(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_tn(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_fp8_fp4_gemm_tt(TensorView a, TensorView a_sf,
                        TensorView b, TensorView b_sf,
                        TensorView d,
                        Optional<TensorView> c,
                        Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                        Optional<Tuple<int64_t, int64_t>> recipe_a,
                        Optional<Tuple<int64_t, int64_t>> recipe_b,
                        std::string compiled_dims,
                        bool disable_ue8m0_cast) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::fp8_fp4_gemm_tt(std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
                           std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
                           convert_to_torch_tensor(d), c_opt,
                           recipe_opt, recipe_a_opt, recipe_b_opt,
                           compiled_dims, disable_ue8m0_cast);
}

void dg_m_grouped_fp8_fp4_gemm_nt_contiguous(TensorView a, TensorView a_sf,
                                             TensorView b, TensorView b_sf,
                                             TensorView d,
                                             TensorView grouped_layout,
                                             Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                             Optional<Tuple<int64_t, int64_t>> recipe_a,
                                             Optional<Tuple<int64_t, int64_t>> recipe_b,
                                             std::string compiled_dims,
                                             bool disable_ue8m0_cast,
                                             bool use_psum_layout,
                                             Optional<int64_t> expected_m_for_psum_layout) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    auto expected_m_for_psum_layout_opts = expected_m_for_psum_layout.has_value()? std::make_optional((int) expected_m_for_psum_layout.value()) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast,
        use_psum_layout, expected_m_for_psum_layout_opts
    );
}

void dg_m_grouped_fp8_fp4_gemm_nn_contiguous(TensorView a, TensorView a_sf,
                                             TensorView b, TensorView b_sf,
                                             TensorView d,
                                             TensorView grouped_layout,
                                             Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                             Optional<Tuple<int64_t, int64_t>> recipe_a,
                                             Optional<Tuple<int64_t, int64_t>> recipe_b,
                                             std::string compiled_dims,
                                             bool disable_ue8m0_cast,
                                             bool use_psum_layout) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nn_contiguous(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(grouped_layout),
        recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast, use_psum_layout
    );
}

void dg_m_grouped_fp8_fp4_gemm_nt_masked(TensorView a, TensorView a_sf,
                                 TensorView b, TensorView b_sf,
                                 TensorView d,
                                 TensorView masked_m,
                                 int64_t expected_m,
                                 Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                 Optional<Tuple<int64_t, int64_t>> recipe_a,
                                 Optional<Tuple<int64_t, int64_t>> recipe_b,
                                 std::string compiled_dims,
                                 bool disable_ue8m0_cast) {
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    auto recipe_a_opt = recipe_a.has_value() ? std::make_optional(std::make_tuple((int)recipe_a.value().get<0>(), (int)recipe_a.value().get<1>())) : std::nullopt;
    auto recipe_b_opt = recipe_b.has_value() ? std::make_optional(std::make_tuple((int)recipe_b.value().get<0>(), (int)recipe_b.value().get<1>())) : std::nullopt;
    gemm::m_grouped_fp8_fp4_gemm_nt_masked(
        std::make_pair(convert_to_torch_tensor(a), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), convert_to_torch_tensor(masked_m),
        (int) expected_m, recipe_opt, recipe_a_opt, recipe_b_opt,
        compiled_dims, disable_ue8m0_cast
    );
}


void dg_bf16_gemm_nt(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_nt(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims);
}

void dg_bf16_gemm_nn(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_nn(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims);
}

void dg_bf16_gemm_tn(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_tn(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims);
}

void dg_bf16_gemm_tt(TensorView a, TensorView b, TensorView d,
                     Optional<TensorView> c,
                     std::string compiled_dims) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    gemm::bf16_gemm_tt(convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, compiled_dims);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nt, dg_fp8_fp4_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_nn, dg_fp8_fp4_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tn, dg_fp8_fp4_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_gemm_tt, dg_fp8_fp4_gemm_tt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nt_contiguous, dg_m_grouped_fp8_fp4_gemm_nt_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nn_contiguous, dg_m_grouped_fp8_fp4_gemm_nn_contiguous);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(m_grouped_fp8_fp4_gemm_nt_masked, dg_m_grouped_fp8_fp4_gemm_nt_masked);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nt, dg_bf16_gemm_nt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_nn, dg_bf16_gemm_nn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tn, dg_bf16_gemm_tn);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tt, dg_bf16_gemm_tt);

// Einsum
void dg_einsum(std::string expr, TensorView a, TensorView b, TensorView d,
               Optional<TensorView> c, bool use_cublaslt) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    einsum::einsum(expr, convert_to_torch_tensor(a), convert_to_torch_tensor(b), convert_to_torch_tensor(d), c_opt, use_cublaslt);
}

void dg_fp8_einsum(std::string expr,
                   TensorView a_data, TensorView a_sf,
                   TensorView b_data, TensorView b_sf,
                   TensorView d,
                   Optional<TensorView> c,
                   Tuple<int64_t, int64_t, int64_t> recipe) {
    auto c_opt = c.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(c.value())) : std::nullopt;
    auto recipe_opt = std::make_tuple((int)recipe.get<0>(), (int)recipe.get<1>(), (int)recipe.get<2>());
    einsum::fp8_einsum(expr, std::make_pair(convert_to_torch_tensor(a_data), convert_to_torch_tensor(a_sf)),
                       std::make_pair(convert_to_torch_tensor(b_data), convert_to_torch_tensor(b_sf)),
                       convert_to_torch_tensor(d), c_opt, recipe_opt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(einsum, dg_einsum);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_einsum, dg_fp8_einsum);

// Hyperconnection
void dg_tf32_hc_prenorm_gemm(TensorView a, TensorView b, TensorView d,
                              TensorView sqr_sum, Optional<int64_t> num_splits) {
    auto ns = num_splits.has_value() ? std::make_optional(static_cast<int>(num_splits.value())) : std::nullopt;
    hyperconnection::tf32_hc_prenorm_gemm(convert_to_torch_tensor(a), convert_to_torch_tensor(b),
                                          convert_to_torch_tensor(d), convert_to_torch_tensor(sqr_sum), ns);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tf32_hc_prenorm_gemm, dg_tf32_hc_prenorm_gemm);

// Attention
void dg_fp8_gemm_nt_skip_head_mid(TensorView a_data, TensorView a_sf,
                                   TensorView b_data, TensorView b_sf,
                                   TensorView d,
                                   Tuple<int64_t, int64_t, int64_t> head_splits,
                                   Optional<Tuple<int64_t, int64_t, int64_t>> recipe,
                                   std::string compiled_dims,
                                   bool disable_ue8m0_cast) {
    auto head_splits_opt = std::make_tuple((int)head_splits.get<0>(), (int)head_splits.get<1>(), (int)head_splits.get<2>());
    auto recipe_opt = recipe.has_value()? std::make_optional(std::make_tuple((int)recipe.value().get<0>(), (int)recipe.value().get<1>(), (int)recipe.value().get<2>())) : std::nullopt;
    attention::fp8_gemm_nt_skip_head_mid(
        std::make_pair(convert_to_torch_tensor(a_data), convert_to_torch_tensor(a_sf)),
        std::make_pair(convert_to_torch_tensor(b_data), convert_to_torch_tensor(b_sf)),
        convert_to_torch_tensor(d), head_splits_opt, recipe_opt, compiled_dims, disable_ue8m0_cast);
}

Tensor dg_fp8_mqa_logits(TensorView q, TensorView kv_data, TensorView kv_sf,
                        TensorView weights, TensorView cu_seq_len_k_start,
                        TensorView cu_seq_len_k_end,
                        bool clean_logits, int64_t max_seqlen_k) {
    auto result = attention::fp8_mqa_logits(
        convert_to_torch_tensor(q),
        std::make_pair(convert_to_torch_tensor(kv_data), convert_to_torch_tensor(kv_sf)),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end), clean_logits, static_cast<int>(max_seqlen_k));
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_get_paged_mqa_logits_metadata(TensorView context_lens, int64_t block_kv,
                                       int64_t num_sms, Optional<TensorView> indices) {
    auto indices_val = indices.has_value()?
        std::optional<torch::Tensor>(convert_to_torch_tensor(indices.value()))
        : std::nullopt;
    auto result = attention::get_paged_mqa_logits_metadata(
        convert_to_torch_tensor(context_lens), static_cast<int>(block_kv),
        static_cast<int>(num_sms), indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_paged_mqa_logits(TensorView q, TensorView fused_kv_cache,
                              TensorView weights, TensorView context_lens,
                              TensorView block_table, TensorView schedule_meta,
                              int64_t max_context_len, bool clean_logits,
                              Optional<TensorView> indices) {
    auto indices_val = indices.has_value()? std::optional(convert_to_torch_tensor(indices.value())) : std::nullopt;
    auto result = attention::fp8_paged_mqa_logits(
        convert_to_torch_tensor(q), convert_to_torch_tensor(fused_kv_cache),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(context_lens),
        convert_to_torch_tensor(block_table), convert_to_torch_tensor(schedule_meta),
        static_cast<int>(max_context_len), clean_logits, indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_fp4_mqa_logits(TensorView q, Optional<TensorView> q_sf, TensorView kv_data, TensorView kv_sf,
                            TensorView weights, TensorView cu_seq_len_k_start,
                            TensorView cu_seq_len_k_end, bool clean_logits, int64_t max_seqlen_k,
                            std::string logits_dtype) {
    auto q_sf_val = q_sf.has_value()? std::make_optional(convert_to_torch_tensor(q_sf.value())) : std::nullopt;
    auto result = attention::fp8_fp4_mqa_logits(
        std::make_pair(convert_to_torch_tensor(q), q_sf_val),
        std::make_pair(convert_to_torch_tensor(kv_data), convert_to_torch_tensor(kv_sf)),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(cu_seq_len_k_start),
        convert_to_torch_tensor(cu_seq_len_k_end), clean_logits, static_cast<int>(max_seqlen_k),
        string_to_dtype(logits_dtype));
    return Tensor::FromDLPack(at::toDLPack(result));
}

Tensor dg_fp8_fp4_paged_mqa_logits(TensorView q, Optional<TensorView> q_sf, TensorView fused_kv_cache,
                              TensorView weights, TensorView context_lens,
                              TensorView block_table, TensorView schedule_meta,
                              int64_t max_context_len, bool clean_logits,
                              std::string logits_dtype, Optional<TensorView> indices) {
    auto q_sf_val = q_sf.has_value()? std::make_optional(convert_to_torch_tensor(q_sf.value())) : std::nullopt;
    auto indices_val = indices.has_value()? std::optional(convert_to_torch_tensor(indices.value())) : std::nullopt;
    auto result = attention::fp8_fp4_paged_mqa_logits(
        std::make_pair(convert_to_torch_tensor(q), q_sf_val),
        convert_to_torch_tensor(fused_kv_cache),
        convert_to_torch_tensor(weights), convert_to_torch_tensor(context_lens),
        convert_to_torch_tensor(block_table), convert_to_torch_tensor(schedule_meta),
        static_cast<int>(max_context_len), clean_logits,
        string_to_dtype(logits_dtype), indices_val);
    return Tensor::FromDLPack(at::toDLPack(result));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_gemm_nt_skip_head_mid, dg_fp8_gemm_nt_skip_head_mid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_mqa_logits, dg_fp8_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_paged_mqa_logits_metadata, dg_get_paged_mqa_logits_metadata);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_paged_mqa_logits, dg_fp8_paged_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_mqa_logits, dg_fp8_fp4_mqa_logits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_paged_mqa_logits, dg_fp8_fp4_paged_mqa_logits);

// Mega MoE
int64_t dg_get_token_alignment_for_mega_moe() {
    return (int64_t)mega::get_token_alignment_for_mega_moe();
}

Tuple<int64_t, TypedFunction<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(TensorView)>>
dg_get_symm_buffer_size_for_mega_moe(int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank, int64_t num_topk, int64_t hidden,
                                    int64_t intermediate_hidden, bool use_fp8_dispatch, std::string activation) {
    auto [num_bytes, fn] = mega::get_symm_buffer_size_for_mega_moe(
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(hidden),
        static_cast<int>(intermediate_hidden),
        use_fp8_dispatch,
        activation
    );

    auto slice_input_buffers = [=](TensorView buffer) {
        auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] =  fn(convert_to_torch_tensor(buffer));
        return Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
            Tensor::FromDLPack(at::toDLPack(x.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(x_sf)),
            Tensor::FromDLPack(at::toDLPack(topk_idx)),
            Tensor::FromDLPack(at::toDLPack(topk_weights)),
            Tensor::FromDLPack(at::toDLPack(l1_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l1_acts_sf)),
            Tensor::FromDLPack(at::toDLPack(l2_acts.view(at::kChar))),
            Tensor::FromDLPack(at::toDLPack(l2_acts_sf))
        );
    };
    return Tuple<int64_t, TypedFunction<Tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(TensorView)>>(
        num_bytes, slice_input_buffers);
}

void dg_fp8_fp4_mega_moe(TensorView y, TensorView l1_weights, TensorView l1_weights_sf, TensorView l2_weights, TensorView l2_weights_sf,
                        Optional<TensorView> cumulative_local_expert_recv_stats, TensorView sym_buffer, Array<int64_t> sym_buffer_ptrs,
                        int64_t rank_idx, int64_t num_max_tokens_per_rank, int64_t num_experts, int64_t num_topk,
                        Tuple<int64_t, int64_t, int64_t> recipe, std::string activation, Optional<double> activation_clamp_opt, bool fast_math) {
    auto c_val = cumulative_local_expert_recv_stats.has_value()? std::optional<torch::Tensor>(convert_to_torch_tensor(cumulative_local_expert_recv_stats.value())) : std::nullopt;
    auto act_clamp_opt_val = activation_clamp_opt.has_value()? std::optional<float>(static_cast<float>(activation_clamp_opt.value())) : std::nullopt;
    std::vector<int64_t> sym_buffer_ptrs_val;
    sym_buffer_ptrs_val.reserve(sym_buffer_ptrs.size());

    for (Array<int64_t>::iterator it = sym_buffer_ptrs.begin(); it != sym_buffer_ptrs.end(); ++it) {
        sym_buffer_ptrs_val.push_back(*it);
    }
    auto [recipe_a, recipe_b, recipe_c] = recipe;
    auto recipe_val = std::make_tuple(static_cast<int>(recipe_a), static_cast<int>(recipe_b), static_cast<int>(recipe_c));

    mega::fp8_fp4_mega_moe(
        convert_to_torch_tensor(y),
        std::make_pair(convert_to_torch_tensor(l1_weights), convert_to_torch_tensor(l1_weights_sf)),
        std::make_pair(convert_to_torch_tensor(l2_weights), convert_to_torch_tensor(l2_weights_sf)),
        c_val, convert_to_torch_tensor(sym_buffer), sym_buffer_ptrs_val, static_cast<int>(rank_idx),
        static_cast<int>(num_max_tokens_per_rank), static_cast<int>(num_experts),
        static_cast<int>(num_topk), recipe_val, activation, act_clamp_opt_val, fast_math
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_token_alignment_for_mega_moe, dg_get_token_alignment_for_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_symm_buffer_size_for_mega_moe, dg_get_symm_buffer_size_for_mega_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_fp4_mega_moe, dg_fp8_fp4_mega_moe);


#endif  // DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
