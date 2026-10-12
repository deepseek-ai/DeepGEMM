#pragma once

#include <torch/python.h>
#include <cstdint>
#include <hggc_fp8.h>
#include "cute/tensor.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/layout_type_name.hpp"
#include "cute/arch/mma.hpp"
#include "../heuristics/common_bf16.hpp"
// #include "../heuristics/gemm_search_space.hpp"
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/scheduler/densegemm_scheduler_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

class BF16GemmCutlass3Runtime final : public LaunchRuntime<BF16GemmCutlass3Runtime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        cutlass::bfloat16_t const* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::bfloat16_t const* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;               ///< scales accumulators
        float beta = 0.0f;                ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
        float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
        float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

    // Epilogue
    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n, num_groups, num_stages;
        std::string gemm_type, kernel_type, kernel_name;
        bool enable_sbo_overlap;
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler{};
        int32_t* signal{nullptr};
    };

    using CollectiveMainloopParams = MainLoopArguments;

    // One epilogue-params layout per device generation (must match the device Params bytes).
    using EpilogueParamNoTsm = EpilogueArgs;  // PPU1.5: the args are the params, as-is
    struct EpilogueParamWithTsm {
        struct {} thread{};
        cutlass::bfloat16_t* ptr_C{};
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C{};
        cutlass::bfloat16_t* ptr_D{};
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D{};
    };

    template <typename EpilogueParamsT>
    struct GemmKernelParamsT {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        EpilogueParamsT collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        TileSchedulerArguments scheduler{};
        void* workspace{nullptr}; // workspace,
        int32_t* signal{nullptr};
    };
    // Same prefix for both generations; the member filled by to_underlying_arguments_rtc must match the device compile.
    union KernelParams {
        GemmKernelParamsT<EpilogueParamWithTsm> with_tsm;
        GemmKernelParamsT<EpilogueParamNoTsm> no_tsm;
    };
    using GemmKernelParams = KernelParams;

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        // Both epilogues only convert the accumulator; enforce the {alpha=1, beta=0} contract.
        DG_HOST_ASSERT(args.epilogueargs.callback.alpha == 1.0f && args.epilogueargs.callback.alpha_ptr == nullptr &&
                       args.epilogueargs.callback.beta == 0.0f && args.epilogueargs.callback.beta_ptr == nullptr);

        // Same decision source as the injected kWithTsmEpilogue; must match the device compile.
        // Initialize the selected union member directly to start its lifetime.
        if (is_ppu1v5_device()) {
            return KernelParams{.no_tsm = GemmKernelParamsT<EpilogueParamNoTsm>{
                args.mode, problem_shape, args.mainloopargs, args.epilogueargs,
                hw_info, args.scheduler, workspace, args.signal}};
        } else {
            // Map the flat inputs on the host (once per launch); the device converts nothing.
            return KernelParams{.with_tsm = GemmKernelParamsT<EpilogueParamWithTsm>{
                args.mode, problem_shape, args.mainloopargs,
                {{},
                 args.epilogueargs.ptr_C, args.epilogueargs.stride_C,
                 args.epilogueargs.ptr_D, args.epilogueargs.stride_D},
                hw_info, args.scheduler, workspace, args.signal}};
        }
    }

    static std::string generate_impl(const Args& args) {
        // Query device hardware constants from the driver and inject into generated kernel (see gemm_occ_model.cuh).
        const PpuHwParams& hw = PpuHwParams::instance();
        // One snapshot drives both the union member read and the injected epilogue flag.
        const bool is_ppu1v5 = is_ppu1v5_device();
        // Read the common head (mode/problem_shape/mainloop) through the matching member.
        const GemmProblemSize problem_shape = is_ppu1v5 ? args.kernel_params.no_tsm.problem_shape
                                                        : args.kernel_params.with_tsm.problem_shape;
        // Injected into the generated source; same source as to_underlying_arguments_rtc.
        const bool with_tsm_epilogue = not is_ppu1v5;
        // sizeof of what the driver copies; the device asserts it equals its own Params size.
        const auto host_params_size = with_tsm_epilogue ? sizeof(args.kernel_params.with_tsm)
                                                        : sizeof(args.kernel_params.no_tsm);
        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_gemm_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

// Injected device hardware constants (host-side hggcDeviceGetAttribute query).
constexpr int kHwTsmPerCu         = {13};
constexpr int kHwMaxThreadsPerCta = {14};
constexpr int kHwMaxWarpsPerCu    = {15};
constexpr int kHwTotalVregPerCu   = {16};

constexpr int SHAPE_N = {0};
constexpr int SHAPE_K = {1};
constexpr int BLOCK_M = {2};
constexpr int BLOCK_N = {3};
constexpr int BLOCK_K = {4};
constexpr int NUM_GROUPS = {5};
constexpr int WARP_M = {6};
constexpr int WARP_N = {7};
constexpr int STAGES = {8};

static constexpr GemmType kGemmType = GemmType::{9};
static constexpr KernelType kKernelType = KernelType::{10}; //Default;
static constexpr bool kEnableSboOverlap = {11};// false;

using ElementA    = cutlass::bfloat16_t;
using ElementB    = cutlass::bfloat16_t;
using ElementC    = cutlass::bfloat16_t;
using LayoutA     = cutlass::layout::RowMajor;
using LayoutB     = cutlass::layout::ColumnMajor;
using LayoutC     = cutlass::layout::RowMajor;
using ElementD    = ElementC;
using LayoutD     = cutlass::layout::RowMajor;
using ElementCompute      = float;
using ElementScalar       = ElementCompute;
using LinearCombOutType   = ElementD;
using OperatorClass = cutlass::arch::OpClassTensorOp;
#if __HGGC_ARCH__ == 100
using ArchTag = cutlass::arch::PPU0010;
#else
using ArchTag = cutlass::arch::PPU0015;
#endif

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape = Shape<Int<WARP_M>, Int<WARP_N>, Int<BLOCK_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, cutlass::bfloat16_t, cutlass::bfloat16_t, float>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, _1>>,  // 1x4x1 thread group
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, _16>>;       // 1x1x1 value group

static constexpr int N_EXPAND = kKernelType == KernelType::MultistageOnN && (SHAPE_N % (BLOCK_N) == 0) ? KernelAiuMultistageOnN::N_EXPAND : 1;

using KernelSchedule = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    KernelAiuMultistageOverlapMainloop,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      KernelAiuMultistageOverlapPrologue,
      cute::conditional_t<
        kKernelType == KernelType::MultistageOnN,
        KernelAiuMultistageOnN,
        cutlass::gemm::KernelAiuMultistage>>>;
using DispatchPolicy = cute::conditional_t<
    kKernelType == KernelType::OverlapMainloop,
    cutlass::gemm::MainloopPPUOverlapMainloop<STAGES, KernelSchedule>,
    cute::conditional_t<
      kKernelType == KernelType::OverlapPrologue,
      cutlass::gemm::MainloopPPUOverlapPrologue<STAGES, KernelSchedule>,
      cutlass::gemm::MainloopPPUAiuOpt<STAGES, KernelSchedule>>>;

static constexpr bool TransA = cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value ? false : true;
static constexpr bool TransB = cutlass::platform::is_same<LayoutB, cutlass::layout::ColumnMajor>::value ? false : true;
static constexpr int TSM_LD_NUM = BLOCK_M == 8 ? 2 : 4;

static constexpr int SmemLayoutStageStrideA = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = kKernelType == KernelType::OverlapMainloop || kKernelType == KernelType::OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementA, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;
// using t1 = DefaultOperandB::xhzhao;
// A
using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom; // M, K
using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
// B
using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom; // N, K
using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

// Mainloop
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    ArchTag,
    DispatchPolicy, TileShape,
    ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
>;

// Epilogue
static constexpr bool IsAlignedN = SHAPE_N % BLOCK_N == 0 ? true : false;
using CollectiveEpilogue_noTsm = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;

// TSM epilogue for alpha=1, beta=0;
static constexpr int AlignmentC = 16 / sizeof(ElementC);
using DefaultOperation = cutlass::epilogue::fusion::PPUEVT<
    cutlass::epilogue::fusion::PPUCompute<cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
                                       cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::PPUAccFetch>;
using EpilogueSchedule = typename cutlass::epilogue::EpilogueSimtVectorized;
using CollectiveEpilogue_withTsm = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    TileShape, WarpShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueSchedule,
    DefaultOperation
>::CollectiveOp;

// Injected epilogue choice (is_ppu1v5_device()); matches to_underlying_arguments_rtc.
static constexpr bool kWithTsmEpilogue = {17};
using CollectiveEpilogue = typename cutlass::platform::conditional<
    kWithTsmEpilogue,
    CollectiveEpilogue_withTsm,
    CollectiveEpilogue_noTsm
>::type;

using TileScheduler = DeepGemmScheduler<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N * N_EXPAND, NUM_GROUPS>;
using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler,
    kEnableSboOverlap>;

// Host/device Params ABI check (the driver copies sizeof(device Params) bytes from the host struct).
static_assert(sizeof(GemmKernel::Params) == {18}, "host/device kernel Params size mismatch");

// Derive MinBlocksPerMultiprocessor from GemmOccModel with device hardware
// constants injected by the host-side query (see kHw* above).
using GemmOcc = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_K, GemmKernel::SharedStorageSize,
                            cute::sizeof_bits_v<ElementA>, cute::sizeof_bits_v<ElementB>,
                            cute::sizeof_bits_v<ElementCompute>,
                            kHwTsmPerCu, kHwMaxThreadsPerCta, kHwMaxWarpsPerCu, kHwTotalVregPerCu>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)
__global__ void {12}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}}
)",
            cute::get<1>(problem_shape), cute::get<2>(problem_shape),
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k, args.launch_info.num_groups,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.num_stages, args.launch_info.gemm_type,
            args.launch_info.kernel_type, args.launch_info.enable_sbo_overlap, args.launch_info.kernel_name,
            hw.tsm_per_cu, hw.max_threads_per_cta, hw.max_warps_per_cu, hw.total_vreg_per_cu,
            with_tsm_epilogue, host_params_size);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// Dedicated BF16 DenseGemm runtime (standalone kernel, no SHAPE_N/K template params).
// Architecture-aligned with DenseINT8GemmCutlass3Runtime. The MoE grouped-path runtime
// (BF16GemmCutlass3Runtime) is intentionally left untouched so that m_grouped_bf16_gemm.hpp
// stays byte-identical to baseline 753f28f.
class DenseBF16GemmCutlass3Runtime final : public LaunchRuntime<DenseBF16GemmCutlass3Runtime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        cutlass::bfloat16_t const* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::bfloat16_t const* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;               ///< scales accumulators
        float beta = 0.0f;                ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
        float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
        float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

    // Epilogue
    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;           // WARP_K tile size for K-dim split (= block_k / WarpOnK). Used in WarpShape as Int<WARP_K>.
        bool dense_s2_opt;
        int num_stages;
        std::string gemm_type, kernel_type, kernel_name;
        bool enable_sbo_overlap;
        bool overlap_prologue; // next-tile prologue overlap, enabled by host when wave > 3 (acblas parity)
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler{};
        // Carried for API uniformity with bf16/fp8 GemmArguments and the grouped BF16 paths.
        // The dense path leaves it null; to_underlying_arguments_rtc ignores it (GemmKernelParams has no signal field).
        int32_t* signal{nullptr};
    };

    using CollectiveMainloopParams = MainLoopArguments;
    using CollectiveEpilogueParams = EpilogueArgs;

    struct GemmKernelParams {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        CollectiveEpilogueParams collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler;
        void* workspace{nullptr}; // workspace,
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace};
    }

    static std::string generate_impl(const Args& args) {
        // Query device hardware constants from the driver and inject into generated kernel (see gemm_occ_model.cuh).
        const PpuHwParams& hw = PpuHwParams::instance();
        // Byte-identical to the original BF16DenseGemmHelper::generate_code output:
        // IsAlignedN is derived from the runtime problem N and block_n (n % block_n == 0).
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % args.launch_info.block_n == 0);
        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_densegemm_cutlass3.cuh>
#include <deep_gemm/common/gemm_occ_model.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

// Injected device hardware constants (host-side hggcDeviceGetAttribute query).
constexpr int kHwTsmPerCu         = {11};
constexpr int kHwMaxThreadsPerCta = {12};
constexpr int kHwMaxWarpsPerCu    = {13};
constexpr int kHwTotalVregPerCu   = {14};

constexpr int BLOCK_M = {0};
constexpr int BLOCK_N = {1};
constexpr int BLOCK_K = {2};
constexpr int WARP_M = {3};
constexpr int WARP_N = {4};
constexpr int WARP_K = {5};
constexpr int kNumStages = {6};

using ElementAB = cutlass::bfloat16_t;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementC = cutlass::bfloat16_t;
using ElementD = ElementC;
using LayoutD = cutlass::layout::RowMajor;
using ElementCompute = float;
using ArchTag = cutlass::arch::PPU0015;

using TileShape = Shape<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using WarpShape_t = Shape<Int<WARP_M>, Int<WARP_N>, Int<WARP_K>>;
static constexpr int WarpOnM = BLOCK_M / WARP_M;
static constexpr int WarpOnN = BLOCK_N / WARP_N;
static constexpr int WarpOnK = BLOCK_K / WARP_K;
static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be divisible by WARP_K");

using MmaInst = typename cutlass::gemm::config::GetAiuMmaInst<ArchTag, ElementAB, ElementAB, ElementAcc>::type;
using TiledMma = TiledMMA<
    MMA_Atom<MmaInst>,
    Layout<Shape<Int<WarpOnM>, Int<WarpOnN>, Int<WarpOnK>>>,
    Tile<Int<WarpOnM * 16>, Int<WarpOnN * 16>, Int<WarpOnK * 16>>>;

using KernelSchedule = cutlass::gemm::KernelAiuMultistage;
using DispatchPolicy = cutlass::gemm::MainloopPPUAiuOpt<kNumStages, KernelSchedule, {7}, {10}>;

static constexpr bool TransA = false;
static constexpr bool TransB = false;

// OverlapPrologue interleaves A/B stages in one contiguous smem region, so the
// TSM swizzle-load atom needs the combined stage stride (validated grouped-path pattern).
static constexpr bool OverlapPrologue = {10};
static constexpr int SmemLayoutStageStrideA = OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_M * BLOCK_K;
static constexpr int SmemLayoutStageStrideB = OverlapPrologue ? (BLOCK_M + BLOCK_N) * BLOCK_K : BLOCK_N * BLOCK_K;
using DefaultOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementAB, TransA, Int<BLOCK_M>, Int<BLOCK_K>, false, SmemLayoutStageStrideA>;
using DefaultOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<ArchTag, ElementAB, TransB, Int<BLOCK_N>, Int<BLOCK_K>, true, SmemLayoutStageStrideB>;

using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    ArchTag,
    DispatchPolicy, TileShape,
    ElementAB, cutlass::detail::TagToStrideA_t<LayoutA>,
    ElementAB, cutlass::detail::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity
>;

static constexpr bool IsAlignedN = {8};
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::detail::TagToStrideA_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault,
    IsAlignedN>;

using TileScheduler = DenseGemmScheduler<BLOCK_M, BLOCK_N>;
using GemmKernel = cutlass::gemm::kernel::BF16DenseGemmKernel<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;

using GemmOcc = GemmOccModel<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, GemmKernel::SharedStorageSize,
                            cute::sizeof_bits_v<ElementAB>, cute::sizeof_bits_v<ElementAB>,
                            cute::sizeof_bits_v<ElementCompute>,
                            kHwTsmPerCu, kHwMaxThreadsPerCta, kHwMaxWarpsPerCu, kHwTotalVregPerCu>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)
__global__ void {9}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  GemmKernel op;
  op(params, smem);
}}
}}
)",
            args.launch_info.block_m, args.launch_info.block_n, args.launch_info.block_k,
            args.launch_info.warp_m, args.launch_info.warp_n, args.launch_info.warp_k,
            args.launch_info.num_stages,
            args.launch_info.dense_s2_opt ? "true" : "false",
            is_aligned_n ? "true" : "false",
            args.launch_info.kernel_name,
            args.launch_info.overlap_prologue ? "true" : "false",
            hw.tsm_per_cu, hw.max_threads_per_cta, hw.max_warps_per_cu, hw.total_vreg_per_cu);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& configs, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, configs, args.kernel_params));
    }
};

class DenseBF16GemmCuteFreeRuntime final : public LaunchRuntime<DenseBF16GemmCuteFreeRuntime> {
public:
    using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;
    using GemmProblemSize = cute::tuple<int32_t, int32_t, int32_t, int32_t>;

    struct MainLoopArguments {
        cutlass::bfloat16_t const* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        cutlass::bfloat16_t const* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
    };

    struct LinearCombinationArgs {
        float alpha = 1.0f;               ///< scales accumulators
        float beta = 0.0f;                ///< scales source tensor
        float const* alpha_ptr = nullptr; ///< pointer to accumulator scalar - if not null, loads it from memory
        float const* beta_ptr = nullptr;  ///< pointer to source scalar - if not null, loads it from memory
        float const* const* alpha_ptr_array = nullptr; ///< array of pointers to accumulator scalar per group/batch
        float const* const* beta_ptr_array = nullptr;  ///< array of pointers to source scalar per group/batch
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

    // Epilogue
    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int block_m, block_n, block_k, warp_m, warp_n;
        int warp_k;           // WARP_K tile size for K-dim split (= block_k / WarpOnK). Used in WarpShape as Int<WARP_K>.
        bool dense_s2_opt;
        int num_stages;
        std::string gemm_type, kernel_type, kernel_name;
        bool enable_sbo_overlap;
        bool overlap_prologue;
    };

    struct GemmArguments {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        MainLoopArguments mainloopargs;
        EpilogueArgs epilogueargs;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler{};
        // Carried for API uniformity with bf16/fp8 GemmArguments and the grouped BF16 paths.
        // The dense path leaves it null; to_underlying_arguments_rtc ignores it (GemmKernelParams has no signal field).
        int32_t* signal{nullptr};
    };

    using CollectiveMainloopParams = MainLoopArguments;
    using CollectiveEpilogueParams = EpilogueArgs;

    struct GemmKernelParams {
        GemmUniversalMode mode;
        GemmProblemSize problem_shape;
        CollectiveMainloopParams collective_mainloop_params;
        CollectiveEpilogueParams collective_epilogue_params;
        cutlass::KernelHardwareInfo hw_info;
        DenseGemmTileSchedulerArguments scheduler;
        void* workspace{nullptr}; // workspace,
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmKernelParams kernel_params;
    };

    static GemmKernelParams to_underlying_arguments_rtc(GemmArguments args, void* workspace) {
        auto problem_shape = args.problem_shape;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);

        int sm_count = args.hw_info.cu_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST(
                "  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

        return {args.mode, problem_shape,  args.mainloopargs, args.epilogueargs,
                hw_info,   args.scheduler, workspace};
    }

    static std::string generate_impl(const Args& args) {
        const auto& info = args.launch_info;
        const int shape_n = cute::get<1>(args.kernel_params.problem_shape);
        const bool is_aligned_n = (shape_n % info.block_n == 0);

        return fmt::format(
            R"(
#define BF16_HGRTC
#include <deep_gemm/impls/bf16_gemm_cute_free.cuh>

namespace deep_gemm {{

using GemmTypeTag = std::integral_constant<GemmType, GemmType::{0}>;

using Kernel = BF16GemmCuteFreeKernel<{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {11}, {9}, GemmTypeTag>;

}} // namespace deep_gemm

extern "C"
__launch_bounds__(deep_gemm::Kernel::MaxThreadsPerBlock,
                  deep_gemm::Kernel::MinBlocksPerMultiprocessor)
__global__ void {10}(typename deep_gemm::Kernel::Params params) {{
  extern __shared__ char smem[];
  deep_gemm::Kernel kernel;
  kernel(params, smem);
}}
)",
            info.gemm_type,
            info.block_m, info.block_n, info.block_k,
            info.warp_m, info.warp_n, info.warp_k, info.num_stages,
            info.dense_s2_opt ? "true" : "false",
            is_aligned_n ? "true" : "false",
            info.kernel_name,
            info.overlap_prologue ? "true" : "false");
    }

};

// Dense GEMV (m == 1) JIT runtime. Host-side mirror of the device GemvDenseArgs
// in deep_gemm/impls/gemv_dense.cuh -- the two PODs must stay field-for-field
// identical (same convention as GemvRuntime::GemvtArgs vs gemvt.cuh), since the
// kernel receives it by value.
class DenseGemvRuntime final : public LaunchRuntime<DenseGemvRuntime> {
public:
    struct GemvDenseArgs {
        int N;
        int K;
        const void* x_ptr;   // lhs [1, K]
        const void* w_ptr;   // rhs [N, K] row-major
        void* y_ptr;         // out [1, N]
        int64_t stride_wn;   // W row stride (elements, == K for contiguous rhs)
    };

    struct LaunchInfo {
        int block_x, block_y, k_per_thread;
        int m;   // 1 or 2
        int npt; // W rows per thread (m == 2 policy: 2 shares the x load pair)
        std::string kernel_name;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemvDenseArgs kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        const bool is_m2 = args.launch_info.m == 2;
        const char* entry = is_m2 ? "gemv_dense_m2_kernel_impl" : "gemv_dense_kernel_impl";
        // m2 entry carries a trailing NPT template arg (1 or 2 W rows/thread); m1 has none.
        const std::string npt_arg = is_m2 ? fmt::format(", {}", args.launch_info.npt) : "";
        return fmt::format(
            R"(
#include <deep_gemm/impls/gemv_dense.cuh>

namespace deep_gemm {{

extern "C" __global__
void {}(const GemvDenseArgs args) {{
    {}<__ppu_bfloat16, __ppu_bfloat16, float,
       int4, int4, {}, {}, {}{}>(args);
}}

}}
)",
            args.launch_info.kernel_name, entry,
            args.launch_info.block_x, args.launch_info.block_y, args.launch_info.k_per_thread,
            npt_arg);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

// bf16 dense GEMV fast path (m == 1 or 2). Returns true when launched; false
// means the shape is uncovered and the caller must fall back to the tile path.
static bool gemv_dense_bf16(const torch::Tensor& lhs, const torch::Tensor& rhs,
                            const torch::Tensor& out, const int& m, const int& n, const int& k) {
    DG_HOST_ASSERT(m == 1 || m == 2);
    TORCH_CHECK(lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous(),
                "dense gemv requires contiguous tensors");

    int block_x = 0, block_y = 0, k_per_thread = 0, npt = 1;
    if (!deep_gemm_bf16_common::dense_gemv_select_configs(m, n, k, rhs.data_ptr(), lhs.data_ptr(),
                                                          block_x, block_y, k_per_thread, npt))
        return false;

    DenseGemvRuntime::GemvDenseArgs params;
    params.N = n;
    params.K = k;
    params.x_ptr = lhs.data_ptr<at::BFloat16>();
    params.w_ptr = rhs.data_ptr<at::BFloat16>();
    params.y_ptr = out.data_ptr<at::BFloat16>();
    params.stride_wn = k;

    dim3 block(block_x, block_y);
    dim3 grid(ceil_div(n, block_y * npt));

    // m == 2 -> dual-accumulator variant; npt == 2 walks two W rows/thread sharing
    // (x0, x1), each block owning block_y * npt rows. Names split so a trace shows npt.
    const char* kernel_name = m == 1 ? "gemv_dense_bf16"
                                     : (npt == 2 ? "gemv_dense_m2r2_bf16" : "gemv_dense_m2_bf16");
    auto args = DenseGemvRuntime::Args{
        .launch_info = {block_x, block_y, k_per_thread, m, npt, kernel_name},
        .launch_args = {grid, block, 0},
        .kernel_params = params,
    };
    const auto& code = DenseGemvRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;

    static constexpr GemmType kGemmType = GemmType::DenseGemm;
    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16"), 1, m, n, k, 0, nullptr,
                                  current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);
    DenseGemvRuntime::launch(runtime, args);
    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[DenseGemm_BF16_GemV:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", 1, m, n, k);
        printf("BlockX:%d, BlockY:%d, k_per_thread:%d, npt:%d\n", block_x, block_y, k_per_thread,
               npt);
        printf("threadblock_count:%d, vreg:%d, stack:%d\n", (int)grid.x, int(numRegs), int(localSize));
    }
    return true;
}

// 8-element public boundary ConfigTuple (matches Python/vLLM stable contract).
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
static void bf16_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out, const int& m,
                      const int& n, const int& k, std::optional<ConfigTuple> configs = std::nullopt) {
    int num_sms = get_num_sms();
    hggcDeviceProp device_props;
    hggcGetDeviceProperties(&device_props, 0);
    std::vector<int> shape = {m, n, k};
    static constexpr GemmType kGemmType = GemmType::DenseGemm;

    using Config = std::tuple<int, int, int, int, int, int, int, int, std::tuple<int, int, int>>;

    Config cfg;
    if (configs.has_value()) {
        // Explicit config path: unpack the 8-element public tuple; force warp_k = block_k
        // (mirrors deep_gemm/jit_kernels/gemm.py lines 406-408/422;
        // adaptive is NOT re-evaluated here).
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, /*warp_k=*/bk, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        bool shape_large_aligned = true;
        for (int64_t a : shape) {
            if (!(a >= 4096 && (a % 64 == 0))) {
                shape_large_aligned = false;
                break;
            }
        }

        std::string dev_name(device_props.name);
        bool is_ppu0010_device = (dev_name.find("ZW810E") != std::string::npos) || (dev_name.find("ZW810") != std::string::npos);

        if (shape_large_aligned && is_ppu0010_device) {
            cfg = get_gemm_best_configs_v2(shape, 2, num_sms);
        } else {
            auto [ns, bm, bn, bk, wm, wn, nst, sc] = deep_gemm_bf16_common::get_best_configs(m, n, k, 1, num_sms);
            int warp_k = bk;  // default: WarpOnK=1 (non-adaptive)
            if (is_ppu1v5_device() && deep_gemm_adaptive::bf16_adaptive_enabled(m, n, k)) {
                auto adaptive_cfg = deep_gemm_adaptive::get_adaptive_configs(m, n, k, num_sms);
                warp_k = std::get<6>(adaptive_cfg);
            }
            cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, warp_k, nst, sc);
        }
    }
    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages, smem_config] = cfg;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = 1;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int* grouped_layout = nullptr;
    int* layout_info = grouped_layout;

    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    // TODO get hw info from real env
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    bool enable_sbo_overlap = false;
    if (is_ppu1v5_device()) {
        // Dense GEMV fast path (m == 1 or 2): SIMT avoids the tile path's BM=16
        // padding (15/16 lane waste at m==1, 14/16 at m==2). Explicit configs force tile.
        if ((m == 1 || m == 2) && !configs.has_value() && gemv_dense_bf16(lhs, rhs, out, m, n, k)) {
            return;
        }

        // Runtime path selection: DG_USE_CUTE=0 selects original CUTLASS 3 path
        bool use_cute_free = true;
        if (const char* env_ct = std::getenv("DG_USE_CUTE")) {
            if (std::string(env_ct) == "0") use_cute_free = false;
        }

        // Shared occupancy/overlap heuristic (identical for both paths). Computed
        // once here so both the CuteFree and Cutlass3 branches reuse it.
        // Step 1: compute blocks_per_cu purely from the SMEM budget so waves and
        // overlap_on can be decided BEFORE any kernel is compiled -- no probe
        // compile / occupancy query is needed.
        // SMEM-based occupancy heuristic (PPU 890P / ppu1.5: 256 KB shared memory per CU)
        // NOTE: the CuteFree path sets smem_cute_free = SMSIZE, so SMSIZE is equivalent.
        constexpr int kSmemPerCU = 256 * 1024;
        int blocks_per_cu = kSmemPerCU / SMSIZE;
        if (blocks_per_cu < 1) blocks_per_cu = 1;

        // Step 2: compute waves and decide overlap. Overlap-prologue gating
        // (acblas parity): wave = ceil(tiles / (num_sms * blocks_per_cu)); enable
        // next-tile prologue overlap only when wave > 3 AND k <= 2048, since the
        // overlap carries overhead. K-sweep A/B (M=2048, N=4096..40960): benefit
        // is +3.8% at K=1024 and +2.8% at K=2048, but drops to ~1% at K=4096 and
        // to noise level (<=0.2%) at K>=8192 -- the larger mainloop amortizes
        // the hidden prologue. acblas uses the same K<=2048 cutoff.
        const int tiles_m = (m + block_m - 1) / block_m;
        const int tiles_n = (n + block_n - 1) / block_n;
        const int num_tiles = tiles_m * tiles_n;
        const int blocks_per_wave = num_sms_new * blocks_per_cu;
        const int waves = blocks_per_wave > 0 ? (num_tiles + blocks_per_wave - 1) / blocks_per_wave : 0;
        const bool overlap_on = (waves > 3) && (k <= 2048);

        if (use_cute_free) {
        // --- CuteFree path (default) ---
        // warp_k comes from the adaptive tile selector (std::get<6> of
        // adaptive_cfg); WarpOnK (block_k / warp_k) is gated inside
        // get_warp_k (memory-bound region only + env/guard checks).
        const int warps_k = block_k / warp_k;
        dim3 const block_cute_free = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;
        int smem_cute_free = SMSIZE;

        // generate_impl only reads problem_shape (for IsAlignedN) + launch_info.
        const auto gemm_args = DenseBF16GemmCuteFreeRuntime::GemmArguments{
            .mode = cutlass::gemm::GemmUniversalMode::kGemm,
            .problem_shape = {m, n, k, 1},
            .mainloopargs = {input_a, stride_A, input_b, stride_B},
            .epilogueargs =
                {
                    {1.0f, 0.0f},
                    output,
                    stride_D,
                    output,
                    stride_D,
                },
            .hw_info = hw_info,
            .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr},
        };

        DenseBF16GemmCuteFreeRuntime::GemmKernelParams params =
            DenseBF16GemmCuteFreeRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

        dim3 grid_cute_free;
        grid_cute_free.x = blocks_per_cu * num_sms_new;

        // Step 3: build the kernel name for the matching variant (explicit '='
        // assignment avoids _ovlp_ovlp on re-entry) and compile ONLY that one
        // kernel. Single compilation, single launch.
        std::string cute_free_kernel_name =
            overlap_on ? "bf16_deep_gemm_cute_free_ovlp" : "bf16_deep_gemm_cute_free";
        auto args = DenseBF16GemmCuteFreeRuntime::Args{
            .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, /*dense_s2_opt=*/true, num_stages,
                            "DenseGemm", "Default", cute_free_kernel_name, enable_sbo_overlap, overlap_on},
            .launch_args = {grid, block_cute_free, smem_cute_free},
            .kernel_params = params,
        };

        auto code = DenseBF16GemmCuteFreeRuntime::generate(args);
        auto runtime = compiler->build(cute_free_kernel_name, code);
        auto kernel = runtime->kernel;

        // Must stay byte-identical to GemmKernel::Params (cute_free/kernel/gemm.cuh).
        struct CuteFreeParams {
            cutlass::bfloat16_t const* ptr_A;
            cutlass::bfloat16_t const* ptr_B;
            cutlass::bfloat16_t* ptr_D;
            int M, N, K;
            int lda, ldb, ldd;
            DenseGemmTileSchedulerArguments scheduler;
        };

        CuteFreeParams hparams;
        hparams.ptr_A = input_a;
        hparams.ptr_B = input_b;
        hparams.ptr_D = output;
        hparams.M = m;
        hparams.N = n;
        hparams.K = k;
        hparams.lda = k;    // A is row-major MxK
        hparams.ldb = k;    // B is col-major KxN, stored as row-major NxK
        hparams.ldd = n;    // D is row-major MxN
        hparams.scheduler = DenseGemmTileSchedulerArguments{(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr};

        const auto& stream = current_stream();
        auto config = construct_launch_config(kernel, stream, smem_cute_free, grid_cute_free, block_cute_free);

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, nullptr, stream);
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        DG_HGGC_CHECK(launch_kernel(kernel, config, hparams));

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_BF16_CuteFree:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   grid_cute_free.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
                   block_k, warp_m, warp_n, warp_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n", smem_cute_free, int(numRegs), int(localSize));
            printf("waves:%d, overlap_on:%d\n", waves, int(overlap_on));
        }
        } else {
            // --- Cutlass3 path (DG_USE_CUTE=0) ---
            // warp_k is obtained exactly as in the CuteFree path (adaptive
            // tile selector, std::get<6> of adaptive_cfg); all WarpOnK gating
            // (memory-bound region, K-depth gate, fat-tile cap, BK whitelist)
            // lives inside get_warp_k, so both
            // paths see identical BM/BN/BK/S/warp_k for the same shape.
            // The JIT-codegened TiledMMA carries Int<WarpOnK> in its thread
            // layout, so the threadblock must launch WarpOnK extra warp rows;
            // BF16DenseGemmKernel performs the partial-sum combine via
            // warp_on_k_reduce when WarpOnK > 1.
            const int warps_k = block_k / warp_k;
            dim3 const block_cutlass3 = (block_m / warp_m) * (block_n / warp_n) * warps_k * 32;
            const auto gemm_args = DenseBF16GemmCutlass3Runtime::GemmArguments{
                .mode = cutlass::gemm::GemmUniversalMode::kGemm,
                .problem_shape = {m, n, k, 1},
                .mainloopargs = {input_a, stride_A, input_b, stride_B},
                .epilogueargs =
                    {
                        {1.0f, 0.0f},
                        output,
                        stride_D,
                        output,
                        stride_D,
                    },
                .hw_info = hw_info,
                .scheduler = {(uint32_t)m, (uint32_t)n, (uint32_t)k, nullptr},
            };

            DenseBF16GemmCutlass3Runtime::GemmKernelParams params =
                DenseBF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

            // Single compilation: overlap decision is baked into the generated code.
            auto args = DenseBF16GemmCutlass3Runtime::Args{
                .launch_info = {block_m, block_n, block_k, warp_m, warp_n, warp_k, /*dense_s2_opt=*/true, num_stages,
                                "DenseGemm", "Default", "bf16_dense_gemm", false, /*overlap_prologue=*/overlap_on},
                .launch_args = {grid, block_cutlass3, SMSIZE},
                .kernel_params = params,
            };
            auto code = DenseBF16GemmCutlass3Runtime::generate(args);
            auto runtime = compiler->build("bf16_dense_gemm", code);
            auto kernel = runtime->kernel;

            args.launch_args.grid_dim.x *= blocks_per_cu;
            // Persistent DenseGemm scheduler reads final grid extent from hw_info.cu_count
            args.kernel_params.hw_info.cu_count = args.launch_args.grid_dim.x;

            DgProfParam dg_prof_params;
            if (ProfilingInterface::Instance().get_op_info()) {
                dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, nullptr,
                                          current_stream());
            }
            ProfilingInterface::Instance().instrument(true, dg_prof_params);

            DenseBF16GemmCutlass3Runtime::launch(runtime, args);

            ProfilingInterface::Instance().instrument(false, dg_prof_params);

            char* pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                int numRegs = 0, localSize = 0;
                hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
                hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

                printf("[DenseGemm_BF16_Cutlass3:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
                printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                       args.launch_args.grid_dim.x);
                printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
                       block_m, block_n, block_k, warp_m, warp_n, warp_k, num_stages);
                printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
                printf("waves:%d, overlap_prologue:%d\n", waves, int(overlap_on));
            }
        }
    } else {
        const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
            .mode = cutlass::gemm::GemmUniversalMode::kGemm,
            .problem_shape = {m, n, k, 1},
            .mainloopargs = {input_a, stride_A, input_b, stride_B},
            .epilogueargs =
                {
                    {1, 0},
                    output,
                    stride_D,
                    output,
                    stride_D,
                },
            .hw_info = hw_info,
            .scheduler = {(uint32_t)m, layout_info},
            .signal = nullptr};

        BF16GemmCutlass3Runtime::GemmKernelParams params =
            BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

        auto args = BF16GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                                  num_stages, "DenseGemm", "Default", "bf16_deep_gemm",
                                                                  enable_sbo_overlap},
                                                  .launch_args = {grid, block, SMSIZE},
                                                  .kernel_params = params};

        const auto& code = BF16GemmCutlass3Runtime::generate(args);
        const auto& runtime = compiler->build("bf16_deep_gemm", code);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;
        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, 0, grouped_layout,
                                      current_stream());
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        BF16GemmCutlass3Runtime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);

        char* pEnv_params = std::getenv("show_log");
        if (pEnv_params && isdigit(*pEnv_params)) {
            int numRegs = 0, localSize = 0;
            hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
            hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

            printf("[DenseGemm_BF16:]\n");
            printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
            printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                   args.launch_args.grid_dim.x);
            printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n, block_k,
                   warp_m, warp_n, block_k, num_stages);
            printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
        }
    }
}
} // namespace deep_gemm
