#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd>
__device__ __host__ void outer_launch_k_iterations(const auto& inner_launch_k_iterations, const auto& func, uint32_t num_former_iters) {
    if (num_former_iters == kNumFormerIters) {
        inner_launch_k_iterations(func, cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        outer_launch_k_iterations<kNumFormerIters + kGap, kGap, kEnd>(inner_launch_k_iterations, func, num_former_iters);
}
// BLOCK_M, BLOCK_N, BLOCK_K，编译阶段的优化参数，表示不同维度切分块的大小
// A(M×K) × B(K×N) = D(M×N)
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumLastStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type; // 根据BLOCK_N（N 维度的分块大小）选择适配的FP8 精度 warp 级 MMA 指令类型， WGMMA（可理解为 “Warp-Level MMA”）是选中的指令类型的别名，该类型中包含硬件指令支持的固定维度信息（如WGMMA::M、WGMMA::N、WGMMA::K，分别对应硬件一次能处理的 M、N、K 维度大小）
    using Barrier = cutlass::arch::ClusterTransactionBarrier; // 支持多个 CTA（线程块）组成的集群之间的同步
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size"); // 在编译期检查BLOCK_M（M 维度的分块大小）是否为硬件 MMA 指令支持的WGMMA::M的整数倍，确保分块大小与硬件能力兼容

    // Overwrite shape constants if the compiler gives，将调用写入的shape参数覆盖写入编译阶段的优化参数
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    // 假设原始矩阵为A_fp32（M×K）、B_fp32（K×N），计算C = A_fp32 × B_fp32
    // A_fp8[m][k] = A_fp32[m][k] ÷ SFA[m]， B_fp8[k][n] = B_fp32[k][n] ÷ SFB[n]，  C_bf8[m][n] = sum_k (A_fp8[m][k] × B_fp8[k][n])，  C_fp32[m][n] = C_fp8[m][n] × SFA[m] × SFB[n]， 把C存储到D中。
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0); // 判断矩阵B的缩放因子是否可以 “统一使用”（即单个缩放因子覆盖整个B块）
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16); // 存储矩阵D（计算结果）的共享内存大小
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);// 每个流水线阶段（stage）中，矩阵A的共享内存大小。A的块大小是BLOCK_M × BLOCK_K（M×K维度），元素类型是__nv_fp8_e4m3（8 位浮点数），用于存储单次加载的A块数据
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);// 每个流水线阶段中，矩阵B的共享内存大小。B的块大小是BLOCK_N × BLOCK_K（N×K维度），元素类型同样是 8 位浮点数，存储单次加载的B块数据
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);// 每个阶段中，矩阵A的缩放因子（SFA）的共享内存大小。SFA与A的M维度对应（每个M元素一个缩放因子），类型是float（32 位），因此大小为BLOCK_M × 4字节
    const uint32_t& shape_k_scales = ceil_div(shape_k, BLOCK_K);// 计算K维度上缩放因子的总数量
    const uint32_t& smem_sfb_size = align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)); // 计算共享内存中存储B的缩放因子（SFB）所需的空间，并确保内存对齐

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K; // 计算所有存储阶段（stage）能覆盖的 K 维度总长度。 kNumStages是流水线阶段数量，最小两阶段，计算+数据加载。
    const uint32_t num_iterations = ceil_div(shape_k, kFullKOfAllStages); // 计算处理整个 K 维度需要的总迭代次数。
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0); // 获取当前线程所在的warp（线程束）索引
    const uint32_t lane_idx = get_lane_idx(); // 获取当前线程在warp 内的车道索引（lane index）（lane 0 到 lane 31）

    // Prefetch TMA descriptors at the very beginning
    if (threadIdx.x == kNumMathThreads) { // 指定一个专门的辅助线程（而非负责核心计算的线程）执行预取操作
        // NOTES: `reinterpret_cast` must be here, or NVRTC will fail
        // TmaDescriptor包含张量的内存地址、形状、步长、数据类型等信息，是 TMA 操作的 “说明书”——TMA 硬件需要根据描述符才能正确传输数据
        // cute::prefetch_tma_descriptor函数：将 TMA 描述符从全局内存预加载到更靠近 SM（流多处理器）的缓存（如 L2 缓存或 SM 的本地缓存）中，加快后续访问速度
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp(); // 确保当前 warp 内的所有线程（包括执行预取的线程和其他线程）都完成当前步骤后再继续执行

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[]; // 声明共享内存（SMEM）变量，并指定其对齐方式
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer); // // 输出矩阵D的存储区域（bfloat16类型，用于暂存计算结果）
    __nv_fp8_e4m3* smem_a[kNumStages]; // 输入矩阵A的存储区域（FP8类型，多阶段流水线用，kNumStages是流水线阶段数）
    __nv_fp8_e4m3* smem_b[kNumStages]; // 输入矩阵B的存储区域（FP8类型，多阶段流水线用，kNumStages是流水线阶段数）
    float* smem_sfa[kNumStages]; // 矩阵A的缩放因子存储区域（float类型，FP8计算需要缩放以保证精度）
    float* smem_sfb; // 矩阵B的缩放因子存储区域（float类型）

    // TMA Barrier for both divisible and non-divisible cases // TMA数据传输的同步屏障（用于协调数据加载/计算的时序）
    Barrier* full_barriers[kNumStages]; // 表示 “共享内存区域已填满数据” 的屏障，通知计算线程可以读取该区域数据
    Barrier* empty_barriers[kNumStages]; // 表示 “共享内存区域已被消费（数据已计算完毕）” 的屏障，通知 TMA 线程可以重新填充新数据

    // Fill shared memory pointers 划分smem_buffer作为多个子区域
    #pragma unroll // #pragma unroll指令会强制编译器展开循环
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_sfa[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SFA_SIZE_PER_STAGE);
    }
    smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE));

    // Fill barriers 为每个stage分配一个full_barrier和empty_barrier
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast"); // 通过 TMA 同时向多个 CTA 传输数据的数量不能超过 32
    if (threadIdx.x == kNumMathThreads) { // 特定线程负责屏障初始化
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1); // 参数1表示该屏障需要等待1 个实体（通常是 TMA 传输完成的信号）到达后才会触发（通知计算线程可以读取数据）
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32); // kNumMathThreads 是计算线程的总数，除以 32（每个 warp 的线程数）得到计算线程的 warp 总数；乘以 kNumTMAMulticast（TMA 多播数量），得到需要同步的总 warp 数
        }

        // Make initialized barrier visible in async proxy  内存屏障（fence），用于确保屏障的初始化状态被所有线程（包括其他 CTA 的线程）可见
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling 通过DivisibleK/NotDivisibleK告诉func当前 K 迭代是否可分（影响数据加载范围）；通过SkipComputation/NotSkipComputation告诉func是否需要执行计算（如某些场景下只需加载数据无需计算）
    struct DivisibleK {};         // 标记K维度可被流水线总步长整除的情况， 这些是空结构体，仅作为编译期标签使用
    struct NotDivisibleK {};      // 标记K维度不可被流水线总步长整除的情况， 这些是空结构体，仅作为编译期标签使用
    struct SkipComputation {};    // 标记需要跳过计算的情况， 这些是空结构体，仅作为编译期标签使用
    struct NotSkipComputation {}; // 标记需要执行计算的情况， 这些是空结构体，仅作为编译期标签使用
    auto launch_k_iterations = [=](const auto& func, bool skip_computation, uint32_t num_former_iters) {
        constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
        constexpr uint32_t kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
        constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

        // NOTES: for too-many branches (> 5), we disable this optimization
        // Otherwise, the compiler must know the dynamic variable `num_former_iters`'s real value
        outer_launch_k_iterations<0, kGap, kEnd>([=](const auto& func, auto num_former_iters_type) {
            if (skip_computation) {
                // 跳过计算的场景：对所有K迭代，传递DivisibleK和SkipComputation标签
                for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter)
                    func(k_iter, DivisibleK{}, SkipComputation{}, num_former_iters_type);
            } else if (shape_k % kFullKOfAllStages == 0) {
                // K维度可分的场景：对所有K迭代，传递DivisibleK和NotSkipComputation标签
                for (uint32_t k_iter = 0; k_iter < num_iterations; ++ k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            } else {
                // K维度不可分的场景：前n-1次迭代用DivisibleK，最后一次用NotDivisibleK
                for (uint32_t k_iter = 0; k_iter < num_iterations - 1; ++ k_iter)
                    func(k_iter, DivisibleK{}, NotSkipComputation{}, num_former_iters_type);
                func(num_iterations - 1, NotDivisibleK{}, NotSkipComputation{}, num_former_iters_type);
            }
        }, func, kShouldOptimize ? num_former_iters : 0);
    };

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 40; // TMA线程使用的寄存器数量
    constexpr uint32_t kNumMathRegisters = 232; // 计算线程使用的寄存器数量

    // Block scheduler
    uint32_t m_block_idx, n_block_idx; // 用于存储当前线程块（CTA）被分配到的矩阵块在 M 维度（行）和 N 维度（列）上的索引（后续会由调度器赋值）
    // 管理线程块（CTA）与矩阵块的映射关系，即决定每个 CTA 负责计算矩阵D中哪一块子矩阵（D = A × B）。其模板参数和构造参数的含义如下：
    // 模板参数：配置调度规则的静态信息，包括：
    // kGemmType：GEMM 的类型（如普通矩阵乘、分组矩阵乘等）；
    // BLOCK_M/BLOCK_N：每个 CTA 负责的子矩阵在 M/N 维度上的大小；
    // kNumGroups：分组数量（用于分组 GEMM 场景）；
    // kNumTMAMulticast/kIsTMAMulticastOnA：TMA 多播的配置（是否在 A 矩阵上启用多播、多播数量）；
    // kNumSMs：GPU 的 SM 数量（用于均衡负载）。
    // 构造参数：动态输入信息，包括：
    // shape_m/shape_n：输入矩阵A/B的整体大小（M/N 维度）；
    // grouped_layout：分组布局信息（用于分组 GEMM，指定每组的矩阵大小）。
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, grouped_layout);

    if (threadIdx.x >= kNumMathThreads) { // 当前属于 TMA 线程
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>(); // 为 TMA 线程组（warp-group）分配预定义的寄存器资源

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x < kNumMathThreads + 32 and cute::elect_one_sync()) { //  限制 TMA 控制逻辑仅在一个 warp（32 线程）内执行（TMA 操作通常由单个 warp 协调即可）， cute::elect_one_sync() 是同步选举函数，确保最终只有一个线程（或 warp） 执行后续的 TMA 加载逻辑，避免多线程重复发起 TMA 传输
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) { // 调度器获取下一个需要处理的矩阵块索引（m_block_idx 对应 M 维度，n_block_idx 对应 N 维度）
                launch_k_iterations([&](uint32_t k_iter, auto divisible_type, auto _, auto __) {
                    // 根据 K 维度是否可分（DivisibleK 标签），确定当前迭代需要处理的流水线阶段数（kNumInnerStages）
                    constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(divisible_type), DivisibleK>;
                    constexpr uint32_t kNumInnerStages = kHasDivisibleStages ? kNumStages : kNumLastStages;

                    // Assign TMA multicast number into A and B
                    // NOTES: there may be additional odd rows/columns or cases where multicast is not possible. 支持多播时用kNumTMAMulticast，否则为 1（单播）
                    const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                    const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                    // NOTES: unrolling and `kNumInnerStages` are vital for performance, NVCC will try to eliminate all
                    // shared memory pointers, e.g. `full_barriers` registers, if all the access indices are constant
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1); // 等待共享内存区域空闲（计算线程已消费完数据）

                        // Issue TMA A
                        constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked; // 对于分组 GEMM，不同组的 A 矩阵在全局内存中的存储位置不连续，需要通过 “组偏移量” 定位到当前组的起始地址
                        auto& full_barrier = *full_barriers[s];
                        uint32_t k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K; // 计算当前需要加载的 K 维度全局起始索引，精准定位 TMA 要加载的 A/B 矩阵 “K 切片”
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_a[s], k_idx, scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx), // k_idx 和scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx) 是2D坐标
                                 num_tma_multicast_a); // 加载A矩阵到共享内存smem_a[s] 形状为 BLOCK_M × BLOCK_K 
                        tma_copy(&tensor_map_sfa, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_sfa[s], m_block_idx * BLOCK_M,
                                 scheduler.get_global_idx<kWithGroupOffsetA>(shape_k_scales, 1, k_idx / BLOCK_K),
                                 num_tma_multicast_a); // 加载A的缩放因子sfa到smem_sfa[s]

                        // Issue TMA B
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                                 num_tma_multicast_b); // 加载B矩阵到共享内存smem_b[s]
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SFA_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases  对于超出kNumInnerStages的阶段（通常是 K 维度末尾的未对齐部分），无需实际加载数据，只需通过屏障同步标记 “完成”，确保流水线节奏一致
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                }, false, 0);
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * num_iterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>(); // 为每个计算线程组分配 232 个寄存器。

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0); // 计算当前线程所属的 “计算线程组索引”，并通过线程束洗牌（shuffle）操作让组内所有线程共享该索引
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8; // 预计算共享内存访问的偏移量，用于高效加载 A 矩阵的缩放因子（smem_sfa）

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0); // 在设备端（GPU）断言 N 维度的总大小（shape_n）必须是 8 的倍数
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters; // BLOCK_N是每个线程块（CTA）负责的 N 维度大小（例如 256）， 除以 8 的原因：后续处理 B 矩阵时，通常以 8 个元素为一组（如 WGMMA 指令一次处理 8 列）
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2); // 计算需要加载的 B 矩阵缩放因子（sfb）的总数量

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks 让计算线程（除第一个 warp 外）并行加载 B 矩阵的缩放因子（sfb）到共享内存
            if (threadIdx.x >= 32) {
                // 计算sfb在全局内存中的起始地址
                auto num_previous_lines = scheduler.get_global_idx<true>(ceil_div(shape_n, BLOCK_K), 0, 0, m_block_idx);
                auto local_sfb = sfb + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * shape_k_scales;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32) // 让符合条件的线程（threadIdx.x >= 32）按 “跨步访问” 的方式并行加载数据
                    st_shared(smem_sfb + i, __ldg(local_sfb + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync(); // 通过命名屏障（NamedBarrier）同步所有计算线程（共kNumMathThreads个），确保所有sfb数据都已加载到共享内存后，再执行后续的矩阵乘法计算。

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = WGMMA::M * (BLOCK_M <= 64 ? 1 : 2); // 将BLOCK_M划分为更小的 “计算子块”（wave），每个子块由一组线程负责，适配 WGMMA 指令的处理粒度，平衡线程负载
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes"); // 保证BLOCK_M能被均匀划分为多个WAVE_BLOCK_M大小的子块，避免计算时出现未对齐的边界问题，确保每个子块的计算逻辑一致
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0}; // accum[WGMMA::kNumAccum]：WGMMA 指令的中间累加器，用于暂存当前子块的乘加结果；final_accum[...]：最终累加结果数组，大小为 “每个子块的累加器数量 × 子块总数”（WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)）

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](uint32_t s) { // 定义 发送 “数据已消费” 信号
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](uint32_t k_iter, auto divisible_type, auto skip_type, auto _) {
                constexpr bool kSkipComputation = cute::is_same_v<decltype(skip_type), SkipComputation>;
                constexpr bool kHasDivisibleStages = cute::is_same_v<decltype(divisible_type), DivisibleK>;
                constexpr uint32_t kNumInnerStages = kSkipComputation ? 0 : (kHasDivisibleStages ? kNumStages : kNumLastStages); // 如果跳过计算，则不进行任何计算，否则根据是否有可整除的阶段（kHasDivisibleStages）来决定是否使用 kNumStages 或 kNumLastStages

                #pragma unroll
                for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                    // Read B scales
                    float scale_b_0 = ld_shared(smem_sfb + k_iter * kNumStages + s), scale_b_1; //  读取 B 矩阵的缩放因子（scale_b_0、scale_b_1）
                    // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_b_1 = ld_shared(smem_sfb + k_iter * kNumStages + s + shape_k_scales);

                    // Wait TMA arrivals 由 TMA 线程在加载完smem_a[s]、smem_b[s]后触发
                    full_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter) & 1);

                    // TODO: remove some useless computation for unaligned Ms
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) { // 将大的 M 维度块分解为 WGMMA 指令可高效处理的子块，平衡线程负载，提升并行效率
                      	auto m_offset = local_idx * WAVE_BLOCK_M;

                    	// Read A scales 读取 A 矩阵的缩放因子（scale_a_0、scale_a_1）
                    	// NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    	auto scale_a_0 = ld_shared(smem_sfa[s] + r_0 + m_offset);
                        auto scale_a_1 = ld_shared(smem_sfa[s] + r_1 + m_offset);

                    	// Commit WGMMA instructions
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]); // 确保累加器（accum）的寄存器操作顺序，避免乱序执行导致的数据错误。
                    	warpgroup_arrive(); // 线程组（warpgroup）内的线程同步，确保所有线程准备好执行矩阵乘法
                    	#pragma unroll
                    	for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) { // 将 K 维度的子块（BLOCK_K）拆分为 WGMMA 指令可处理的粒度（WGMMA::K)
                            auto desc_a = make_smem_desc(smem_a[s] + (math_wg_idx * WGMMA::M + m_offset) * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                    	}
                    	warpgroup_commit_batch(); // 提交计算任务
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]); // // 确保累加器（accum）的寄存器操作顺序，避免乱序执行导致的数据错误。
                    	warpgroup_wait<0>(); // 等待，确保所有线程的 WGMMA 指令执行完毕

                    	// Notify barrier arrival at the last warpgroup wave
                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                    	    empty_barrier_arrive(s); // 告知 TMA 线程 “当前阶段的共享内存数据已处理完毕，可安全加载新数据”

                    	// Promote with scales 将accum中的中间结果（WGMMA 计算结果）按缩放因子缩放后，累加到final_accum中（final_accum是存储最终结果的累加器数组，按 M 子块划分）
                    	// NOTES: making it as predicates is very important for performance, comparing to two loops
                    	float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                    	float scale_0_1, scale_1_1;
                    	if constexpr (not kMustUseUniformedScaleB)
                            scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    	#pragma unroll
                    	for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                            bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                            shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                            shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                            shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                            shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    	}
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * num_iterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            }, not scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M), num_former_iters);

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16); // 目标数据类型（nv_bfloat16）的字节大小
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes); // TMA 存储时 N 维度的块大小（每次 TMA 操作处理的列数）
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4; // 每个 warp 负责的 M 维度子块大小
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) 
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync(); // 确保所有计算线程（共kNumMathThreads个）都完成前序计算，统一进入结果写入阶段

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) { // 按WAVE_BLOCK_M（M 维度的子块粒度，如 32）划分BLOCK_M，每个子块对应final_accum中的一段结果（shifted_accum）
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) { // 按 4 个累加器为一组处理（WGMMA::kNumAccum / 4），匹配向量指令的处理粒度（一次处理 4 个元素）
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy( // 用 SM90 架构的专用向量存储指令（STSM），将转换后的 bfloat16 向量写入smem_ptr指向的共享内存位置，一次操作完成 4 个元素的存储
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}), //将 float 类型的累加结果（shifted_accum中的 4 个元素）转换为 bfloat16 类型（2 字节），并打包为向量（每次转换 2 对元素，共 4 个元素）
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence(); // TMA 存储的内存屏障，确保所有 TMA 存储操作完成
            cutlass::arch::NamedBarrier(kNumMathThreads).sync(); // 确保所有计算线程（共kNumMathThreads个）都完成结果写入，统一进入 TMA 存储阶段

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumMathThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) { // 仅threadIdx.x < BLOCK_N / TMA_D_BLOCK_N的线程参与 TMA 存储
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N; // 当前线程负责的 N 维度在本地块内的偏移
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M; // 共享内存中待存储数据的起始地址
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                              n_block_idx * BLOCK_N + in_block_n_offset,
                                              scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx)); // 使用 TMA 存储指令（TMA_STORE_2D），将数据从共享内存中存储到全局内存中
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
