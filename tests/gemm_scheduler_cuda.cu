// Standalone device test: no host shims, PyTorch or DeepGEMM extension needed.
// git submodule update --init third-party/cutlass
// nvcc -std=c++17 -arch=sm_90a -I deep_gemm/include -I third-party/cutlass/include \
//     tests/gemm_scheduler_cuda.cu -o /tmp/gemm_scheduler_cuda
// /tmp/gemm_scheduler_cuda
// compute-sanitizer --tool memcheck --error-exitcode 1 /tmp/gemm_scheduler_cuda
// Use -arch=sm_100a on Blackwell. This tests scheduler arithmetic on device,
// not GEMM math, TMA multicast, pipeline synchronization or performance.
#include <cuda_runtime.h>
#include <deep_gemm/scheduler/gemm.cuh>

#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

using deep_gemm::GemmType;
using deep_gemm::sched::IndexType;

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static void cuda_check(cudaError_t error) {
    require(error == cudaSuccess, cudaGetErrorString(error));
}

struct Tile {
    uint32_t group, m, n, effective_m, valid_rows;
    uint32_t global_m, global_n, global_k, global_sf, local_n, local_k, local_sf;
};

template <GemmType Layout, uint32_t Workers, uint32_t BlockM, bool Padding, bool OnA>
__global__ void collect(int* layout, uint32_t shape_m, uint32_t n,
                        uint32_t capacity, Tile* tiles, int* counts) {
    using Scheduler = deep_gemm::sched::Scheduler<Layout, BlockM, 64, 5, 1, OnA, Workers, Padding>;
    Scheduler scheduler(shape_m, n, 384, layout);
    uint32_t count = 0;
    while (true) {
        uint32_t m = UINT32_MAX, col = UINT32_MAX;
        if (!scheduler.get_next_block(m, col))
            break;
        if (count == capacity) {
            counts[blockIdx.x] = -1;
            return;
        }
        Tile tile{};
        tile.group = scheduler.current_group_idx;
        tile.m = m;
        tile.n = col;
        tile.effective_m = scheduler.get_aligned_effective_m_in_block(m);
        for (uint32_t row = 0; row < BlockM; ++row)
            tile.valid_rows += scheduler.is_computation_valid(m, row);
        tile.global_m = scheduler.template get_global_idx<Layout == GemmType::MGroupedMasked>(shape_m, BlockM, m);
        tile.global_n = scheduler.template get_global_idx<true>(n, 64, col, m);
        tile.local_n = scheduler.template get_global_idx<false>(n, 64, col, m);
        tile.global_k = scheduler.template get_global_idx<true, IndexType::K>(384, 128, 2, m);
        tile.local_k = scheduler.template get_global_idx<false, IndexType::K>(384, 128, 2, m);
        tile.global_sf = scheduler.template get_global_idx<true, IndexType::SF_K>(3, 1, 2, m);
        tile.local_sf = scheduler.template get_global_idx<false, IndexType::SF_K>(3, 1, 2, m);
        tiles[blockIdx.x * capacity + count++] = tile;
    }
    counts[blockIdx.x] = count;
}

template <GemmType Layout, uint32_t Workers, uint32_t BlockM, bool Padding, bool OnA>
void check(const std::array<uint32_t, 5>& rows, uint32_t n) {
    std::array<int, 5> layout{};
    std::array<uint32_t, 5> starts{};
    uint32_t end = 0, shape_m = 0;
    using Key = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::map<Key, uint32_t> expected;
    for (uint32_t group = 0; group < 5; ++group) {
        starts[group] = Layout == GemmType::MGroupedMasked ? 0 : end;
        layout[group] = starts[group] + rows[group];
        for (uint32_t row = 0; row < rows[group]; row += BlockM)
            for (uint32_t col = 0; col < n; col += 64)
                expected[{group, (starts[group] + row) / BlockM, col / 64}] = 0;
        end = layout[group];
        while (end % BlockM) ++end;
        if (rows[group] > shape_m) shape_m = rows[group];
    }
    if constexpr (Layout != GemmType::MGroupedMasked) shape_m = end;
    const uint32_t capacity = expected.size() / Workers + 2;
    int* device_layout;
    int* counts;
    Tile* tiles;
    cuda_check(cudaMallocManaged(&device_layout, sizeof(layout)));
    cuda_check(cudaMallocManaged(&counts, Workers * sizeof(int)));
    cuda_check(cudaMallocManaged(&tiles, Workers * capacity * sizeof(Tile)));
    for (uint32_t i = 0; i < 5; ++i) device_layout[i] = layout[i];
    for (uint32_t i = 0; i < Workers; ++i) counts[i] = -1;
    collect<Layout, Workers, BlockM, Padding, OnA><<<Workers, 1>>>(device_layout, shape_m, n, capacity, tiles, counts);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    for (uint32_t worker = 0; worker < Workers; ++worker) {
        require(counts[worker] >= 0 && counts[worker] <= int(capacity), "device scheduler exceeded capacity");
        for (int i = 0; i < counts[worker]; ++i) {
            const auto& tile = tiles[worker * capacity + i];
            auto found = expected.find({tile.group, tile.m, tile.n});
            require(found != expected.end(), "unexpected device tile");
            require(++found->second == 1, "duplicate device tile");
            uint32_t valid_rows = 0;
            for (uint32_t row = starts[tile.group]; row < starts[tile.group] + rows[tile.group]; ++row)
                valid_rows += row / BlockM == tile.m;
            uint32_t effective_m = BlockM;
            if constexpr (Layout == GemmType::MGroupedContiguousWithPsumLayout && !Padding) {
                effective_m = 16;
                while (effective_m < valid_rows) effective_m += 16;
            }
            require(tile.valid_rows == valid_rows && tile.effective_m == effective_m, "device row extent mismatch");
            // Enumerate group origins from the input tensor extents, independent
            // of scheduler state and its group-boundary search.
            uint32_t m_base = 0, n_base = 0, k_base = 0, sf_base = 0;
            for (uint32_t group = 0; group < tile.group; ++group) {
                if constexpr (Layout == GemmType::MGroupedMasked) m_base += shape_m;
                n_base += n; k_base += 384; sf_base += 3;
            }
            require(tile.global_m == m_base + tile.m * BlockM, "device M coordinate mismatch");
            require(tile.global_n == n_base + tile.n * 64 && tile.local_n == tile.n * 64, "device N coordinate mismatch");
            require(tile.global_k == k_base + 256 && tile.local_k == 256, "device K coordinate mismatch");
            require(tile.global_sf == sf_base + 2 && tile.local_sf == 2, "device scale coordinate mismatch");
        }
    }
    for (const auto& entry : expected) require(entry.second == 1, "missing device tile");
    cuda_check(cudaFree(tiles)); cuda_check(cudaFree(counts)); cuda_check(cudaFree(device_layout));
}

template <uint32_t Workers, bool OnA>
void run() {
    for (auto rows : {std::array<uint32_t, 5>{0, 0, 0, 0, 0},
                      std::array<uint32_t, 5>{0, 1, 15, 16, 17},
                      std::array<uint32_t, 5>{31, 32, 33, 0, 49},
                      std::array<uint32_t, 5>{63, 64, 65, 0, 0},
                      std::array<uint32_t, 5>{0, 127, 128, 129, 0},
                      std::array<uint32_t, 5>{511, 512, 513, 0, 1025}}) {
        for (auto n : {1u, 65u, 513u, 1025u}) {
            check<GemmType::MGroupedMasked, Workers, 64, true, OnA>(rows, n);
            check<GemmType::MGroupedMasked, Workers, 128, false, OnA>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, 64, true, OnA>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, 64, false, OnA>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, 128, false, OnA>(rows, n);
        }
    }
}

int main() {
    run<1, false>(); run<1, true>();
    run<8, false>(); run<8, true>();
    run<132, false>(); run<132, true>();
    std::cout << "720 device scheduler cases passed\n";
}
