// Run through test_gemm_scheduler.py, which supplies host dependency shims.
#include <deep_gemm/scheduler/gemm.cuh>

#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

using deep_gemm::GemmType;
using Tile = std::tuple<uint32_t, uint32_t, uint32_t>;
static size_t cases = 0;

// Model a concatenated tensor axis by enumerating its element coordinates.
// The oracle never reads current_group_idx, current_psum_m or num_m_blocks.
static std::array<std::vector<uint32_t>, 5> axis_coordinates(uint32_t extent) {
    std::array<std::vector<uint32_t>, 5> coordinates;
    uint32_t next = 0;
    for (auto& group : coordinates)
        for (uint32_t element = 0; element < extent; ++ element)
            group.push_back(next++);
    return coordinates;
}

static void require(bool condition, const char* message) {
    if (not condition) {
        std::cerr << "Case " << cases << ": " << message << '\n';
        std::exit(1);
    }
}

template <GemmType Layout, uint32_t Workers, bool OnA, uint32_t Multicast = 1,
          uint32_t BlockM = 64, uint32_t BlockN = 64, bool EnsureZeroPadding = true>
void check(const std::array<uint32_t, 5>& rows, uint32_t n) {
    using Scheduler = deep_gemm::sched::Scheduler<
        Layout, BlockM, BlockN, 5, Multicast, OnA, Workers, EnsureZeroPadding>;
    ++ cases;
    std::array<int, 5> layout{};
    std::array<uint32_t, 5> starts{};
    uint32_t end = 0;
    // Independent oracle: enumerate physical row/column tile origins. It does
    // not reproduce swizzling, persistent strides, or scheduler group search.
    std::map<Tile, uint32_t> expected;
    for (uint32_t group = 0; group < rows.size(); ++ group) {
        const uint32_t start = Layout == GemmType::MGroupedMasked ? 0 : end;
        starts[group] = start;
        layout[group] = Layout == GemmType::MGroupedMasked ? rows[group] : start + rows[group];
        for (uint32_t m = 0; m < rows[group]; m += BlockM)
            for (uint32_t col = 0; col < n; col += BlockN)
                expected[{group, (start + m) / BlockM, col / BlockN}] = 0;
        end = start + rows[group];
        while (end % BlockM != 0)
            ++ end;
    }

    const auto shape_m = Layout == GemmType::MGroupedMasked ?
        *std::max_element(rows.begin(), rows.end()) : end;
    const auto m_coordinates = axis_coordinates(shape_m);
    const auto n_coordinates = axis_coordinates(n);
    const auto k_coordinates = axis_coordinates(384);
    const auto sf_coordinates = axis_coordinates(3);
    size_t visited = 0;
    for (uint32_t worker = 0; worker < Workers; ++ worker) {
        blockIdx.x = worker;
        Scheduler scheduler(shape_m, n, 384, layout.data());
        size_t iterations = 0;
        while (true) {
            // Reset on every call: a missing output assignment must not reuse
            // a previous valid tile (or read an uninitialized host variable).
            uint32_t m = UINT32_MAX, col = UINT32_MAX;
            if (not scheduler.get_next_block(m, col))
                break;
            require(++ iterations <= expected.size(), "scheduler did not terminate");
            const auto group = scheduler.current_group_idx;
            auto found = expected.find({group, m, col});
            require(found != expected.end(), "out-of-range or empty-group tile");
            require(++ found->second == 1, "tile has multiple owners");
            ++ visited;
            using deep_gemm::sched::IndexType;
            // Masked A/D have a group axis; psum A/D already use physical M.
            constexpr bool m_group_offset = Layout == GemmType::MGroupedMasked;
            require(scheduler.template get_global_idx<m_group_offset>(shape_m, BlockM, m) ==
                    m_coordinates[m_group_offset ? group : 0].at(m * BlockM), "incorrect M coordinate");
            require(scheduler.template get_global_idx<false>(n, BlockN, col, m) ==
                    n_coordinates[0].at(col * BlockN), "incorrect local N coordinate");
            require(scheduler.template get_global_idx<true>(n, BlockN, col, m) ==
                    n_coordinates[group].at(col * BlockN), "incorrect grouped N coordinate");
            for (uint32_t k = 0; k < 3; ++ k) {
                require(scheduler.template get_global_idx<false, IndexType::K>(384, 128, k, m) ==
                        k_coordinates[0].at(k * 128), "incorrect local K coordinate");
                require(scheduler.template get_global_idx<true, IndexType::K>(384, 128, k, m) ==
                        k_coordinates[group].at(k * 128), "incorrect grouped K coordinate");
                require(scheduler.template get_global_idx<false, IndexType::SF_K>(3, 1, k, m) ==
                        sf_coordinates[0].at(k), "incorrect local scale coordinate");
                require(scheduler.template get_global_idx<true, IndexType::SF_K>(3, 1, k, m) ==
                        sf_coordinates[group].at(k), "incorrect grouped scale coordinate");
            }
            // Check every row of a tail, including the first invalid row.
            uint32_t valid_rows = 0;
            for (uint32_t offset = 0; offset < BlockM; ++ offset) {
                const auto local_row = m * BlockM + offset - starts[group];
                valid_rows += local_row < rows[group];
                require(scheduler.is_computation_valid(m, offset) == (local_row < rows[group]),
                        "incorrect row validity at tile tail");
            }
            uint32_t effective_m = BlockM;
            if constexpr (Layout == GemmType::MGroupedContiguousWithPsumLayout and not EnsureZeroPadding) {
                // Smallest legal UMMA extent covering the independently counted
                // valid rows, without scheduler tail detection or math::align.
                effective_m = 16;
                while (effective_m < valid_rows)
                    effective_m += 16;
            }
            require(scheduler.get_aligned_effective_m_in_block(m) == effective_m,
                    "incorrect effective M extent");
        }
    }
    require(visited == expected.size(), "missing tiles");
    for (const auto& entry : expected)
        require(entry.second == 1, "tile was not assigned exactly once");
}

template <uint32_t Workers, bool OnA>
void run() {
    const std::vector<std::array<uint32_t, 5>> examples = {
        {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 0, 0, 0}, {0, 0, 0, 0, 1},
        {0, 63, 64, 65, 0}, {65, 0, 1, 0, 129}, {511, 512, 513, 1023, 1025},
        {1, 15, 16, 17, 31}, {32, 33, 47, 48, 49}, {127, 128, 129, 0, 17},
    };
    // N tails plus both sides of the 8/16-block swizzle group boundaries.
    for (const auto& rows : examples) {
        for (auto n : {1u, 63u, 64u, 65u, 448u, 512u, 513u, 960u, 1024u, 1025u}) {
            check<GemmType::MGroupedMasked, Workers, OnA>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA>(rows, n);
            check<GemmType::MGroupedMasked, Workers, OnA, 1, 128, 64>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA, 1, 64, 128>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA, 1, 64, 64, false>(rows, n);
            check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA, 1, 128, 64, false>(rows, n);
            check<GemmType::MGroupedMasked, Workers, OnA, 1, 64, 64, false>(rows, n);
        }
    }
    // Exact wave boundaries: total tiles = Workers - 1, Workers, Workers + 1.
    for (auto tiles : {Workers - 1, Workers, Workers + 1}) {
        std::array<uint32_t, 5> rows = {0, 64 * tiles, 0, 0, 0};
        check<GemmType::MGroupedMasked, Workers, OnA>(rows, 64);
        check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA>(rows, 64);
    }
    std::mt19937 random(20261102);
    for (uint32_t trial = 0; trial < 100; ++ trial) {
        std::array<uint32_t, 5> rows{};
        for (auto& count : rows)
            count = random() % 4 == 0 ? 0 : random() % 1153;
        const uint32_t n = 1 + random() % 1153;
        check<GemmType::MGroupedMasked, Workers, OnA>(rows, n);
        check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA>(rows, n);
    }
    if constexpr (Workers % 2 == 0) {
        // Aligned multicast cases only: SM100 cannot disable 2-CTA multicast.
        // This checks coverage only, not peer liveness or multicast safety.
        for (auto rows : {std::array<uint32_t, 5>{0, 128, 256, 0, 1152},
                          std::array<uint32_t, 5>{128, 0, 0, 128, 0}}) {
            for (auto n : {128u, 896u, 1024u, 1152u}) {
                check<GemmType::MGroupedMasked, Workers, OnA, 2>(rows, n);
                check<GemmType::MGroupedContiguousWithPsumLayout, Workers, OnA, 2>(rows, n);
            }
        }
#if __CUDA_ARCH__ < 1000
        // Exercise SM90's odd primary swizzle tail with an even secondary
        // dimension, so each group still contains whole pairs of CTAs.
        for (auto primary : {1u, 7u, 9u, 15u, 17u}) {
            const auto m = OnA ? 128u : primary * 64;
            const auto n = OnA ? primary * 64 : 128u;
            check<GemmType::MGroupedMasked, Workers, OnA, 2>({0, m, 0, m, 0}, n);
        }
#endif
    }
}

template <uint32_t Workers>
void run_workers() {
    run<Workers, false>();
    run<Workers, true>();
}

int main() {
    run_workers<1>();
    run_workers<2>();
    run_workers<7>();
    run_workers<8>();
    run_workers<31>();
    run_workers<32>();
    run_workers<132>();
    run_workers<148>();
    std::cout << cases << " grouped scheduler coverage cases passed (host, arch branch "
              << __CUDA_ARCH__ << ")\n";
}
