#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

#include <deep_gemm/common/exception.cuh>

// Probes compile to nothing unless the JIT defines `DG_PROBE_ENABLE 1`
#ifndef DG_PROBE_ENABLE
#define DG_PROBE_ENABLE 0
#endif

namespace deep_gemm::probe {

// Per-warp `%clock64` trace: a fixed-capacity stream of 8-byte records in global memory, identified by its index,
// framed by `mark()` at both ends for clock calibration; the buffer must be zeroed and stay inside one 4 GiB window
template <uint32_t kNumRecords>
struct Stream {
    static constexpr bool kEnabled = DG_PROBE_ENABLE and kNumRecords > 0;
    static constexpr uint32_t kNumBytesPerRecord = 8;
    static constexpr uint32_t kNumBytes = kNumRecords * kNumBytesPerRecord;

    CUTLASS_HOST_DEVICE static constexpr uint64_t get_num_bytes(const uint32_t& num_streams) {
        return static_cast<uint64_t>(num_streams) * kNumBytes;
    }

    CUTLASS_HOST_DEVICE static bool is_valid_buffer(const void* buffer, const uint32_t& num_streams) {
        const auto base = reinterpret_cast<uint64_t>(buffer);
        return kNumRecords % 2 == 0 and base % 16 == 0 and
               (base >> 32) == ((base + get_num_bytes(num_streams) - 1) >> 32);
    }

    uint64_t cursor = 0;

    CUTLASS_DEVICE Stream(void* buffer, const uint32_t& stream_idx) {
        if constexpr (kEnabled)
            cursor = reinterpret_cast<uint64_t>(buffer) + static_cast<uint64_t>(stream_idx) * kNumBytes;
    }

    // NOTES: advance the cursor first and record with negative offsets, so that ptxas keeps the add in place
    template <int32_t kRecordOffset = 0>
    CUTLASS_DEVICE void record() const {
        if constexpr (kEnabled) {
            asm volatile("{\n\t"
                         ".reg .b64 t;\n\t"
                         "mov.u64 t, %%clock64;\n\t"
                         "st.global.b64 [%0+%1], t;\n\t"
                         "}" :: "l"(cursor), "n"(kRecordOffset * kNumBytesPerRecord) : "memory");
        }
    }

    // Two records of payload (e.g. a task descriptor) for attributing the surrounding records
    template <int32_t kRecordOffset>
    CUTLASS_DEVICE void record_payload(const uint32_t& a, const uint32_t& b,
                                       const uint32_t& c, const uint32_t& d) const {
        DG_STATIC_ASSERT(kRecordOffset % 2 == 0, "Payloads are 16-byte aligned");
        if constexpr (kEnabled) {
            asm volatile("st.global.v4.b32 [%0+%1], {%2, %3, %4, %5};"
                         :: "l"(cursor), "n"(kRecordOffset * kNumBytesPerRecord), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
        }
    }

    // `{clock64, globaltimer}` for calibrating the SM clock, as two 8-byte stores (the cursor may not be 16-byte aligned)
    CUTLASS_DEVICE void mark() {
        if constexpr (kEnabled) {
            asm volatile("{\n\t"
                         ".reg .b64 t, g;\n\t"
                         "mov.u64 t, %%clock64;\n\t"
                         "mov.u64 g, %%globaltimer;\n\t"
                         "st.global.b64 [%0], t;\n\t"
                         "st.global.b64 [%0+8], g;\n\t"
                         "}" :: "l"(cursor) : "memory");
            advance<2>();
        }
    }

    // Only the low 32 bits are advanced (a single `IADD3`)
    template <uint32_t kNumRecordsToAdvance>
    CUTLASS_DEVICE void advance() {
        if constexpr (kEnabled) {
            asm volatile("{\n\t"
                         ".reg .b32 lo, hi;\n\t"
                         "mov.b64 {lo, hi}, %0;\n\t"
                         "add.u32 lo, lo, %1;\n\t"
                         "mov.b64 %0, {lo, hi};\n\t"
                         "}" : "+l"(cursor) : "n"(kNumRecordsToAdvance * kNumBytesPerRecord));
        }
    }
};

} // namespace deep_gemm::probe
