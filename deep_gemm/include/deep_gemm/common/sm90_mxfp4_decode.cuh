#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace deep_gemm {

// SM90 MXFP4 software decoder, ported verbatim from PR #411 (`impls/sm90_fp8_mega_moe.cuh`, commit 5d32216).
// Packed E2M1 words carry the SM90 sign order [s0,s4,s1,s5,s2,s6,s3,s7] produced by
// `deep_gemm.mega.mxfp4._reorder_mxfp4_sign_bits_for_sm90`; `exponent_offset` is the processed relative UE8M0 byte of
// the K32 group. The output bytes are E4M3 bit patterns of e2m1 * 2^offset / 64; the x64 is restored together with the
// per-expert secondary scale in the accumulator promotion.
// The model-load preprocessing keeps every magnitude nibble in place but moves
// the eight signs in each packed word to [s0,s4,s1,s5,s2,s6,s3,s7].  The low
// nibble signs of the four bytes therefore belong to outputs 0..3, and the
// high nibble signs belong to outputs 4..7.  This removes the two sign-gather
// PRMTs from every decode without changing its byte size.
CUTLASS_DEVICE uint2 sm90_mxfp4_e4m3_lookup(
    const uint32_t exponent_offset) {
    uint2 result;
    asm volatile(
        "{\n\t"
        "mad.lo.u32 %0, %2, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 %1, %2, 0x08080808, 0x1c181410;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(exponent_offset));
    return result;
}

CUTLASS_DEVICE uint2 sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
    const uint32_t packed, const uint2 lookup) {
    uint2 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 temp0, temp1, out_lo;\n\t"
        "shl.b32 out_lo, %2, 4;\n\t"
        "and.b32 temp0, %2, 0x77777777;\n\t"
        "prmt.b32 temp1, %3, %4, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, %3, %4, temp0;\n\t"
        "lop3.b32 %1, %2, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(packed), "r"(lookup.x), "r"(lookup.y));
    return result;
}

CUTLASS_DEVICE uint2 sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(
    const uint32_t packed, const uint32_t exponent_offset) {
    uint2 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 lookup_lo, lookup_hi, temp0, temp1, out_lo;\n\t"
        "mad.lo.u32 lookup_lo, %3, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 lookup_hi, %3, 0x08080808, 0x1c181410;\n\t"
        "shl.b32 out_lo, %2, 4;\n\t"
        "and.b32 temp0, %2, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %1, %2, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(packed), "r"(exponent_offset));
    return result;
}

CUTLASS_DEVICE uint4 sm90_mxfp4_reordered_signs_e2m1x16_to_e4m3x16_bits(
    const uint32_t packed0,
    const uint32_t packed1,
    const uint32_t exponent_offset) {
    uint4 result;
    asm volatile(
        "{\n\t"
        ".reg .b32 lookup_lo, lookup_hi, temp0, temp1, out_lo;\n\t"
        "mad.lo.u32 lookup_lo, %6, 0x08080800, 0x0c080000;\n\t"
        "mad.lo.u32 lookup_hi, %6, 0x08080808, 0x1c181410;\n\t"
        "shl.b32 out_lo, %4, 4;\n\t"
        "and.b32 temp0, %4, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %1, %4, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %0, out_lo;\n\t"
        "shl.b32 out_lo, %5, 4;\n\t"
        "and.b32 temp0, %5, 0x77777777;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;\n\t"
        "shr.u32 temp0, temp0, 16;\n\t"
        "prmt.b32 temp1, lookup_lo, lookup_hi, temp0;\n\t"
        "lop3.b32 %3, %5, 0x80808080, temp1, 0xea;\n\t"
        "mov.b32 %2, out_lo;\n\t"
        "}"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "r"(packed0), "r"(packed1), "r"(exponent_offset));
    return result;
}

// LDS.32 at a compile-time byte offset from a 32-bit shared address: the offset is the instruction's immediate, so the 16 word and
// 4 scale loads of a register-A stage decode share one address register (a generic pointer per load costs an IADD / IMAD each)
template <uint32_t kOffset>
CUTLASS_DEVICE uint32_t sm90_mxfp4_ld_shared_u32(const uint32_t& smem_addr) {
    uint32_t value;
    asm volatile("ld.shared.u32 %0, [%1+%2];" : "=r"(value) : "r"(smem_addr), "n"(kOffset));
    return value;
}

// UE8M0 byte of K32 group `kGroup` of a scale word, zero-extended: one PRMT for groups 1 / 2 and an FMA-pipe mul.hi for group 3
// in place of SHF + LOP3 on the ALU pipe, which the register-A decode saturates
template <uint32_t kGroup>
CUTLASS_DEVICE uint32_t sm90_mxfp4_scale_byte(const uint32_t& scale_word) {
    uint32_t value;
    if constexpr (kGroup == 0) {
        value = scale_word & 0xffu;
    } else if constexpr (kGroup == 3) {
        asm("mul.hi.u32 %0, %1, 0x100;" : "=r"(value) : "r"(scale_word));
    } else {
        asm("prmt.b32 %0, %1, 0, %2;" : "=r"(value) : "r"(scale_word), "n"(0x4440 + kGroup));
    }
    return value;
}

// Register-A (m64k32 E4M3 A fragment) decode of weight row `row + kRow` of a tiled stage for the 4 K32 groups: word c of the
// unit is lane c's fragment register pair (frag[g][kS], frag[g][2 + kS]). `word_addr` / `scale_addr` are the lane's 32-bit shared
// addresses of its K32-group-0 word and its scale word for kRow = 0; every other term is an LDS immediate. Free function templates
// (force-inlined) rather than device lambdas: ptxas left template lambdas of this as ABI calls in the N24 kernel (STACK 1744)
template <uint32_t kGroupBytes, uint32_t kRow, uint32_t kS, uint32_t kG>
CUTLASS_DEVICE void sm90_mxfp4_decode_rs_group(const uint32_t& word_addr, const uint32_t& scale_word, uint32_t (&frag)[4][4]) {
    const uint32_t word = sm90_mxfp4_ld_shared_u32<kG * kGroupBytes + kRow * 16>(word_addr);
    const uint2 d = sm90_mxfp4_reordered_signs_e2m1x8_to_e4m3x8_bits(word, sm90_mxfp4_e4m3_lookup(sm90_mxfp4_scale_byte<kG>(scale_word)));
    frag[kG][kS] = d.x;
    frag[kG][2 + kS] = d.y;
}

template <uint32_t kGroupBytes, uint32_t kRow, uint32_t kS>
CUTLASS_DEVICE void sm90_mxfp4_decode_rs_row(const uint32_t& word_addr, const uint32_t& scale_addr, uint32_t (&frag)[4][4]) {
    const uint32_t scale_word = sm90_mxfp4_ld_shared_u32<kRow * 4>(scale_addr);
    sm90_mxfp4_decode_rs_group<kGroupBytes, kRow, kS, 0>(word_addr, scale_word, frag);
    sm90_mxfp4_decode_rs_group<kGroupBytes, kRow, kS, 1>(word_addr, scale_word, frag);
    sm90_mxfp4_decode_rs_group<kGroupBytes, kRow, kS, 2>(word_addr, scale_word, frag);
    sm90_mxfp4_decode_rs_group<kGroupBytes, kRow, kS, 3>(word_addr, scale_word, frag);
}

} // namespace deep_gemm
