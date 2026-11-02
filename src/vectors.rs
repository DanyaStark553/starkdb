use crate::format::{Sq8Meta, FLAG_DELETED};
use crate::graph::StarkGraph;
use crate::storage::StarkStorage;
use anyhow::{bail, Result};
use half::f16;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

pub const HNSW_M: usize = 16;
pub const HNSW_M_MAX0: usize = 32;
pub const HNSW_EF_CONSTRUCTION: usize = 64;
pub const HNSW_EF_SEARCH: usize = 32;
pub const HNSW_AUTO_THRESHOLD: usize = 256;

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

#[inline]
pub fn cosine_similarity_f16(query_f32: &[f32], stored_f16: &[f16]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y_f16) in query_f32.iter().zip(stored_f16.iter()) {
        let y = y_f16.to_f32();
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = (norm_a * norm_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// Symmetric per-vector SQ8 quantization mapping `src` into `[-127, 127]`.
/// Never uses `-128` so `_mm256_abs_epi8` and `_mm256_sign_epi8` never overflow.
#[inline]
pub fn quantize_sq8(src: &[f32], dst: &mut [i8]) -> Sq8Meta {
    let n = src.len().min(dst.len());
    let mut max_abs = 0.0f32;
    for &x in &src[..n] {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs <= 1e-12 {
        dst[..n].fill(0);
        return Sq8Meta {
            scale: 0.0,
            sum_q: 0,
        };
    }
    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;
    let mut sum_q: i32 = 0;
    for i in 0..n {
        let scaled = src[i] * inv_scale;
        let rounded = (scaled + if scaled >= 0.0 { 0.5 } else { -0.5 }) as i32;
        let q = rounded.clamp(-127, 127) as i8;
        dst[i] = q;
        sum_q += q as i32;
    }
    Sq8Meta { scale, sum_q }
}

#[allow(dead_code)]
type DotSq8Fn = fn(&[i8], &[i8]) -> i32;

#[allow(dead_code)]
static DOT_SQ8_FN: OnceLock<DotSq8Fn> = OnceLock::new();

/// One-time dispatched SIMD integer dot product for symmetric SQ8 (`[-127, 127]`) vectors.
#[inline(always)]
pub fn dot_sq8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_sq8_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_sq8_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let f = DOT_SQ8_FN.get_or_init(|| dot_sq8_scalar);
        f(a, b)
    }
}

/// # Safety
/// Caller must ensure the `avx2` target feature is supported by the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_sq8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let ones = _mm256_set1_epi16(1);
    let (mut acc0, mut acc1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 64 <= n {
        let a0 = _mm256_loadu_si256(pa.add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(pb.add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(pa.add(i + 32) as *const __m256i);
        let b1 = _mm256_loadu_si256(pb.add(i + 32) as *const __m256i);
        // |a| (u8) * sign(b, a) (i8) == a*b; pairs summed in i16: 2*127*127 = 32258 < 32767 -> no saturation
        let p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(b0, a0));
        let p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(a1), _mm256_sign_epi8(b1, a1));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
        i += 64;
    }
    if i + 32 <= n {
        let a0 = _mm256_loadu_si256(pa.add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(pb.add(i) as *const __m256i);
        let p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(a0), _mm256_sign_epi8(b0, a0));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
        i += 32;
    }
    let mut sum = hsum_epi32_avx2(_mm256_add_epi32(acc0, acc1));
    while i < n {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_avx2(v: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    let low = _mm256_castsi256_si128(v);
    let high = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(low, high);
    let shuf64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, shuf64);
    let shuf32 = _mm_shuffle_epi32(sum64, 1);
    let sum32 = _mm_add_epi32(sum64, shuf32);
    _mm_cvtsi128_si32(sum32)
}

#[inline]
pub fn dot_sq8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let mut sum = 0i32;
    let mut i = 0;
    while i + 4 <= n {
        sum += a[i] as i32 * b[i] as i32
            + a[i + 1] as i32 * b[i + 1] as i32
            + a[i + 2] as i32 * b[i + 2] as i32
            + a[i + 3] as i32 * b[i + 3] as i32;
        i += 4;
    }
    while i < n {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    sum
}

/// ARM64 NEON integer dot product for symmetric SQ8 (`[-127, 127]`) vectors.
/// Uses `vmull_s8` / `vpadalq_s16` / `vaddvq_s32` with dual-lane accumulation.
///
/// # Safety
/// Requires an aarch64 CPU with standard NEON support. `a` and `b` must be valid slices.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_sq8_neon(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut i = 0;

    while i + 32 <= n {
        let a0 = vld1q_s8(pa.add(i));
        let b0 = vld1q_s8(pb.add(i));
        let p0_low = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
        let p0_high = vmull_high_s8(a0, b0);
        acc0 = vpadalq_s16(acc0, p0_low);
        acc0 = vpadalq_s16(acc0, p0_high);

        let a1 = vld1q_s8(pa.add(i + 16));
        let b1 = vld1q_s8(pb.add(i + 16));
        let p1_low = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
        let p1_high = vmull_high_s8(a1, b1);
        acc1 = vpadalq_s16(acc1, p1_low);
        acc1 = vpadalq_s16(acc1, p1_high);

        i += 32;
    }

    if i + 16 <= n {
        let a0 = vld1q_s8(pa.add(i));
        let b0 = vld1q_s8(pb.add(i));
        let p0_low = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
        let p0_high = vmull_high_s8(a0, b0);
        acc0 = vpadalq_s16(acc0, p0_low);
        acc0 = vpadalq_s16(acc0, p0_high);
        i += 16;
    }

    let mut sum = vaddvq_s32(vaddq_s32(acc0, acc1));
    while i < n {
        sum += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    sum
}

/// Pure SIMD dot product for unit-normalized vectors (L2 = 1.0).
/// Uses runtime CPU dispatch: AVX2 + F16C + FMA on x86_64, NEON FMLA on aarch64, with scalar fallback.
#[inline(always)]
pub fn dot_product_f16(query_normalized: &[f32], stored_f16: &[f16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_product_f16_avx2(query_normalized, stored_f16) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_f16_neon(query_normalized, stored_f16) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_product_f16_scalar(query_normalized, stored_f16)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_product_f16_avx2(query: &[f32], stored: &[f16]) -> f32 {
    use core::arch::x86_64::*;

    let len = query.len().min(stored.len());
    let mut i = 0;

    let mut dot_acc0 = _mm256_setzero_ps();
    let mut dot_acc1 = _mm256_setzero_ps();

    let q_ptr = query.as_ptr();
    let s_ptr = stored.as_ptr() as *const u16;

    while i + 16 <= len {
        let s_raw0 = _mm_loadu_si128(s_ptr.add(i) as *const __m128i);
        let s_ps0 = _mm256_cvtph_ps(s_raw0);
        let q_ps0 = _mm256_loadu_ps(q_ptr.add(i));
        dot_acc0 = _mm256_fmadd_ps(q_ps0, s_ps0, dot_acc0);

        let s_raw1 = _mm_loadu_si128(s_ptr.add(i + 8) as *const __m128i);
        let s_ps1 = _mm256_cvtph_ps(s_raw1);
        let q_ps1 = _mm256_loadu_ps(q_ptr.add(i + 8));
        dot_acc1 = _mm256_fmadd_ps(q_ps1, s_ps1, dot_acc1);

        i += 16;
    }

    if i + 8 <= len {
        let s_raw = _mm_loadu_si128(s_ptr.add(i) as *const __m128i);
        let s_ps = _mm256_cvtph_ps(s_raw);
        let q_ps = _mm256_loadu_ps(q_ptr.add(i));
        dot_acc0 = _mm256_fmadd_ps(q_ps, s_ps, dot_acc0);
        i += 8;
    }

    let mut dot = hsum_avx2(_mm256_add_ps(dot_acc0, dot_acc1));

    while i < len {
        dot += query[i] * stored[i].to_f32();
        i += 1;
    }

    dot
}

#[allow(dead_code)]
#[inline]
fn dot_product_f16_scalar(query: &[f32], stored: &[f16]) -> f32 {
    let mut dot = 0.0f32;
    let len = query.len().min(stored.len());
    let mut i = 0;

    while i + 4 <= len {
        dot += query[i] * stored[i].to_f32()
            + query[i + 1] * stored[i + 1].to_f32()
            + query[i + 2] * stored[i + 2].to_f32()
            + query[i + 3] * stored[i + 3].to_f32();
        i += 4;
    }

    while i < len {
        dot += query[i] * stored[i].to_f32();
        i += 1;
    }

    dot
}

/// Accelerated ARM64 NEON dot product for unit-normalized vectors (f32 query x f16 stored).
/// Unpacks f16 in 4-wide lanes and uses `vfmaq_f32` (hardware FMLA) with `vaddvq_f32` reduction.
///
/// # Safety
/// Requires an aarch64 CPU with standard NEON support. `query` and `stored` must be valid slices.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_product_f16_neon(query: &[f32], stored: &[f16]) -> f32 {
    use core::arch::aarch64::*;
    let len = query.len().min(stored.len());
    let mut i = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let q_ptr = query.as_ptr();

    while i + 8 <= len {
        let q0 = vld1q_f32(q_ptr.add(i));
        let s0_arr = [
            stored[i].to_f32(),
            stored[i + 1].to_f32(),
            stored[i + 2].to_f32(),
            stored[i + 3].to_f32(),
        ];
        let s0 = vld1q_f32(s0_arr.as_ptr());
        acc0 = vfmaq_f32(acc0, q0, s0);

        let q1 = vld1q_f32(q_ptr.add(i + 4));
        let s1_arr = [
            stored[i + 4].to_f32(),
            stored[i + 5].to_f32(),
            stored[i + 6].to_f32(),
            stored[i + 7].to_f32(),
        ];
        let s1 = vld1q_f32(s1_arr.as_ptr());
        acc1 = vfmaq_f32(acc1, q1, s1);

        i += 8;
    }

    if i + 4 <= len {
        let q0 = vld1q_f32(q_ptr.add(i));
        let s0_arr = [
            stored[i].to_f32(),
            stored[i + 1].to_f32(),
            stored[i + 2].to_f32(),
            stored[i + 3].to_f32(),
        ];
        let s0 = vld1q_f32(s0_arr.as_ptr());
        acc0 = vfmaq_f32(acc0, q0, s0);
        i += 4;
    }

    let mut dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < len {
        dot += query[i] * stored[i].to_f32();
        i += 1;
    }
    dot
}

/// Pure SIMD dot product for two f16 slices directly (used during HNSW heuristic diversity checks).
#[inline(always)]
pub fn dot_product_f16_pair(a: &[f16], b: &[f16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_product_f16_pair_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_f16_pair_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_product_f16_pair_scalar(a, b)
}

#[allow(dead_code)]
#[inline]
fn dot_product_f16_pair_scalar(a: &[f16], b: &[f16]) -> f32 {
    let mut dot = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x.to_f32() * y.to_f32();
    }
    dot
}

/// Accelerated ARM64 NEON dot product for two f16 slices.
///
/// # Safety
/// Requires an aarch64 CPU with standard NEON support. `a` and `b` must be valid slices.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_product_f16_pair_neon(a: &[f16], b: &[f16]) -> f32 {
    use core::arch::aarch64::*;
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);

    while i + 8 <= len {
        let a0_arr = [
            a[i].to_f32(),
            a[i + 1].to_f32(),
            a[i + 2].to_f32(),
            a[i + 3].to_f32(),
        ];
        let b0_arr = [
            b[i].to_f32(),
            b[i + 1].to_f32(),
            b[i + 2].to_f32(),
            b[i + 3].to_f32(),
        ];
        let a0 = vld1q_f32(a0_arr.as_ptr());
        let b0 = vld1q_f32(b0_arr.as_ptr());
        acc0 = vfmaq_f32(acc0, a0, b0);

        let a1_arr = [
            a[i + 4].to_f32(),
            a[i + 5].to_f32(),
            a[i + 6].to_f32(),
            a[i + 7].to_f32(),
        ];
        let b1_arr = [
            b[i + 4].to_f32(),
            b[i + 5].to_f32(),
            b[i + 6].to_f32(),
            b[i + 7].to_f32(),
        ];
        let a1 = vld1q_f32(a1_arr.as_ptr());
        let b1 = vld1q_f32(b1_arr.as_ptr());
        acc1 = vfmaq_f32(acc1, a1, b1);

        i += 8;
    }

    if i + 4 <= len {
        let a0_arr = [
            a[i].to_f32(),
            a[i + 1].to_f32(),
            a[i + 2].to_f32(),
            a[i + 3].to_f32(),
        ];
        let b0_arr = [
            b[i].to_f32(),
            b[i + 1].to_f32(),
            b[i + 2].to_f32(),
            b[i + 3].to_f32(),
        ];
        let a0 = vld1q_f32(a0_arr.as_ptr());
        let b0 = vld1q_f32(b0_arr.as_ptr());
        acc0 = vfmaq_f32(acc0, a0, b0);
        i += 4;
    }

    let mut dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < len {
        dot += a[i].to_f32() * b[i].to_f32();
        i += 1;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_product_f16_pair_avx2(a: &[f16], b: &[f16]) -> f32 {
    use core::arch::x86_64::*;

    let len = a.len().min(b.len());
    let mut i = 0;

    let mut dot_acc0 = _mm256_setzero_ps();
    let mut dot_acc1 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr() as *const u16;
    let b_ptr = b.as_ptr() as *const u16;

    while i + 16 <= len {
        let a_raw0 = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
        let b_raw0 = _mm_loadu_si128(b_ptr.add(i) as *const __m128i);
        let a_ps0 = _mm256_cvtph_ps(a_raw0);
        let b_ps0 = _mm256_cvtph_ps(b_raw0);
        dot_acc0 = _mm256_fmadd_ps(a_ps0, b_ps0, dot_acc0);

        let a_raw1 = _mm_loadu_si128(a_ptr.add(i + 8) as *const __m128i);
        let b_raw1 = _mm_loadu_si128(b_ptr.add(i + 8) as *const __m128i);
        let a_ps1 = _mm256_cvtph_ps(a_raw1);
        let b_ps1 = _mm256_cvtph_ps(b_raw1);
        dot_acc1 = _mm256_fmadd_ps(a_ps1, b_ps1, dot_acc1);

        i += 16;
    }

    if i + 8 <= len {
        let a_raw = _mm_loadu_si128(a_ptr.add(i) as *const __m128i);
        let b_raw = _mm_loadu_si128(b_ptr.add(i) as *const __m128i);
        let a_ps = _mm256_cvtph_ps(a_raw);
        let b_ps = _mm256_cvtph_ps(b_raw);
        dot_acc0 = _mm256_fmadd_ps(a_ps, b_ps, dot_acc0);
        i += 8;
    }

    let mut dot = hsum_avx2(_mm256_add_ps(dot_acc0, dot_acc1));
    while i < len {
        dot += a[i].to_f32() * b[i].to_f32();
        i += 1;
    }
    dot
}

/// Pure SIMD dot product for f32 unit-normalized vectors.
#[inline]
pub fn dot_product_f32(query_normalized: &[f32], stored_f32: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_f32_avx2(query_normalized, stored_f32) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_f32_neon(query_normalized, stored_f32) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    dot_product_f32_scalar(query_normalized, stored_f32)
}

/// Accelerated ARM64 NEON dot product for f32 vectors using native `vld1q_f32` and `vfmaq_f32`.
///
/// # Safety
/// Requires an aarch64 CPU with standard NEON support. `query` and `stored` must be valid slices.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_product_f32_neon(query: &[f32], stored: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let len = query.len().min(stored.len());
    let mut i = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let q_ptr = query.as_ptr();
    let s_ptr = stored.as_ptr();

    while i + 8 <= len {
        let q0 = vld1q_f32(q_ptr.add(i));
        let s0 = vld1q_f32(s_ptr.add(i));
        acc0 = vfmaq_f32(acc0, q0, s0);

        let q1 = vld1q_f32(q_ptr.add(i + 4));
        let s1 = vld1q_f32(s_ptr.add(i + 4));
        acc1 = vfmaq_f32(acc1, q1, s1);

        i += 8;
    }

    if i + 4 <= len {
        let q0 = vld1q_f32(q_ptr.add(i));
        let s0 = vld1q_f32(s_ptr.add(i));
        acc0 = vfmaq_f32(acc0, q0, s0);
        i += 4;
    }

    let mut dot = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < len {
        dot += query[i] * stored[i];
        i += 1;
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_f32_avx2(query: &[f32], stored: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = query.len().min(stored.len());
    let mut i = 0;

    let mut dot_acc0 = _mm256_setzero_ps();
    let mut dot_acc1 = _mm256_setzero_ps();

    let q_ptr = query.as_ptr();
    let s_ptr = stored.as_ptr();

    while i + 16 <= len {
        let q0 = _mm256_loadu_ps(q_ptr.add(i));
        let s0 = _mm256_loadu_ps(s_ptr.add(i));
        dot_acc0 = _mm256_fmadd_ps(q0, s0, dot_acc0);

        let q1 = _mm256_loadu_ps(q_ptr.add(i + 8));
        let s1 = _mm256_loadu_ps(s_ptr.add(i + 8));
        dot_acc1 = _mm256_fmadd_ps(q1, s1, dot_acc1);

        i += 16;
    }

    if i + 8 <= len {
        let q = _mm256_loadu_ps(q_ptr.add(i));
        let s = _mm256_loadu_ps(s_ptr.add(i));
        dot_acc0 = _mm256_fmadd_ps(q, s, dot_acc0);
        i += 8;
    }

    let mut dot = hsum_avx2(_mm256_add_ps(dot_acc0, dot_acc1));

    while i < len {
        dot += query[i] * stored[i];
        i += 1;
    }

    dot
}

#[allow(dead_code)]
#[inline]
fn dot_product_f32_scalar(query: &[f32], stored: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let len = query.len().min(stored.len());
    let mut i = 0;

    while i + 4 <= len {
        dot += query[i] * stored[i]
            + query[i + 1] * stored[i + 1]
            + query[i + 2] * stored[i + 2]
            + query[i + 3] * stored[i + 3];
        i += 4;
    }

    while i < len {
        dot += query[i] * stored[i];
        i += 1;
    }

    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorMatch {
    pub node_id: u32,
    pub name: String,
    pub score: f32,
}

pub const HNSW_CACHE_MAGIC: &[u8; 8] = b"STARKHNC";
pub const HNSW_CACHE_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HnswCacheHeader {
    pub magic: [u8; 8],
    pub generation: u64,
    pub version: u32,
    pub vector_count: u32,
    pub active_count: u32,
    pub max_layer: u32,
    pub entry_point: u32,
    pub nodes_len: u32,
    pub payload_crc32: u32,
    pub rng_state_lo: u32,
    pub rng_state_hi: u32,
    pub l0_links_len: u32,
    pub _reserved: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<HnswCacheHeader>() == 64);

/// Stack-inline / heap BitSet for zero-allocation visited node tracking during HNSW search.
/// Stack array [u64; 32] covers up to 2,048 nodes with ZERO heap allocations.
#[allow(clippy::large_enum_variant)]
enum VisitedBitSet {
    Inline([u64; 32]),
    Heap(Vec<u64>),
}

impl VisitedBitSet {
    #[inline]
    fn new(capacity: usize) -> Self {
        if capacity <= 2048 {
            Self::Inline([0u64; 32])
        } else {
            let num_words = capacity.div_ceil(64);
            Self::Heap(vec![0u64; num_words.max(1)])
        }
    }

    /// Marks `id` as visited. Returns `true` if `id` was NOT previously visited.
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        let idx = (id as usize) >> 6;
        let mask = 1u64 << ((id as usize) & 63);
        match self {
            Self::Inline(ref mut arr) => {
                if idx < 32 {
                    let was_visited = (arr[idx] & mask) != 0;
                    arr[idx] |= mask;
                    !was_visited
                } else {
                    let mut heap = Vec::with_capacity((idx + 1).max(64));
                    heap.extend_from_slice(arr);
                    heap.resize(idx + 1, 0);
                    let was_visited = (heap[idx] & mask) != 0;
                    heap[idx] |= mask;
                    *self = Self::Heap(heap);
                    !was_visited
                }
            }
            Self::Heap(ref mut words) => {
                if idx >= words.len() {
                    words.resize(idx + 1, 0);
                }
                let was_visited = (words[idx] & mask) != 0;
                words[idx] |= mask;
                !was_visited
            }
        }
    }
}

/// Max-Heap item ordered by highest similarity first (for candidate exploration queue `C`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct MaxSimNode(f32, u32);

impl Eq for MaxSimNode {}

impl PartialOrd for MaxSimNode {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxSimNode {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// Min-Heap item ordered by lowest similarity first (for dynamic result window `W` of size `ef`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct MinSimNode(f32, u32);

impl Eq for MinSimNode {}

impl PartialOrd for MinSimNode {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinSimNode {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

#[derive(Clone, Debug)]
pub struct HnswNode {
    pub node_id: u32,
    pub vector_id: u32,
    pub level: usize,
    /// Upper layers `1..=level` (`layers[lc - 1]` corresponds to HNSW layer `lc`).
    /// Layer 0 links are stored contiguously in `HnswIndex::l0_links`.
    pub layers: Vec<Vec<u32>>,
    pub deleted: bool,
}

/// Hierarchical Navigable Small World (HNSW) multi-layer graph index (v1.0 Ultimate Legacy).
#[derive(Clone, Debug)]
pub struct HnswIndex {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub level_mult: f64,
    pub nodes: Vec<Option<HnswNode>>,
    /// Flat Layer-0 neighbor array with stride `m_max0 + 1` per `node_id`.
    /// Element `base` = neighbor count; elements `base + 1 .. base + 1 + count` = neighbor IDs.
    pub l0_links: Vec<u32>,
    /// Flat `node_id -> vector_id` table (`u32::MAX` if absent or deleted) for L1-resident traversal.
    pub vector_ids: Vec<u32>,
    pub entry_point: Option<u32>,
    pub max_layer: usize,
    pub active_count: usize,
    rng_state: u64,
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            m: HNSW_M,
            m_max0: HNSW_M_MAX0,
            ef_construction: HNSW_EF_CONSTRUCTION,
            ef_search: HNSW_EF_SEARCH,
            level_mult: 1.0 / (HNSW_M as f64).ln(),
            nodes: Vec::new(),
            l0_links: Vec::new(),
            vector_ids: Vec::new(),
            entry_point: None,
            max_layer: 0,
            active_count: 0,
            rng_state: 0x853C49E6748FEA9B,
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.l0_links.clear();
        self.vector_ids.clear();
        self.entry_point = None;
        self.max_layer = 0;
        self.active_count = 0;
        self.rng_state = 0x853C49E6748FEA9B;
    }

    #[inline(always)]
    pub fn layer_neighbors(&self, node_id: u32, layer: usize) -> &[u32] {
        if layer == 0 {
            let stride = self.m_max0 + 1;
            let base = (node_id as usize) * stride;
            if base < self.l0_links.len() {
                let count = (self.l0_links[base] as usize).min(self.m_max0);
                &self.l0_links[base + 1..base + 1 + count]
            } else {
                &[]
            }
        } else if let Some(Some(hnode)) = self.nodes.get(node_id as usize) {
            hnode
                .layers
                .get(layer - 1)
                .map(|v| v.as_slice())
                .unwrap_or(&[])
        } else {
            &[]
        }
    }

    #[inline]
    fn set_layer_neighbors(&mut self, node_id: u32, layer: usize, neighbors: &[u32]) {
        if layer == 0 {
            let stride = self.m_max0 + 1;
            let base = (node_id as usize) * stride;
            if base + stride > self.l0_links.len() {
                self.l0_links.resize(base + stride, 0);
            }
            let count = neighbors.len().min(self.m_max0);
            self.l0_links[base] = count as u32;
            self.l0_links[base + 1..base + 1 + count].copy_from_slice(&neighbors[..count]);
        } else if let Some(Some(hnode)) = self.nodes.get_mut(node_id as usize) {
            if layer - 1 < hnode.layers.len() {
                hnode.layers[layer - 1].clear();
                hnode.layers[layer - 1].extend_from_slice(neighbors);
            }
        }
    }

    #[inline]
    fn remove_neighbor_from_layer(&mut self, host_id: u32, layer: usize, target_id: u32) {
        if layer == 0 {
            let stride = self.m_max0 + 1;
            let base = (host_id as usize) * stride;
            if base < self.l0_links.len() {
                let count = (self.l0_links[base] as usize).min(self.m_max0);
                let mut write_idx = 0usize;
                for read_idx in 0..count {
                    let v = self.l0_links[base + 1 + read_idx];
                    if v != target_id {
                        self.l0_links[base + 1 + write_idx] = v;
                        write_idx += 1;
                    }
                }
                self.l0_links[base] = write_idx as u32;
            }
        } else if let Some(Some(hnode)) = self.nodes.get_mut(host_id as usize) {
            if let Some(vec) = hnode.layers.get_mut(layer - 1) {
                vec.retain(|&x| x != target_id);
            }
        }
    }

    /// Removes all backreferences to `node_id` across the HNSW graph before recycling its slot (V2 Fix).
    fn detach_node_edges(&mut self, node_id: u32) {
        let total_nodes = self.nodes.len() as u32;
        for other_id in 0..total_nodes {
            if other_id == node_id {
                continue;
            }
            self.remove_neighbor_from_layer(other_id, 0, node_id);
            if let Some(Some(hnode)) = self.nodes.get_mut(other_id as usize) {
                for layer_vec in &mut hnode.layers {
                    layer_vec.retain(|&x| x != node_id);
                }
            }
        }
        let stride = self.m_max0 + 1;
        let base = (node_id as usize) * stride;
        if base < self.l0_links.len() {
            self.l0_links[base] = 0;
        }
    }

    /// Repairs tombstone nodes by reconnecting their neighbors and clearing deleted slots.
    pub fn repair_tombstones(&mut self, storage: &StarkStorage) {
        let is_f16 = storage.header().vector_format == 1;
        let deleted_ids: Vec<u32> = self
            .nodes
            .iter()
            .flatten()
            .filter(|n| n.deleted)
            .map(|n| n.node_id)
            .collect();
        if deleted_ids.is_empty() {
            return;
        }

        for &del_id in &deleted_ids {
            let del_l0: Vec<u32> = self
                .layer_neighbors(del_id, 0)
                .iter()
                .copied()
                .filter(|&id| {
                    self.nodes
                        .get(id as usize)
                        .and_then(|x| x.as_ref())
                        .map(|n| !n.deleted)
                        .unwrap_or(false)
                })
                .collect();

            let total_nodes = self.nodes.len() as u32;
            for u in 0..total_nodes {
                if u == del_id {
                    continue;
                }
                let u_active = self
                    .nodes
                    .get(u as usize)
                    .and_then(|x| x.as_ref())
                    .map(|n| !n.deleted)
                    .unwrap_or(false);
                if !u_active {
                    continue;
                }
                if self.layer_neighbors(u, 0).contains(&del_id) {
                    let mut merged: Vec<u32> = self
                        .layer_neighbors(u, 0)
                        .iter()
                        .copied()
                        .filter(|&x| x != del_id && x != u)
                        .collect();
                    for &cand in &del_l0 {
                        if cand != u && !merged.contains(&cand) {
                            merged.push(cand);
                        }
                    }
                    let mut scored: Vec<(f32, u32)> = merged
                        .into_iter()
                        .map(|other_id| (self.compute_pair_sim(u, other_id, storage, is_f16), other_id))
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                    let selected = self.select_neighbors_heuristic(&scored, self.m_max0, storage, is_f16);
                    self.set_layer_neighbors(u, 0, &selected);
                }
                self.remove_neighbor_from_layer(u, 0, del_id);
                if let Some(Some(hnode)) = self.nodes.get_mut(u as usize) {
                    for layer_vec in &mut hnode.layers {
                        layer_vec.retain(|&x| x != del_id);
                    }
                }
            }

            let stride = self.m_max0 + 1;
            let base = (del_id as usize) * stride;
            if base < self.l0_links.len() {
                self.l0_links[base] = 0;
            }
            if let Some(slot) = self.nodes.get_mut(del_id as usize) {
                *slot = None;
            }
            if let Some(vslot) = self.vector_ids.get_mut(del_id as usize) {
                *vslot = u32::MAX;
            }
        }
    }

    /// Updates `vector_id` references across all indexed HNSW nodes after vector arena compaction.
    pub fn remap_vector_ids(&mut self, remap: &[u32]) {
        for slot in self.nodes.iter_mut().flatten() {
            if let Some(&new_vid) = remap.get(slot.vector_id as usize) {
                if new_vid != u32::MAX {
                    slot.vector_id = new_vid;
                    if !slot.deleted {
                        if let Some(vslot) = self.vector_ids.get_mut(slot.node_id as usize) {
                            *vslot = new_vid;
                        }
                    }
                }
            }
        }
    }

    /// Serializes HNSW graph structure to binary cache format v2 with CRC32 protection (V7 Fix)
    pub fn serialize_cache(&self, generation: u64, vector_count: u32) -> Vec<u8> {
        let mut payload = Vec::with_capacity(self.l0_links.len() * 4 + self.nodes.len() * 24);
        for &val in &self.l0_links {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        for slot in &self.nodes {
            match slot {
                None => {
                    payload.push(0u8);
                }
                Some(node) => {
                    payload.push(1u8);
                    payload.extend_from_slice(&node.node_id.to_le_bytes());
                    payload.extend_from_slice(&node.vector_id.to_le_bytes());
                    payload.extend_from_slice(&(node.level as u32).to_le_bytes());
                    payload.push(if node.deleted { 1u8 } else { 0u8 });
                    payload.extend_from_slice(&(node.layers.len() as u32).to_le_bytes());
                    for layer in &node.layers {
                        payload.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                        for &nb in layer {
                            payload.extend_from_slice(&nb.to_le_bytes());
                        }
                    }
                }
            }
        }

        let crc = crc32fast::hash(&payload);
        let header = HnswCacheHeader {
            magic: *HNSW_CACHE_MAGIC,
            generation,
            version: HNSW_CACHE_VERSION,
            vector_count,
            active_count: self.active_count as u32,
            max_layer: self.max_layer as u32,
            entry_point: self.entry_point.unwrap_or(u32::MAX),
            nodes_len: self.nodes.len() as u32,
            payload_crc32: crc,
            rng_state_lo: self.rng_state as u32,
            rng_state_hi: (self.rng_state >> 32) as u32,
            l0_links_len: self.l0_links.len() as u32,
            _reserved: [0; 2],
        };

        let mut bytes = Vec::with_capacity(std::mem::size_of::<HnswCacheHeader>() + payload.len());
        bytes.extend_from_slice(bytemuck::bytes_of(&header));
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Deserializes and validates HNSW graph structure from binary cache bytes (V7 Fix: pod_read_unaligned)
    pub fn deserialize_cache(
        &mut self,
        bytes: &[u8],
        expected_generation: u64,
        expected_vector_count: u32,
    ) -> Result<bool> {
        let header_size = std::mem::size_of::<HnswCacheHeader>();
        if bytes.len() < header_size {
            return Ok(false);
        }
        let header: HnswCacheHeader = bytemuck::pod_read_unaligned(&bytes[..header_size]);

        if &header.magic != HNSW_CACHE_MAGIC || header.version != HNSW_CACHE_VERSION {
            return Ok(false);
        }
        if header.generation != expected_generation || header.vector_count != expected_vector_count {
            return Ok(false);
        }

        let payload = &bytes[header_size..];
        let calc_crc = crc32fast::hash(payload);
        if calc_crc != header.payload_crc32 {
            return Ok(false);
        }

        let l0_len = header.l0_links_len as usize;
        if payload.len() < l0_len * 4 {
            return Ok(false);
        }
        let mut l0_links = Vec::with_capacity(l0_len);
        let mut offset = 0usize;
        for _ in 0..l0_len {
            let v = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            offset += 4;
            l0_links.push(v);
        }

        let nodes_len = header.nodes_len as usize;
        let mut nodes: Vec<Option<HnswNode>> = Vec::with_capacity(nodes_len);
        let mut vector_ids = vec![u32::MAX; nodes_len];

        for idx in 0..nodes_len {
            if offset >= payload.len() {
                return Ok(false);
            }
            let is_some = payload[offset];
            offset += 1;

            if is_some == 0 {
                nodes.push(None);
            } else if is_some == 1 {
                if offset + 13 > payload.len() {
                    return Ok(false);
                }
                let node_id = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                offset += 4;
                let vector_id = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                offset += 4;
                let level = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                let deleted = payload[offset] != 0;
                offset += 1;

                if offset + 4 > payload.len() {
                    return Ok(false);
                }
                let layers_count = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;

                let mut layers = Vec::with_capacity(layers_count);
                for _ in 0..layers_count {
                    if offset + 4 > payload.len() {
                        return Ok(false);
                    }
                    let nb_len = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4;

                    if offset + nb_len * 4 > payload.len() {
                        return Ok(false);
                    }
                    let mut layer = Vec::with_capacity(nb_len);
                    for _ in 0..nb_len {
                        let nb = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        layer.push(nb);
                    }
                    layers.push(layer);
                }

                if !deleted && idx < vector_ids.len() {
                    vector_ids[idx] = vector_id;
                }

                nodes.push(Some(HnswNode {
                    node_id,
                    vector_id,
                    level,
                    layers,
                    deleted,
                }));
            } else {
                return Ok(false);
            }
        }

        self.nodes = nodes;
        self.l0_links = l0_links;
        self.vector_ids = vector_ids;
        self.active_count = header.active_count as usize;
        self.max_layer = header.max_layer as usize;
        self.rng_state = (header.rng_state_lo as u64) | ((header.rng_state_hi as u64) << 32);
        self.entry_point = if header.entry_point == u32::MAX {
            None
        } else {
            Some(header.entry_point)
        };

        Ok(true)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Probabilistic level generation: l = floor(-ln(uniform(0,1)) * m_L)
    #[inline]
    fn sample_level(&mut self) -> usize {
        let u = ((self.next_u64() >> 11) as f64 + 1.0) * (1.0 / ((1u64 << 53) as f64));
        let l = (-u.ln() * self.level_mult).floor() as usize;
        l.min(12)
    }

    #[inline]
    fn compute_sim(
        &self,
        query_norm: &[f32],
        node_id: u32,
        storage: &StarkStorage,
        is_f16: bool,
    ) -> f32 {
        let vid = *self.vector_ids.get(node_id as usize).unwrap_or(&u32::MAX);
        if vid != u32::MAX {
            if is_f16 {
                if let Ok(vf16) = storage.get_vector_f16(vid) {
                    return dot_product_f16(query_norm, vf16);
                }
            } else if let Ok(vf32) = storage.get_vector_f32(vid) {
                return dot_product_f32(query_norm, vf32);
            }
        }
        -1.0
    }

    #[inline]
    fn compute_sim_sq8(
        &self,
        query_i8: &[i8],
        query_scale: f32,
        node_id: u32,
        storage: &StarkStorage,
    ) -> f32 {
        let vid = *self.vector_ids.get(node_id as usize).unwrap_or(&u32::MAX);
        if vid != u32::MAX {
            if let Ok((codes, meta)) = storage.get_vector_sq8(vid) {
                let dot = dot_sq8(query_i8, codes) as f32;
                return dot * (query_scale * meta.scale);
            }
        }
        -1.0
    }

    #[inline]
    fn compute_pair_sim(
        &self,
        node_a: u32,
        node_b: u32,
        storage: &StarkStorage,
        is_f16: bool,
    ) -> f32 {
        let va = *self.vector_ids.get(node_a as usize).unwrap_or(&u32::MAX);
        let vb = *self.vector_ids.get(node_b as usize).unwrap_or(&u32::MAX);
        if va == u32::MAX || vb == u32::MAX {
            return -1.0;
        }
        if is_f16 {
            if let (Ok(sa), Ok(sb)) = (storage.get_vector_f16(va), storage.get_vector_f16(vb)) {
                return dot_product_f16_pair(sa, sb);
            }
        } else if let (Ok(sa), Ok(sb)) = (storage.get_vector_f32(va), storage.get_vector_f32(vb)) {
            return dot_product_f32(sa, sb);
        }
        -1.0
    }

    /// Greedy search on upper layers (ef = 1) finding local maximum similarity node.
    fn greedy_search_layer(
        &self,
        query_norm: &[f32],
        mut curr_ep: u32,
        mut curr_sim: f32,
        layer: usize,
        storage: &StarkStorage,
        is_f16: bool,
    ) -> (u32, f32) {
        loop {
            let mut improved = false;
            for &nb_id in self.layer_neighbors(curr_ep, layer) {
                let sim = self.compute_sim(query_norm, nb_id, storage, is_f16);
                if sim > curr_sim {
                    curr_sim = sim;
                    curr_ep = nb_id;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        (curr_ep, curr_sim)
    }

    /// Greedy search on upper layers using SQ8 dot product.
    fn greedy_search_layer_sq8(
        &self,
        query_i8: &[i8],
        query_scale: f32,
        mut curr_ep: u32,
        mut curr_sim: f32,
        layer: usize,
        storage: &StarkStorage,
    ) -> (u32, f32) {
        loop {
            let mut improved = false;
            for &nb_id in self.layer_neighbors(curr_ep, layer) {
                let sim = self.compute_sim_sq8(query_i8, query_scale, nb_id, storage);
                if sim > curr_sim {
                    curr_sim = sim;
                    curr_ep = nb_id;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        (curr_ep, curr_sim)
    }

    /// Beam search at `layer` with dynamic candidate window `ef` using BinaryHeap and VisitedBitSet.
    #[allow(clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query_norm: &[f32],
        ep_id: u32,
        ep_sim: f32,
        ef: usize,
        layer: usize,
        storage: &StarkStorage,
        is_f16: bool,
    ) -> Vec<(f32, u32)> {
        let mut visited = VisitedBitSet::new(self.nodes.len());
        visited.insert(ep_id);

        let mut candidates = BinaryHeap::with_capacity(ef * 2);
        let mut results = BinaryHeap::with_capacity(ef + 1);

        candidates.push(MaxSimNode(ep_sim, ep_id));
        results.push(MinSimNode(ep_sim, ep_id));

        while let Some(MaxSimNode(c_sim, c_id)) = candidates.pop() {
            let worst_sim = results.peek().map(|x| x.0).unwrap_or(-1.0);
            if c_sim < worst_sim && results.len() >= ef {
                break;
            }

            for &nb_id in self.layer_neighbors(c_id, layer) {
                if visited.insert(nb_id) {
                    let sim = self.compute_sim(query_norm, nb_id, storage, is_f16);
                    let current_worst = results.peek().map(|x| x.0).unwrap_or(-1.0);
                    if results.len() < ef || sim > current_worst {
                        candidates.push(MaxSimNode(sim, nb_id));
                        results.push(MinSimNode(sim, nb_id));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = results
            .into_iter()
            .map(|MinSimNode(s, id)| (s, id))
            .collect();
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        out
    }

    /// Beam search at `layer` using SQ8 integer dot product (Phase 1 of Two-Phase HNSW).
    #[allow(clippy::too_many_arguments)]
    fn search_layer_sq8(
        &self,
        query_i8: &[i8],
        query_scale: f32,
        ep_id: u32,
        ep_sim: f32,
        ef: usize,
        layer: usize,
        storage: &StarkStorage,
    ) -> SmallVec<[(f32, u32); 64]> {
        let mut visited = VisitedBitSet::new(self.nodes.len());
        visited.insert(ep_id);

        let mut candidates = BinaryHeap::with_capacity(ef * 2);
        let mut results = BinaryHeap::with_capacity(ef + 1);

        candidates.push(MaxSimNode(ep_sim, ep_id));
        results.push(MinSimNode(ep_sim, ep_id));

        while let Some(MaxSimNode(c_sim, c_id)) = candidates.pop() {
            let worst_sim = results.peek().map(|x| x.0).unwrap_or(-1.0);
            if c_sim < worst_sim && results.len() >= ef {
                break;
            }

            for &nb_id in self.layer_neighbors(c_id, layer) {
                if visited.insert(nb_id) {
                    let sim = self.compute_sim_sq8(query_i8, query_scale, nb_id, storage);
                    let current_worst = results.peek().map(|x| x.0).unwrap_or(-1.0);
                    if results.len() < ef || sim > current_worst {
                        candidates.push(MaxSimNode(sim, nb_id));
                        results.push(MinSimNode(sim, nb_id));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: SmallVec<[(f32, u32); 64]> = results
            .into_iter()
            .map(|MinSimNode(s, id)| (s, id))
            .collect();
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        out
    }

    /// HNSW Algorithm 4: Diversity-preserving neighbor selection heuristic.
    /// Ensures navigable links across manifolds while filling up to `max_m` connections.
    fn select_neighbors_heuristic(
        &self,
        candidates: &[(f32, u32)],
        max_m: usize,
        storage: &StarkStorage,
        is_f16: bool,
    ) -> Vec<u32> {
        if candidates.len() <= max_m {
            return candidates.iter().map(|&(_, id)| id).collect();
        }

        let mut selected: Vec<u32> = Vec::with_capacity(max_m);
        let mut pruned: Vec<u32> = Vec::with_capacity(candidates.len());

        for &(sim_to_q, cand_id) in candidates {
            if selected.len() >= max_m {
                break;
            }
            let mut diverse = true;
            for &sel_id in &selected {
                let sim_between = self.compute_pair_sim(cand_id, sel_id, storage, is_f16);
                if sim_between > sim_to_q {
                    diverse = false;
                    break;
                }
            }
            if diverse {
                selected.push(cand_id);
            } else {
                pruned.push(cand_id);
            }
        }

        for id in pruned {
            if selected.len() >= max_m {
                break;
            }
            selected.push(id);
        }

        selected
    }

    /// Inserts or updates a node's vector in the HNSW multi-layer navigable graph.
    pub fn insert(&mut self, node_id: u32, vector_id: u32, storage: &StarkStorage) {
        let is_f16 = storage.header().vector_format == 1;
        let dims = storage.header().vector_dims as usize;

        // Avoid 6 KB heap allocation per insert by reading into a stack buffer for dims <= 2048
        let mut stack_buf = [0.0f32; 2048];
        let mut heap_buf: Vec<f32>;
        let query_norm: &[f32] = if dims <= 2048 {
            if storage.get_vector_into(vector_id, &mut stack_buf[..dims]).is_err() {
                return;
            }
            &stack_buf[..dims]
        } else {
            heap_buf = vec![0.0f32; dims];
            if storage.get_vector_into(vector_id, &mut heap_buf).is_err() {
                return;
            }
            &heap_buf
        };

        let idx = node_id as usize;
        if idx < self.nodes.len() && self.nodes[idx].is_some() {
            // V2 Fix: Clean up all backreferences from neighbors before reusing node_id slot
            self.detach_node_edges(node_id);
        }

        let level = self.sample_level();
        let mut upper_layers = Vec::with_capacity(level);
        for _ in 1..=level {
            upper_layers.push(Vec::with_capacity(self.m + 1));
        }

        if idx >= self.nodes.len() {
            self.nodes.resize(idx + 1, None);
        }
        if idx >= self.vector_ids.len() {
            self.vector_ids.resize(idx + 1, u32::MAX);
        }
        let stride = self.m_max0 + 1;
        if (idx + 1) * stride > self.l0_links.len() {
            self.l0_links.resize((idx + 1) * stride, 0);
        }
        self.l0_links[idx * stride] = 0;

        let was_active = self.nodes[idx]
            .as_ref()
            .map(|n| !n.deleted)
            .unwrap_or(false);

        self.nodes[idx] = Some(HnswNode {
            node_id,
            vector_id,
            level,
            layers: upper_layers,
            deleted: false,
        });
        self.vector_ids[idx] = vector_id;
        if !was_active {
            self.active_count += 1;
        }

        let ep_id = match self.entry_point {
            Some(ep) if ep != node_id => ep,
            _ => {
                self.entry_point = Some(node_id);
                self.max_layer = level;
                return;
            }
        };

        let max_l = self.max_layer;
        let mut curr_ep = ep_id;
        let mut curr_sim = self.compute_sim(query_norm, curr_ep, storage, is_f16);

        // Phase 1: Greedy descent from top layer down to level + 1
        if max_l > level {
            for lc in ((level + 1)..=max_l).rev() {
                let (next_ep, next_sim) =
                    self.greedy_search_layer(query_norm, curr_ep, curr_sim, lc, storage, is_f16);
                curr_ep = next_ep;
                curr_sim = next_sim;
            }
        }

        // Phase 2: Search and bidirectionally connect from min(level, max_layer) down to 0
        let start_layer = level.min(max_l);
        for lc in (0..=start_layer).rev() {
            let top_candidates = self.search_layer(
                query_norm,
                curr_ep,
                curr_sim,
                self.ef_construction,
                lc,
                storage,
                is_f16,
            );

            if let Some(&(best_sim, best_id)) = top_candidates.first() {
                curr_ep = best_id;
                curr_sim = best_sim;
            }

            let max_m = if lc == 0 { self.m_max0 } else { self.m };
            let filtered: Vec<(f32, u32)> = top_candidates
                .into_iter()
                .filter(|&(_, id)| id != node_id)
                .collect();
            let neighbors = self.select_neighbors_heuristic(&filtered, max_m, storage, is_f16);

            self.set_layer_neighbors(node_id, lc, &neighbors);

            // Add reverse edges and prune if neighbor degree exceeds max_m
            for nb_id in neighbors {
                let nb_idx = nb_id as usize;
                if lc == 0 {
                    let existing = self.layer_neighbors(nb_id, 0);
                    if existing.contains(&node_id) {
                        continue;
                    }
                    if existing.len() < max_m {
                        let base = nb_idx * stride;
                        let cnt = self.l0_links[base] as usize;
                        self.l0_links[base + 1 + cnt] = node_id;
                        self.l0_links[base] = (cnt + 1) as u32;
                    } else {
                        let mut nb_list: SmallVec<[u32; 36]> = SmallVec::from_slice(existing);
                        nb_list.push(node_id);
                        let mut scored: SmallVec<[(f32, u32); 36]> = nb_list
                            .into_iter()
                            .map(|other_id| {
                                (
                                    self.compute_pair_sim(nb_id, other_id, storage, is_f16),
                                    other_id,
                                )
                            })
                            .collect();
                        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                        let pruned_list =
                            self.select_neighbors_heuristic(&scored, max_m, storage, is_f16);
                        self.set_layer_neighbors(nb_id, 0, &pruned_list);
                    }
                } else {
                    let needs_prune = if let Some(Some(nb_node)) = self.nodes.get_mut(nb_idx) {
                        if lc - 1 < nb_node.layers.len() {
                            if !nb_node.layers[lc - 1].contains(&node_id) {
                                nb_node.layers[lc - 1].push(node_id);
                            }
                            nb_node.layers[lc - 1].len() > max_m
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if needs_prune {
                        let nb_list = self.nodes[nb_idx].as_ref().unwrap().layers[lc - 1].clone();
                        let mut scored: Vec<(f32, u32)> = nb_list
                            .into_iter()
                            .map(|other_id| {
                                let s = self.compute_pair_sim(nb_id, other_id, storage, is_f16);
                                (s, other_id)
                            })
                            .collect();
                        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                        let pruned_list =
                            self.select_neighbors_heuristic(&scored, max_m, storage, is_f16);
                        if let Some(Some(nb_node)) = self.nodes.get_mut(nb_idx) {
                            nb_node.layers[lc - 1] = pruned_list;
                        }
                    }
                }
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(node_id);
        }
    }

    /// Marks a node as deleted in HNSW and updates entry_point if necessary.
    pub fn mark_deleted(&mut self, node_id: u32) {
        let idx = node_id as usize;
        if let Some(Some(hnode)) = self.nodes.get_mut(idx) {
            if !hnode.deleted {
                hnode.deleted = true;
                self.active_count = self.active_count.saturating_sub(1);
            }
        }
        if let Some(vslot) = self.vector_ids.get_mut(idx) {
            *vslot = u32::MAX;
        }
        if self.entry_point == Some(node_id) {
            let mut best_ep: Option<(usize, u32)> = None;
            for item in self.nodes.iter().flatten() {
                if !item.deleted {
                    match best_ep {
                        None => best_ep = Some((item.level, item.node_id)),
                        Some((lvl, _)) if item.level > lvl => {
                            best_ep = Some((item.level, item.node_id))
                        }
                        _ => {}
                    }
                }
            }
            if let Some((lvl, id)) = best_ep {
                self.entry_point = Some(id);
                self.max_layer = lvl;
            } else {
                self.entry_point = None;
                self.max_layer = 0;
            }
        }
    }

    /// Two-Phase HNSW Search (Milestone M4):
    /// Phase 1: Coarse upper-layer descent & Layer-0 beam search using AVX2 SQ8 integer dot product (`dot_sq8`).
    /// Phase 2: Exact reranking of top candidates using `dot_product_f16` (or `dot_product_f32`).
    pub fn search_two_phase(
        &self,
        query_norm: &[f32],
        query_i8: &[i8],
        query_scale: f32,
        top_k: usize,
        storage: &StarkStorage,
    ) -> SmallVec<[(f32, u32); 64]> {
        self.search_two_phase_filtered(query_norm, query_i8, query_scale, top_k, storage, |_| true)
    }

    pub fn search_two_phase_filtered<F>(
        &self,
        query_norm: &[f32],
        query_i8: &[i8],
        query_scale: f32,
        top_k: usize,
        storage: &StarkStorage,
        filter: F,
    ) -> SmallVec<[(f32, u32); 64]>
    where
        F: Fn(u32) -> bool,
    {
        if !storage.has_sq8() {
            let hits = self.search(query_norm, top_k, storage);
            return hits.into_iter().filter(|&(_, id)| filter(id)).collect();
        }

        let ep_id = match self.entry_point {
            Some(ep) => ep,
            None => return SmallVec::new(),
        };

        let mut curr_ep = ep_id;
        let mut curr_sim = self.compute_sim_sq8(query_i8, query_scale, curr_ep, storage);

        // Phase 1a: Greedy descent on upper layers using SQ8
        if self.max_layer > 0 {
            for lc in (1..=self.max_layer).rev() {
                let (next_ep, next_sim) = self.greedy_search_layer_sq8(
                    query_i8,
                    query_scale,
                    curr_ep,
                    curr_sim,
                    lc,
                    storage,
                );
                curr_ep = next_ep;
                curr_sim = next_sim;
            }
        }

        // Phase 1b: Layer 0 beam search using SQ8
        let ef = self.ef_search.max(top_k * 3);
        let coarse_candidates =
            self.search_layer_sq8(query_i8, query_scale, curr_ep, curr_sim, ef, 0, storage);

        // Phase 2: Exact reranking of top coarse candidates using f16/f32 SIMD dot product
        let is_f16 = storage.header().vector_format == 1;
        let nodes_slice = storage.nodes();
        let rerank_limit = coarse_candidates.len().min((top_k * 2).max(20));
        let mut exact_results: SmallVec<[(f32, u32); 64]> = SmallVec::with_capacity(rerank_limit);

        for &(_, nid) in coarse_candidates.iter().take(rerank_limit) {
            if *self.vector_ids.get(nid as usize).unwrap_or(&u32::MAX) == u32::MAX {
                continue;
            }
            if !filter(nid) {
                continue;
            }
            if let Some(nrec) = nodes_slice.get(nid as usize) {
                if (nrec.flags & FLAG_DELETED) != 0 || nrec.vector_id == u32::MAX {
                    continue;
                }
                let exact_score = self.compute_sim(query_norm, nid, storage, is_f16);
                exact_results.push((exact_score, nid));
            }
        }

        exact_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        exact_results.truncate(top_k);
        exact_results
    }

    /// Executes multi-layer HNSW search returning top_k active (non-deleted) `(score, node_id)` matches.
    pub fn search(
        &self,
        query_norm: &[f32],
        top_k: usize,
        storage: &StarkStorage,
    ) -> Vec<(f32, u32)> {
        self.search_filtered(query_norm, top_k, storage, |_| true)
    }

    /// Executes multi-layer HNSW search with a candidate predicate `filter(node_id)`.
    pub fn search_filtered<F>(
        &self,
        query_norm: &[f32],
        top_k: usize,
        storage: &StarkStorage,
        filter: F,
    ) -> Vec<(f32, u32)>
    where
        F: Fn(u32) -> bool,
    {
        let ep_id = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let is_f16 = storage.header().vector_format == 1;
        let mut curr_ep = ep_id;
        let mut curr_sim = self.compute_sim(query_norm, curr_ep, storage, is_f16);

        // 1. Greedy descent from max_layer down to layer 1
        if self.max_layer > 0 {
            for lc in (1..=self.max_layer).rev() {
                let (next_ep, next_sim) =
                    self.greedy_search_layer(query_norm, curr_ep, curr_sim, lc, storage, is_f16);
                curr_ep = next_ep;
                curr_sim = next_sim;
            }
        }

        // 2. Beam search on layer 0 with ef = max(ef_search, top_k * 2)
        let ef = self.ef_search.max(top_k * 2);
        let candidates =
            self.search_layer(query_norm, curr_ep, curr_sim, ef, 0, storage, is_f16);

        let nodes_slice = storage.nodes();
        let mut results = Vec::with_capacity(top_k);
        for (score, nid) in candidates {
            if *self.vector_ids.get(nid as usize).unwrap_or(&u32::MAX) == u32::MAX {
                continue;
            }
            if !filter(nid) {
                continue;
            }
            if let Some(nrec) = nodes_slice.get(nid as usize) {
                if (nrec.flags & FLAG_DELETED) != 0 || nrec.vector_id == u32::MAX {
                    continue;
                }
                results.push((score, nid));
                if results.len() >= top_k {
                    break;
                }
            }
        }

        results
    }

    /// Returns the number of deleted tombstone slots still present in the HNSW index.
    pub fn tombstone_count(&self) -> usize {
        self.nodes
            .iter()
            .flatten()
            .filter(|n| n.deleted)
            .count()
    }
}

impl StarkGraph {
    pub fn attach_vector(&mut self, node_id: u32, vector: &[f32]) -> Result<u32> {
        let old_vec_id = {
            let nodes = self.storage.nodes();
            let node = nodes
                .get(node_id as usize)
                .ok_or_else(|| anyhow::anyhow!("Node {} not found", node_id))?;
            if (node.flags & FLAG_DELETED) != 0 {
                bail!("Node {} is deleted", node_id);
            }
            node.vector_id
        };
        if old_vec_id != u32::MAX {
            self.storage.release_vector(old_vec_id)?;
        }

        let vec_id = self.storage.write_vector(vector)?;
        let nodes = self.storage.nodes_mut();
        if let Some(node) = nodes.get_mut(node_id as usize) {
            node.vector_id = vec_id;
            self.storage.flush()?;
            self.hnsw.insert(node_id, vec_id, &self.storage);
            if let Some(j) = &mut self.vacuum {
                j.mark_vector(node_id);
            }
            Ok(vec_id)
        } else {
            bail!("Node {} not found", node_id);
        }
    }

    /// Adaptive vector search: automatically uses HNSW when active vectors >= 256,
    /// or direct SIMD scan for small vector sets (< 256).
    pub fn search_vectors(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorMatch>> {
        if self.hnsw.active_count >= HNSW_AUTO_THRESHOLD {
            self.search_vectors_hnsw(query, top_k)
        } else {
            self.search_vectors_bruteforce(query, top_k)
        }
    }

    /// Sub-millisecond Two-Phase HNSW search using SQ8 AVX2 coarse traversal + F16C exact reranking.
    pub fn search_vectors_hnsw(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorMatch>> {
        if top_k == 0 || self.hnsw.active_count == 0 {
            return Ok(Vec::new());
        }

        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        if query_norm_sq == 0.0 {
            return Ok(Vec::new());
        }

        let normalized_buf: Vec<f32>;
        let query_norm_ref: &[f32] = if (query_norm_sq - 1.0).abs() < 1e-4 {
            query
        } else {
            let inv_query_norm = 1.0 / query_norm_sq.sqrt();
            normalized_buf = query.iter().map(|&x| x * inv_query_norm).collect();
            &normalized_buf
        };

        let nodes = self.storage.nodes();

        if self.hnsw.active_count >= 4096 && self.storage.has_sq8() {
            let dims = query_norm_ref.len();
            let mut stack_qi8 = [0i8; 2048];
            let mut heap_qi8: Vec<i8>;
            let q_i8_slice: &mut [i8] = if dims <= 2048 {
                &mut stack_qi8[..dims]
            } else {
                heap_qi8 = vec![0i8; dims];
                &mut heap_qi8
            };
            let q_meta = quantize_sq8(query_norm_ref, q_i8_slice);

            let raw_hits = self.hnsw.search_two_phase(
                query_norm_ref,
                q_i8_slice,
                q_meta.scale,
                top_k,
                &self.storage,
            );
            let mut matches = Vec::with_capacity(raw_hits.len());
            for (score, node_id) in raw_hits {
                if let Some(node) = nodes.get(node_id as usize) {
                    let name = self
                        .storage
                        .get_string(node.name_offset, node.name_len as u32)
                        .unwrap_or("Unknown")
                        .to_string();
                    matches.push(VectorMatch {
                        node_id,
                        name,
                        score,
                    });
                }
            }
            Ok(matches)
        } else {
            let raw_hits = self.hnsw.search(query_norm_ref, top_k, &self.storage);
            let mut matches = Vec::with_capacity(raw_hits.len());
            for (score, node_id) in raw_hits {
                if let Some(node) = nodes.get(node_id as usize) {
                    let name = self
                        .storage
                        .get_string(node.name_offset, node.name_len as u32)
                        .unwrap_or("Unknown")
                        .to_string();
                    matches.push(VectorMatch {
                        node_id,
                        name,
                        score,
                    });
                }
            }
            Ok(matches)
        }
    }

    /// Exact Brute-Force SIMD scan across all vectors.
    pub fn search_vectors_bruteforce(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<VectorMatch>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let is_f16 = self.storage.header().vector_format == 1;
        let nodes = self.storage.nodes();

        let query_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        if query_norm_sq == 0.0 {
            return Ok(Vec::new());
        }
        let normalized_buf: Vec<f32>;
        let query_norm_ref: &[f32] = if (query_norm_sq - 1.0).abs() < 1e-4 {
            query
        } else {
            let inv_query_norm = 1.0 / query_norm_sq.sqrt();
            normalized_buf = query.iter().map(|&x| x * inv_query_norm).collect();
            &normalized_buf
        };

        let mut top_candidates: Vec<(f32, u32, u32, u16)> = Vec::with_capacity(top_k + 1);

        for node in nodes {
            if (node.flags & FLAG_DELETED) != 0 || node.vector_id == u32::MAX {
                continue;
            }

            let score = if is_f16 {
                if let Ok(vec_f16) = self.storage.get_vector_f16(node.vector_id) {
                    dot_product_f16(query_norm_ref, vec_f16)
                } else {
                    continue;
                }
            } else if let Ok(vec_f32) = self.storage.get_vector_f32(node.vector_id) {
                dot_product_f32(query_norm_ref, vec_f32)
            } else {
                continue;
            }
            ;

            if top_candidates.len() < top_k {
                top_candidates.push((score, node.id, node.name_offset, node.name_len));
                if top_candidates.len() == top_k {
                    top_candidates
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                }
            } else if score > top_candidates[top_k - 1].0 {
                top_candidates[top_k - 1] = (score, node.id, node.name_offset, node.name_len);
                top_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            }
        }

        let mut matches = Vec::with_capacity(top_candidates.len());
        for (score, node_id, name_offset, name_len) in top_candidates {
            let name = self
                .storage
                .get_string(name_offset, name_len as u32)
                .unwrap_or("Unknown")
                .to_string();
            matches.push(VectorMatch {
                node_id,
                name,
                score,
            });
        }

        Ok(matches)
    }

    /// Hybrid Vector + Associative Recall implemented as a thin wrapper over `bound_attack` (`text = None`).
    /// Guarantees exact deduplication by `node_id` and multi-source graph expansion (V5 Fix).
    pub fn hybrid_recall(
        &self,
        query: &[f32],
        top_vectors: usize,
        expansion_depth: usize,
    ) -> Result<Vec<crate::graph::ActivatedNode>> {
        let bq = crate::hybrid::BoundQuery {
            vector: Some(query),
            text: None,
            anchors: &[],
            tags_all: &[],
            node_type: None,
            since: None,
            until: None,
            top_k: top_vectors.max(10),
            depth: expansion_depth.min(6) as u8,
            weights: crate::hybrid::FusionWeights::default(),
        };
        let hits = self.bound_attack(&bq)?;
        let nodes_slice = self.storage.nodes();
        Ok(hits
            .into_iter()
            .map(|h| {
                let (node_type, tags, content) =
                    if let Some(n) = nodes_slice.get(h.node_id as usize) {
                        let nt = crate::format::node_type_to_str(n.node_type).to_string();
                        let tags_raw = self
                            .storage
                            .get_string(n.tags_offset, n.tags_len as u32)
                            .unwrap_or("");
                        let t_vec: Vec<String> = if tags_raw.is_empty() {
                            Vec::new()
                        } else {
                            tags_raw
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect()
                        };
                        let c_str = self
                            .storage
                            .get_string(n.content_offset, n.content_len)
                            .unwrap_or("")
                            .to_string();
                        (nt, t_vec, c_str)
                    } else {
                        ("Concept".to_string(), Vec::new(), String::new())
                    };
                crate::graph::ActivatedNode {
                    id: h.node_id,
                    name: h.name.clone(),
                    node_type,
                    tags,
                    content,
                    energy: h.score,
                    depth: h.depth,
                    path: vec![h.name],
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StarkStorage;

    #[test]
    fn test_f16_vector_storage_and_cosine_similarity() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_f16_vectors");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("vectors_f16.starkdb");

        let dims: usize = 384;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        assert_eq!(db.storage.header().vector_format, 1);
        assert_eq!(db.storage.bytes_per_vector(), dims * 2);

        let node_id1 = db.get_or_create_node("QuantumPhysics", 6).unwrap();
        let node_id2 = db.get_or_create_node("MachineLearning", 6).unwrap();
        let node_id3 = db.get_or_create_node("CookingRecipes", 6).unwrap();

        let mut v1 = vec![0.0f32; dims];
        let mut v2 = vec![0.0f32; dims];
        let mut v3 = vec![0.0f32; dims];

        for i in 0..dims {
            v1[i] = ((i as f32) * 0.05).sin();
            v2[i] = ((i as f32) * 0.05 + 0.1).sin();
            v3[i] = ((i as f32) * 0.9 + 1.5).cos();
        }

        db.attach_vector(node_id1, &v1).unwrap();
        db.attach_vector(node_id2, &v2).unwrap();
        db.attach_vector(node_id3, &v3).unwrap();

        let mut query = v1.clone();
        query[0] += 0.02;

        let results = db.search_vectors(&query, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "QuantumPhysics");
        assert!(results[0].score > 0.99);
        assert_eq!(results[1].name, "MachineLearning");
        assert!(results[1].score > 0.85);
        assert_eq!(results[2].name, "CookingRecipes");
        assert!(results[2].score < results[1].score);

        let retrieved_v1 = db.storage.get_vector(0).unwrap();
        assert_eq!(retrieved_v1.len(), dims);
        let v1_norm: f32 = (v1.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        for i in 0..dims {
            let expected = v1[i] / v1_norm;
            let diff = (retrieved_v1[i] - expected).abs();
            assert!(diff < 0.002);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_hnsw_index_accuracy_and_recall() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_hnsw_recall");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("hnsw_recall.starkdb");

        let dims: usize = 128;
        let num_vectors: usize = 1000;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        let make_vec = |seed: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dims];
            let mut state = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            for (d, item) in v.iter_mut().enumerate().take(dims) {
                state = state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let unit = ((z >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
                let cluster_center = (((seed / 25) * 19 + d) as f32 * 0.17).sin();
                *item = cluster_center + unit * 0.25;
            }
            v
        };

        for i in 0..num_vectors {
            let name = format!("VecNode_{}", i);
            let nid = db.get_or_create_node(&name, 6).unwrap();
            let v = make_vec(i);
            db.attach_vector(nid, &v).unwrap();
        }

        assert_eq!(db.hnsw.active_count, num_vectors);
        assert!(db.hnsw.max_layer >= 1, "HNSW must build multiple hierarchical layers for 1,000 vectors");

        let mut total_recall = 0.0f32;
        let num_queries = 20;
        for q_idx in 0..num_queries {
            let mut q = make_vec(q_idx * 47 + 3);
            q[0] += 0.05;

            let exact = db.search_vectors_bruteforce(&q, 10).unwrap();
            let approx = db.search_vectors(&q, 10).unwrap();
            assert_eq!(approx.len(), 10);

            let exact_ids: Vec<u32> = exact.iter().map(|m| m.node_id).collect();
            let hits = approx
                .iter()
                .filter(|m| exact_ids.contains(&m.node_id))
                .count();
            total_recall += (hits as f32) / 10.0;
        }

        let avg_recall = total_recall / (num_queries as f32);
        println!("[HNSW TEST] Average Recall@10 across {} queries on 1,000 vectors: {:.3}", num_queries, avg_recall);
        assert!(
            avg_recall >= 0.90,
            "Expected HNSW Recall@10 >= 0.90, got {:.3}",
            avg_recall
        );

        let test_q = make_vec(100);
        let top_before = db.search_vectors(&test_q, 10).unwrap();
        let best_name = top_before[0].name.clone();
        let best_id = top_before[0].node_id;

        db.delete_node(&best_name).unwrap();
        assert_eq!(db.hnsw.active_count, num_vectors - 1);

        let top_after = db.search_vectors(&test_q, 10).unwrap();
        assert_eq!(top_after.len(), 10);
        assert!(
            top_after.iter().all(|m| m.node_id != best_id),
            "Deleted node #{} ('{}') must not appear in HNSW search results",
            best_id,
            best_name
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_sq8_dot_matches_scalar() {
        for &dims in &[64usize, 100, 128, 384, 768, 1536] {
            for seed in 0..50u64 {
                let mut f_a = vec![0.0f32; dims];
                let mut f_b = vec![0.0f32; dims];
                let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(17);
                for i in 0..dims {
                    state = state.wrapping_add(0x9E3779B97F4A7C15);
                    let mut z = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    z ^= z >> 31;
                    f_a[i] = ((z & 0xFFFF) as f32 / 32768.0) - 1.0;
                    f_b[i] = (((z >> 16) & 0xFFFF) as f32 / 32768.0) - 1.0;
                }
                let mut q_a = vec![0i8; dims];
                let mut q_b = vec![0i8; dims];
                let meta_a = quantize_sq8(&f_a, &mut q_a);
                let meta_b = quantize_sq8(&f_b, &mut q_b);

                assert!(q_a.iter().all(|&x| x >= -127));
                assert!(q_b.iter().all(|&x| x >= -127));
                assert_eq!(meta_a.sum_q, q_a.iter().map(|&x| x as i32).sum::<i32>());
                assert_eq!(meta_b.sum_q, q_b.iter().map(|&x| x as i32).sum::<i32>());

                let scalar = dot_sq8_scalar(&q_a, &q_b);
                let dispatched = dot_sq8(&q_a, &q_b);
                assert_eq!(
                    dispatched, scalar,
                    "SIMD dot_sq8 must equal scalar dot_sq8_scalar exactly for dims={} seed={}",
                    dims, seed
                );
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let avx2 = unsafe { dot_sq8_avx2(&q_a, &q_b) };
                    assert_eq!(avx2, scalar);
                }
                #[cfg(target_arch = "aarch64")]
                {
                    let neon = unsafe { dot_sq8_neon(&q_a, &q_b) };
                    assert_eq!(neon, scalar);
                }

                let f16_a: Vec<f16> = f_a.iter().map(|&x| f16::from_f32(x)).collect();
                let f16_b: Vec<f16> = f_b.iter().map(|&x| f16::from_f32(x)).collect();

                let scalar_f16 = dot_product_f16_scalar(&f_a, &f16_b);
                let disp_f16 = dot_product_f16(&f_a, &f16_b);
                assert!((disp_f16 - scalar_f16).abs() < 1e-3);

                let scalar_pair = dot_product_f16_pair_scalar(&f16_a, &f16_b);
                let disp_pair = dot_product_f16_pair(&f16_a, &f16_b);
                assert!((disp_pair - scalar_pair).abs() < 1e-3);

                let scalar_f32 = dot_product_f32_scalar(&f_a, &f_b);
                let disp_f32 = dot_product_f32(&f_a, &f_b);
                assert!((disp_f32 - scalar_f32).abs() < 1e-3);

                #[cfg(target_arch = "aarch64")]
                {
                    let neon_f16 = unsafe { dot_product_f16_neon(&f_a, &f16_b) };
                    assert!((neon_f16 - scalar_f16).abs() < 1e-3);

                    let neon_pair = unsafe { dot_product_f16_pair_neon(&f16_a, &f16_b) };
                    assert!((neon_pair - scalar_pair).abs() < 1e-3);

                    let neon_f32 = unsafe { dot_product_f32_neon(&f_a, &f_b) };
                    assert!((neon_f32 - scalar_f32).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn test_hnsw_two_phase_recall() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_hnsw_sq8_10k_recall");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("hnsw_sq8_10k.starkdb");

        let dims: usize = 128;
        let num_vectors: usize = 1_000;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        let make_vec = |seed: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dims];
            let cluster_id = (seed / 25) as u64;
            let mut state = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(7);
            for (d, item) in v.iter_mut().enumerate().take(dims) {
                state = state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                let noise = ((z >> 40) as f32 / ((1u64 << 24) as f32)) * 2.0 - 1.0;
                let center = (((cluster_id as usize) * 19 + d) as f32 * 0.17).sin();
                *item = center + noise * 0.22;
            }
            v
        };

        for i in 0..num_vectors {
            let name = format!("SQ8Node_{}", i);
            let nid = db.get_or_create_node(&name, 6).unwrap();
            let v = make_vec(i);
            db.attach_vector(nid, &v).unwrap();
        }

        assert!(db.storage.has_sq8(), "SQ8 feature must be active after attaching vectors");
        assert_eq!(db.hnsw.active_count, num_vectors);

        let mut total_recall = 0.0f32;
        let num_queries = 20;
        for q_idx in 0..num_queries {
            let mut q = make_vec(q_idx * 97 + 11);
            q[0] += 0.03;
            let q_norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
            for x in &mut q {
                *x /= q_norm;
            }
            let mut q_i8 = vec![0i8; dims];
            let q_meta = quantize_sq8(&q, &mut q_i8);

            let exact = db.search_vectors_bruteforce(&q, 10).unwrap();
            let two_phase = db
                .hnsw
                .search_two_phase(&q, &q_i8, q_meta.scale, 10, &db.storage);
            assert_eq!(two_phase.len(), 10);

            let exact_ids: Vec<u32> = exact.iter().map(|m| m.node_id).collect();
            let hits = two_phase
                .iter()
                .filter(|&&(_, nid)| exact_ids.contains(&nid))
                .count();
            total_recall += (hits as f32) / 10.0;
        }

        let avg_recall = total_recall / (num_queries as f32);
        println!(
            "[TWO-PHASE SQ8 HNSW TEST] Average Recall@10 across {} queries on {} vectors: {:.3}",
            num_queries, num_vectors, avg_recall
        );
        assert!(
            avg_recall >= 0.95,
            "Expected Two-Phase SQ8 HNSW Recall@10 >= 0.95, got {:.3}",
            avg_recall
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_free_vector_slot_reuse() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_vec_slot_reuse");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("vec_reuse.starkdb");

        let dims = 64usize;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        for i in 0..10 {
            let nid = db.get_or_create_node(&format!("Node_{}", i), 0).unwrap();
            let v: Vec<f32> = (0..dims).map(|d| ((i * 11 + d) as f32 * 0.1).sin()).collect();
            db.attach_vector(nid, &v).unwrap();
        }
        assert_eq!(db.storage.header().vector_count, 10);

        // 1. Re-attaching a vector to an existing node must reuse its slot without growing vector_count
        for i in 0..10 {
            let v_new: Vec<f32> = (0..dims).map(|d| ((i * 19 + d) as f32 * 0.2).cos()).collect();
            db.attach_vector(i as u32, &v_new).unwrap();
        }
        assert_eq!(
            db.storage.header().vector_count,
            10,
            "vector_count must not grow when re-attaching vectors to existing nodes"
        );
        assert_eq!(db.storage.header().dead_vector_slots, 0);

        // 2. Deleting 3 nodes must release 3 vector slots into free_vectors_head
        db.delete_node("Node_2").unwrap();
        db.delete_node("Node_5").unwrap();
        db.delete_node("Node_8").unwrap();
        assert_eq!(db.storage.header().dead_vector_slots, 3);
        assert_ne!(db.storage.header().free_vectors_head, u32::MAX);

        // 3. Creating 3 new nodes and attaching vectors must reuse the 3 freed vector slots
        for i in 0..3 {
            let nid = db.get_or_create_node(&format!("FreshNode_{}", i), 0).unwrap();
            let v: Vec<f32> = (0..dims).map(|d| ((i * 29 + d) as f32 * 0.15).sin()).collect();
            db.attach_vector(nid, &v).unwrap();
        }
        assert_eq!(
            db.storage.header().vector_count,
            10,
            "vector_count must remain 10 after recycling 3 freed vector slots"
        );
        assert_eq!(db.storage.header().dead_vector_slots, 0);
        assert_eq!(db.storage.header().free_vectors_head, u32::MAX);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_hnsw_recycled_slot_no_phantom_edges() {
        let tmp_dir = std::env::temp_dir().join("starkdb_test_hnsw_no_phantom");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let db_path = tmp_dir.join("no_phantom.starkdb");

        let dims = 64usize;
        let mut db =
            StarkGraph::new(StarkStorage::open_or_create(&db_path, dims as u16).unwrap()).unwrap();

        // Cluster A: positive first half, zero second half
        let make_cluster_a = |i: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dims];
            for (d, val) in v.iter_mut().enumerate().take(dims / 2) {
                *val = 1.0 + ((i + d) as f32 * 0.01).sin() * 0.05;
            }
            v
        };
        // Cluster B: zero first half, positive second half (orthogonal to Cluster A)
        let make_cluster_b = |i: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; dims];
            for (d, val) in v.iter_mut().enumerate().take(dims).skip(dims / 2) {
                *val = 1.0 + ((i + d) as f32 * 0.01).cos() * 0.05;
            }
            v
        };

        // Insert 40 nodes in Cluster A (IDs 0..40) and 40 nodes in Cluster B (IDs 40..80)
        for i in 0..40 {
            let nid = db.get_or_create_node(&format!("ClusterA_{}", i), 0).unwrap();
            db.attach_vector(nid, &make_cluster_a(i)).unwrap();
        }
        for i in 0..40 {
            let nid = db.get_or_create_node(&format!("ClusterB_{}", i), 0).unwrap();
            db.attach_vector(nid, &make_cluster_b(i)).unwrap();
        }

        // Verify ClusterA_5 (node_id == 5) initially has neighbors in Cluster A
        let old_a_neighbors: Vec<u32> = db
            .hnsw
            .layer_neighbors(5, 0)
            .iter()
            .copied()
            .filter(|&id| id < 40)
            .collect();
        assert!(!old_a_neighbors.is_empty(), "Node 5 must initially have neighbors in Cluster A");

        // Delete ClusterA_5 (node_id == 5), then create a new node that reuses node_id == 5
        // and attach a Cluster B vector to it!
        db.delete_node("ClusterA_5").unwrap();
        let recycled_id = db.get_or_create_node("Recycled_As_ClusterB", 0).unwrap();
        assert_eq!(recycled_id, 5, "Must recycle node slot 5");

        let new_b_vec = make_cluster_b(99);
        db.attach_vector(recycled_id, &new_b_vec).unwrap();

        // Verify that detach_node_edges removed 5 from all old Cluster A neighbors before re-inserting 5,
        // and since there are 40 Cluster B nodes (> m_max0 = 32), recycled node 5 connects to Cluster B!
        let n5_l0 = db.hnsw.layer_neighbors(recycled_id, 0);
        for a_id in 0..40u32 {
            if a_id == recycled_id {
                continue;
            }
            if db.hnsw.layer_neighbors(a_id, 0).contains(&recycled_id) {
                assert!(
                    n5_l0.contains(&a_id),
                    "Phantom edge detected: node {} points to recycled slot 5 without being in 5's new neighbor list!",
                    a_id
                );
            }
        }
        let cluster_b_neighbors = n5_l0.iter().filter(|&&id| id >= 40).count();
        assert!(
            cluster_b_neighbors >= 20,
            "Recycled slot 5 with Cluster B vector must primarily connect to Cluster B neighbors, got {}",
            cluster_b_neighbors
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
