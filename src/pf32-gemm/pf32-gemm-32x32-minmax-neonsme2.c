// Copyright 2024 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

#if defined(__clang__)
// Apple exposes SME but not non-streaming SVE. This compatibility check runs
// before the SME kernel starts streaming mode. Prevent the compiler from
// vectorizing its offset comparisons into non-streaming SVE instructions.
__attribute__((optnone))
#endif
size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_hybrid_n_step(
    size_t k) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_pack_rhs_uker_config pack_config = {0};
  const struct kai_matmul_uker_config elastic_config = {0};
  const struct kai_matmul_pack_rhs_uker_api pack_api =
      kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
  const struct kai_matmul_pack_rhs_uker_api transpose_pack_api =
      kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
  const struct kai_matmul_uker_api elastic_api =
      kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
  const size_t pack_n_step = pack_api.get_step(&pack_config).n;
  const size_t elastic_n_step = elastic_api.get_step(&elastic_config).n;
  const size_t neon_n_step =
      kai_get_n_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla();
  // The NEON kernel reads one 16-channel bias-and-weights block at a time.
  // Other SME vector lengths need a different NEON kernel.
  if (pack_n_step != 16 || pack_n_step != neon_n_step ||
      elastic_n_step != pack_n_step ||
      transpose_pack_api.get_step(&pack_config).n != pack_n_step) {
    return 0;
  }

  const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args shape = {
      .n = 2 * elastic_n_step,
      .k = k,
  };
  const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride =
      pack_api.get_rhs_packed_stride(&pack_config, &shape);
  const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args
      transpose_stride =
          transpose_pack_api.get_rhs_packed_stride(&pack_config, &shape);
  const struct kai_matmul_uker_rhs_dim_args elastic_shape = {shape.n, k};
  const struct kai_matmul_uker_rhs_stride_args elastic_stride =
      elastic_api.get_rhs_stride(&elastic_config, &elastic_shape);
  if (stride.n != (k + 1) * neon_n_step * sizeof(float) ||
      transpose_stride.n != stride.n || elastic_stride.n != stride.n) {
    return 0;
  }
  for (size_t n = pack_n_step; n <= 2 * elastic_n_step; n += pack_n_step) {
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args index = {
        .n = n,
    };
    const size_t expected_offset =
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
            n, k);
    if (pack_api.get_rhs_packed_offset(&pack_config, &index, &stride) !=
            expected_offset ||
        transpose_pack_api.get_rhs_packed_offset(
            &pack_config, &index, &transpose_stride) != expected_offset) {
      return 0;
    }
  }
  return elastic_n_step;
#else
  assert(
      "Calling KleidiAI RHS packing wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_f32_gemm_minmax_ukernel_6x16__kai_neon(
    size_t m, size_t n, size_t k, const float* lhs, size_t lhs_stride,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  assert(k % sizeof(float) == 0);
  kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla(
      m, n, k / sizeof(float), lhs, lhs_stride, rhs_packed, dst, dst_stride_row,
      dst_stride_col, minmax_params->scalar.min, minmax_params->scalar.max);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}

// Wraps the `kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const struct xnn_f32_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
    assert(k % sizeof(float) == 0);
    assert(dst_stride_col == sizeof(float));
    (void) dst_stride_col;
    const size_t k_elements = k / sizeof(float);

    const struct kai_matmul_uker_config config = {0};
    const struct kai_matmul_uker_api api = kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
    const struct kai_matmul_uker_lhs_dim_args lhs_shape = {m, k_elements};
    const struct kai_matmul_uker_rhs_dim_args rhs_shape = {n, k_elements};
    struct kai_matmul_uker_args args = {
        .flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
        .shape.m = m,
        .shape.n = n,
        .shape.k = k_elements,

        .operand.dst.ptr = dst,
        .operand.dst.stride.m = dst_stride_row,
        .operand.lhs.ptr = lhs_packed,
        .operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape),
        .operand.rhs.ptr = rhs_packed,
        .operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape),

        .activation.clamp.min_ptr = &minmax_params->scalar.min,
        .activation.clamp.max_ptr = &minmax_params->scalar.max,
    };
    api.run(&config, &args);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
