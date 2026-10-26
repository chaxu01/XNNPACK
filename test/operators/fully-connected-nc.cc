// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
extern "C" {
#include "src/xnnpack/compute.h"
}
#include "src/xnnpack/internal.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/math.h"
#include <pthreadpool.h>

#include <cstdlib>
#include <cstring>

#include "src/xnnpack/operator.h"
#include "test/operators/fully-connected-operator-tester.h"

TEST(FULLY_CONNECTED_NC_QS8, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255;
       input_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQS8();
  }
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255;
       input_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .input_zero_point(static_cast<uint8_t>(input_zero_point))
        .kernel_zero_point(0)
        .TestQS8QC2W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255;
       output_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .output_zero_point(static_cast<uint8_t>(output_zero_point))
        .kernel_zero_point(0)
        .TestQS8QC2W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .kernel_zero_point(0)
      .TestQS8QC2W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255;
       input_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .input_zero_point(uint8_t(input_zero_point))
        .kernel_zero_point(8)
        .TestQS8QC4W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255;
       output_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .output_zero_point(uint8_t(output_zero_point))
        .kernel_zero_point(8)
        .TestQS8QC4W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .kernel_zero_point(8)
      .TestQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, unsigned_weights_with_bias) {
  FullyConnectedOperatorTester()
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(8)
      .TestPQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, unsigned_weights_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(8)
      .TestPQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, signed_weights_with_bias) {
  FullyConnectedOperatorTester()
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(0)
      .TestPQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, signed_weights_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(17)
      .input_channels(32)
      .output_channels(29)
      .kernel_zero_point(0)
      .TestPQS8QC4W();
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, invalid_kernel_zero_point) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[output_channels * ((input_channels + 1) / 2)] = {};
  float kernel_scale[output_channels];
  int32_t bias[output_channels] = {};
  std::fill_n(kernel_scale, output_channels, 1.0f);

  for (const uint8_t kernel_zero_point : {1, 7, 9, 255}) {
    xnn_operator_t fully_connected_op = nullptr;
    EXPECT_EQ(
        xnn_create_fully_connected_nc_pqs8_qc4w(
            input_channels, output_channels, input_channels, output_channels,
            /*input_zero_point=*/0, /*input_scale=*/1.0f, kernel_zero_point,
            kernel_scale, kernel, bias, /*output_zero_point=*/0,
            /*output_scale=*/1.0f, /*output_min=*/-128, /*output_max=*/127,
            /*flags=*/0, /*weights_cache=*/nullptr, &fully_connected_op),
        xnn_status_invalid_parameter);
    EXPECT_EQ(fully_connected_op, nullptr);
  }
}

TEST(FULLY_CONNECTED_NC_PQS8_QC4W, transpose_weights_unsupported) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[input_channels * ((output_channels + 1) / 2)] = {};
  float kernel_scale[output_channels] = {};
  int32_t bias[output_channels] = {};
  for (size_t i = 0; i < output_channels; i++) {
    kernel_scale[i] = 1.0f;
  }

  xnn_operator_t fully_connected_op = nullptr;
  EXPECT_EQ(
      xnn_create_fully_connected_nc_pqs8_qc4w(
          input_channels, output_channels, input_channels, output_channels,
          /*input_zero_point=*/0, /*input_scale=*/1.0f,
          /*kernel_zero_point=*/8, kernel_scale, kernel, bias,
          /*output_zero_point=*/0, /*output_scale=*/1.0f,
          /*output_min=*/-128, /*output_max=*/127, XNN_FLAG_TRANSPOSE_WEIGHTS,
          /*weights_cache=*/nullptr, &fully_connected_op),
      xnn_status_unsupported_parameter);
  EXPECT_EQ(fully_connected_op, nullptr);
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255;
       input_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQS8QC8W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255;
       output_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .output_zero_point(uint8_t(output_zero_point))
        .TestQS8QC8W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255;
       input_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .input_zero_point(uint8_t(input_zero_point))
        .TestQU8();
  }
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_kernel_zero_point) {
  for (int32_t kernel_zero_point = 1; kernel_zero_point <= 255;
       kernel_zero_point += 51) {
    FullyConnectedOperatorTester()
        .batch_size(1)
        .input_channels(22)
        .output_channels(19)
        .kernel_zero_point(uint8_t(kernel_zero_point))
        .TestQU8();
  }
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQU8();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF32();
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI && XNN_ENABLE_ARM_SME2 && \
    (XNN_COMPILER_CLANG || XNN_COMPILER_GCC)
class ScopedF32EnvVar {
 public:
  explicit ScopedF32EnvVar(const char* name) : name_(name) {
    const char* value = getenv(name);
    if (value != nullptr) {
      was_set_ = true;
      original_ = value;
    }
  }

  ~ScopedF32EnvVar() {
    if (was_set_) {
      setenv(name_, original_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

  void Set(const char* value) { setenv(name_, value, 1); }

 private:
  const char* name_;
  bool was_set_ = false;
  std::string original_;
};

TEST(FULLY_CONNECTED_NC_PF32, hybrid_sme2_neon) {
  ScopedF32EnvVar sme("XNN_F32_SME");
  ScopedF32EnvVar hybrid("XNN_ENABLE_F32_HYBRID");
  sme.Set("2");
  hybrid.Set("1");

  constexpr size_t batch_size = 131;
  constexpr size_t input_channels = 37;
  constexpr size_t output_channels = 257;
  constexpr size_t input_stride = 43;
  constexpr size_t output_stride = 263;
  constexpr float output_min = -10.0f;
  constexpr float output_max = 10.0f;

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
  xnnpack::Buffer<float> input((batch_size - 1) * input_stride + input_channels,
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> kernel(output_channels * input_channels);
  xnnpack::Buffer<float> bias(output_channels);
  xnnpack::Buffer<float> output((batch_size - 1) * output_stride +
                                output_channels);
  xnnpack::Buffer<float> output_ref(batch_size * output_channels);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

  for (const bool transpose : {false, true}) {
    SCOPED_TRACE(transpose ? "transposed, no bias" : "normal, with bias");
    for (size_t m = 0; m < batch_size; m++) {
      for (size_t n = 0; n < output_channels; n++) {
        float acc = transpose ? 0.0f : bias[n];
        for (size_t k = 0; k < input_channels; k++) {
          acc += input[m * input_stride + k] *
                 kernel[transpose ? k * output_channels + n
                                  : n * input_channels + k];
        }
        output_ref[m * output_channels + n] =
            std::max(std::min(acc, output_max), output_min);
      }
    }

    ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
    xnn_operator_t fully_connected_op = nullptr;
    const enum xnn_status status = xnn_create_fully_connected_nc_pf32(
        input_channels, output_channels, input_stride, output_stride,
        kernel.data(), transpose ? nullptr : bias.data(), output_min,
        output_max,
        XNN_FLAG_INLINE_LHS_PACKING |
            (transpose ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0),
        nullptr, &fully_connected_op);
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
    }
    ASSERT_EQ(xnn_status_success, status);
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
        auto_fully_connected_op(fully_connected_op, xnn_delete_operator);

    if (fully_connected_op->gemm_config->arch != xnn_arch_arm_sme2) {
      GTEST_SKIP();
    }

    std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
        auto_threadpool(pthreadpool_create(4), pthreadpool_destroy);
    ASSERT_NE(nullptr, auto_threadpool.get());
    size_t workspace_size = 0;
    ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_pf32(
                                      fully_connected_op, batch_size,
                                      &workspace_size, auto_threadpool.get()));
    if (fully_connected_op->dynamic_context.gemm->gemm
            .f32_hybrid_neon_ukernel == nullptr) {
      GTEST_SKIP() << "16-channel SME packed layout is unavailable";
    }
    ASSERT_EQ(2, fully_connected_op->num_compute_invocations);
    ASSERT_NE(
        nullptr,
        fully_connected_op->dynamic_context.gemm->gemm.f32_hybrid_neon_ukernel);
    ASSERT_EQ(xnn_parallelization_type_f32_hybrid_queue,
              fully_connected_op->compute[1].type);
    struct gemm_context* gemm = &fully_connected_op->dynamic_context.gemm->gemm;
    ASSERT_NE(nullptr, gemm->packed_w);
    ASSERT_EQ(16, gemm->f32_hybrid_tile_n);
    ASSERT_GE(gemm->f32_hybrid_sme_claim, 1);
    ASSERT_GE(gemm->f32_hybrid_neon_claim, 1);
    ASSERT_EQ(2, gemm->f32_hybrid_sme_threads);
    const char* sme_chunk = getenv("XNN_F32_HYBRID_SME_CHUNK_TILES");
    if (sme_chunk != nullptr) {
      ASSERT_EQ(strtoul(sme_chunk, nullptr, 10),
                gemm->f32_hybrid_sme_claim);
    }

    xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> workspace(
        workspace_size);
    ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_pf32(
                                      fully_connected_op, input.data(),
                                      output.data(), workspace.data()));
    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(fully_connected_op, auto_threadpool.get()));

    // Force both kernels on disjoint tiles of the same packed RHS buffer.
    std::fill(output.begin(), output.end(),
              std::numeric_limits<float>::quiet_NaN());
    xnn_compute_f32_hybrid_gemm(gemm, 0, 0, 0, 16, gemm->f32_hybrid_tile_m);
    xnn_compute_f32_hybrid_gemm(gemm, gemm->f32_hybrid_sme_threads, 16, 0, 16,
                                gemm->f32_hybrid_tile_m);
    for (size_t m = 0; m < gemm->f32_hybrid_tile_m; m++) {
      for (size_t n = 0; n < 32; n++) {
        const float expected = output_ref[m * output_channels + n];
        ASSERT_NEAR(expected, output[m * output_stride + n],
                    std::max(1.0e-4f * std::abs(expected), 1.0e-4f));
      }
    }

    // Running again without reshape must restart the global queue.
    std::fill(output.begin(), output.end(),
              std::numeric_limits<float>::quiet_NaN());
    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(fully_connected_op, auto_threadpool.get()));

    for (size_t m = 0; m < batch_size; m++) {
      for (size_t n = 0; n < output_channels; n++) {
        const float expected = output_ref[m * output_channels + n];
        ASSERT_NEAR(expected, output[m * output_stride + n],
                    std::max(1.0e-4f * std::abs(expected), 1.0e-4f))
            << "at " << m << ", " << n;
      }
    }

    // The same worker split must also handle a single output row (GEMV).
    size_t single_row_workspace_size = 0;
    ASSERT_EQ(xnn_status_success, xnn_reshape_fully_connected_nc_pf32(
                                      fully_connected_op, 1,
                                      &single_row_workspace_size,
                                      auto_threadpool.get()));
    ASSERT_EQ(xnn_parallelization_type_f32_hybrid_queue,
              fully_connected_op
                  ->compute[fully_connected_op->num_compute_invocations - 1]
                  .type);
    ASSERT_LE(single_row_workspace_size, workspace_size);
    ASSERT_EQ(xnn_status_success, xnn_setup_fully_connected_nc_pf32(
                                      fully_connected_op, input.data(),
                                      output.data(), workspace.data()));
    ASSERT_EQ(xnn_status_success,
              xnn_run_operator(fully_connected_op, auto_threadpool.get()));
    for (size_t n = 0; n < output_channels; n++) {
      const float expected = output_ref[n];
      ASSERT_NEAR(expected, output[n],
                  std::max(1.0e-4f * std::abs(expected), 1.0e-4f));
    }
  }
}

TEST(FULLY_CONNECTED_NC_PF32, sme_worker_selection) {
  ScopedF32EnvVar sme("XNN_F32_SME");
  ScopedF32EnvVar hybrid("XNN_ENABLE_F32_HYBRID");
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  constexpr size_t input_channels = 37;
  constexpr size_t output_channels = 257;
  std::vector<float> weights(input_channels * output_channels, 0.0f);
  xnn_operator_t op = nullptr;
  const enum xnn_status status = xnn_create_fully_connected_nc_pf32(
      input_channels, output_channels, input_channels, output_channels,
      weights.data(), nullptr, -1.0f, 1.0f, XNN_FLAG_INLINE_LHS_PACKING,
      nullptr, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);
  if (op->gemm_config->arch != xnn_arch_arm_sme2) {
    GTEST_SKIP();
  }
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> pool(
      pthreadpool_create(4), pthreadpool_destroy);
  ASSERT_NE(nullptr, pool.get());

  struct WorkerCase {
    const char* sme;
    const char* hybrid;
    bool queue;
    size_t sme_workers;
  };
  for (const WorkerCase& test_case : {
           WorkerCase{"0", "0", true, 0},
           WorkerCase{"0", "1", true, 0},
           WorkerCase{"1", "0", false, 4},
           WorkerCase{"2", "0", false, 4},
           WorkerCase{"1", "1", true, 1},
           WorkerCase{"2", "1", true, 2},
           WorkerCase{"4", "1", false, 4},
           WorkerCase{"8", "1", false, 4},
       }) {
    SCOPED_TRACE(std::string("SME=") + test_case.sme +
                 " HYBRID=" + test_case.hybrid);
    sme.Set(test_case.sme);
    hybrid.Set(test_case.hybrid);
    for (const size_t batch_size : {size_t{1}, size_t{127}, size_t{128}}) {
      size_t workspace_size = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_fully_connected_nc_pf32(
                    op, batch_size, &workspace_size, pool.get()));
      const bool has_queue =
          op->compute[op->num_compute_invocations - 1].type ==
          xnn_parallelization_type_f32_hybrid_queue;
      ASSERT_EQ(test_case.queue, has_queue);
      ASSERT_EQ(test_case.sme_workers,
                op->dynamic_context.gemm->gemm.f32_hybrid_sme_threads);
    }
  }
}

TEST(FULLY_CONNECTED_NC_PF32, neon_only_override) {
  ScopedF32EnvVar sme("XNN_F32_SME");
  ScopedF32EnvVar hybrid("XNN_ENABLE_F32_HYBRID");
  sme.Set("0");
  hybrid.Set("0");
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  constexpr size_t k = 37;
  constexpr size_t n = 257;
  constexpr size_t m_max = 131;
  constexpr size_t input_stride = 43;
  constexpr size_t output_stride = 263;
  std::vector<float> input(m_max * input_stride);
  std::vector<float> weights(n * k);
  std::vector<float> output(m_max * output_stride);
  for (size_t i = 0; i < input.size(); i++) input[i] = (i % 17) * 0.01f;
  for (size_t i = 0; i < weights.size(); i++) weights[i] = (i % 13) * 0.01f;

  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_fully_connected_nc_pf32(
                k, n, input_stride, output_stride, weights.data(), nullptr,
                -100.0f, 100.0f, XNN_FLAG_INLINE_LHS_PACKING, nullptr, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);
  if (op->gemm_config->arch != xnn_arch_arm_sme2) {
    GTEST_SKIP();
  }

  for (size_t threads : {size_t{1}, size_t{4}}) {
    std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> pool(
        pthreadpool_create(threads), pthreadpool_destroy);
    ASSERT_NE(nullptr, pool.get());
    for (size_t m : {size_t{1}, m_max}) {
      size_t workspace_size = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_fully_connected_nc_pf32(op, m, &workspace_size,
                                                    pool.get()));
      ASSERT_EQ(1, op->num_compute_invocations);
      ASSERT_EQ(xnn_parallelization_type_f32_hybrid_queue,
                op->compute[0].type);
      ASSERT_EQ(0, op->dynamic_context.gemm->gemm.f32_hybrid_sme_threads);
      ASSERT_EQ(0, workspace_size);
      ASSERT_EQ(xnn_status_success,
                xnn_setup_fully_connected_nc_pf32(op, input.data(),
                                                  output.data(), nullptr));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(op, pool.get()));
      for (size_t row = 0; row < m; row++) {
        for (size_t col = 0; col < n; col++) {
          float expected = 0.0f;
          for (size_t depth = 0; depth < k; depth++) {
            expected += input[row * input_stride + depth] *
                        weights[col * k + depth];
          }
          ASSERT_NEAR(expected, output[row * output_stride + col],
                      std::max(1.0e-4f * std::abs(expected), 1.0e-4f))
              << "at " << row << ", " << col;
        }
      }
    }
  }
}
#endif

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_kernel_zero_point) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(15)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_kernel_zero_point) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(15)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .kernel_zero_point(0)
      .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_fp32_weights) {
  FullyConnectedOperatorTester()
      .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_transpose_fp32_weights) {
  FullyConnectedOperatorTester()
      .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestF16();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmin(128)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmax(128)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .output_stride(29)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmin(128)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmax(128)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .output_stride(29)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC2W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQD8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F16_QC2W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .use_weights_cache(true)
      .TestQDU8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmin(128)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmax(128)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .output_stride(29)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QDU8_F32_QC2W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQDU8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmin(128)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmax(128)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .output_stride(29)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmin(128)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .qmax(128)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .output_stride(29)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(0)
      .use_weights_cache(true)
      .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QB4W, bl) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .batch_size(12)
          .output_channels(18)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQD8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QB4W, bl_no_bias) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .has_bias(false)
          .batch_size(12)
          .output_channels(18)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQD8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(8)
      .output_channels(4)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(4)
      .input_channels(1)
      .output_channels(16)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(11)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(8)
      .output_channels(4)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(4)
      .input_channels(1)
      .output_channels(16)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(11)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(8)
      .output_channels(4)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(4)
      .input_channels(2)
      .output_channels(16)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(11)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)
      .use_weights_cache(true)
      .kernel_zero_point(0)
      .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(64)
      .output_channels(19)
      .qmin(128)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(64)
      .output_channels(19)
      .qmax(128)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .input_stride(40)
      .output_channels(19)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .output_stride(29)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(64)
      .output_channels(19)
      .TestQP8F32QC2W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQP8F32QC2W();
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI && XNN_ENABLE_ARM_SME2
TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, config_registers_all_batch_kernels) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc2w_gemm_config();
  ASSERT_NE(nullptr, gemm_config);
  ASSERT_GT(gemm_config->mr, 1);
  EXPECT_NE(nullptr, gemm_config->minmax.qp8gemm[0].function[0]);
  EXPECT_NE(nullptr,
            gemm_config->minmax.qp8gemm[gemm_config->mr - 1].function[0]);
}
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_KLEIDIAI && XNN_ENABLE_ARM_SME2

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, transpose_weights_unsupported) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[output_channels * input_channels / 4] = {};
  float kernel_scale[output_channels];
  float kernel_zero_point[output_channels] = {};
  for (size_t i = 0; i < output_channels; i++) {
    kernel_scale[i] = 1.0f;
  }

  xnn_operator_t fully_connected_op = nullptr;
  EXPECT_EQ(
      xnn_status_unsupported_parameter,
      xnn_create_fully_connected_nc_qp8_f32_qc2w(
          input_channels, output_channels, input_channels, output_channels,
          kernel_zero_point, kernel_scale, kernel, /*bias=*/nullptr,
          /*output_min=*/-1.0f, /*output_max=*/1.0f,
          XNN_FLAG_TRANSPOSE_WEIGHTS, /*weights_cache=*/nullptr,
          &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W,
     input_channels_must_be_multiple_of_32) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t output_channels = 16;
  constexpr size_t max_input_channels = 33;
  uint8_t kernel[output_channels * ((max_input_channels + 3) / 4)] = {};
  float kernel_scale[output_channels];
  float kernel_zero_point[output_channels] = {};
  for (size_t i = 0; i < output_channels; i++) {
    kernel_scale[i] = 1.0f;
  }

  for (const size_t input_channels : {31, 33}) {
    xnn_operator_t fully_connected_op = nullptr;
    EXPECT_EQ(
        xnn_status_unsupported_parameter,
        xnn_create_fully_connected_nc_qp8_f32_qc2w(
            input_channels, output_channels, input_channels, output_channels,
            kernel_zero_point, kernel_scale, kernel, /*bias=*/nullptr,
            /*output_min=*/-1.0f, /*output_max=*/1.0f, /*flags=*/0,
            /*weights_cache=*/nullptr, &fully_connected_op));
    EXPECT_EQ(nullptr, fully_connected_op);
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, static_kernel_required) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  float kernel_scale[output_channels];
  float kernel_zero_point[output_channels] = {};
  for (size_t i = 0; i < output_channels; i++) {
    kernel_scale[i] = 1.0f;
  }

  xnn_operator_t fully_connected_op = nullptr;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qp8_f32_qc2w(
          input_channels, output_channels, input_channels, output_channels,
          kernel_zero_point, kernel_scale, /*kernel=*/nullptr, /*bias=*/nullptr,
          /*output_min=*/-1.0f, /*output_max=*/1.0f, /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, kernel_scale_required) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[output_channels * input_channels / 4] = {};
  float kernel_zero_point[output_channels] = {};

  xnn_operator_t fully_connected_op = nullptr;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qp8_f32_qc2w(
          input_channels, output_channels, input_channels, output_channels,
          kernel_zero_point, /*kernel_scale=*/nullptr, kernel, /*bias=*/nullptr,
          /*output_min=*/-1.0f, /*output_max=*/1.0f, /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W, null_kernel_zero_points) {
  if (xnn_init_qp8_f32_qc2w_gemm_config() == nullptr) {
    GTEST_SKIP() << "QP8 F32 QC2W is not available";
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[output_channels * input_channels / 4] = {};
  float kernel_scale[output_channels];
  std::fill_n(kernel_scale, output_channels, 1.0f);

  xnn_operator_t fully_connected_op = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_fully_connected_nc_qp8_f32_qc2w(
          input_channels, output_channels, input_channels, output_channels,
          /*kernel_zero_point=*/nullptr, kernel_scale, kernel,
          /*bias=*/nullptr, /*output_min=*/-1.0f, /*output_max=*/1.0f,
          /*flags=*/0, /*weights_cache=*/nullptr, &fully_connected_op));
  ASSERT_NE(nullptr, fully_connected_op);
  xnn_delete_operator(fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC2W,
     nonzero_kernel_zero_point_unsupported) {
  if (xnn_init_qp8_f32_qc2w_gemm_config() == nullptr) {
    GTEST_SKIP() << "QP8 F32 QC2W is not available";
  }
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  constexpr size_t input_channels = 32;
  constexpr size_t output_channels = 16;
  uint8_t kernel[output_channels * input_channels / 4] = {};
  float kernel_scale[output_channels];
  float kernel_zero_point[output_channels] = {};
  for (size_t i = 0; i < output_channels; i++) {
    kernel_scale[i] = 1.0f;
  }
  kernel_zero_point[output_channels / 2] = 1.0f;

  xnn_operator_t fully_connected_op = nullptr;
  EXPECT_EQ(
      xnn_status_unsupported_parameter,
      xnn_create_fully_connected_nc_qp8_f32_qc2w(
          input_channels, output_channels, input_channels, output_channels,
          kernel_zero_point, kernel_scale, kernel, /*bias=*/nullptr,
          /*output_min=*/-1.0f, /*output_max=*/1.0f, /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .use_weights_cache(true)
      .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(8)
      .use_weights_cache(true)
      .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QB4W, bl) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .batch_size(12)
          .output_channels(18)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQD8F16QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QB4W_F16_SCALES, scale_buffer_size_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  constexpr size_t input_channels = 32;
  constexpr size_t block_size = 32;
  constexpr size_t num_scale_elements = 33;
  // The byte-size multiplication wraps to a 64-byte allocation, while the
  // conversion loop still attempts to process output_channels scale values.
  constexpr size_t output_channels =
      std::numeric_limits<size_t>::max() / sizeof(xnn_bfloat16) +
      num_scale_elements;
  const std::array<uint16_t, num_scale_elements> f16_scales{};
  const std::array<xnn_float16, num_scale_elements> typed_f16_scales{};
  const uint8_t kernel = 0;
  xnn_operator_t fully_connected_op = nullptr;

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qd8_f16_qb4w_f16_scales(
          input_channels, output_channels, input_channels, output_channels,
          block_size, /*kernel_zero_point=*/8, f16_scales.data(), &kernel,
          /*bias=*/nullptr, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qp8_f32_qb4w_f16_scales(
          input_channels, output_channels, input_channels, output_channels,
          block_size, /*kernel_zero_point=*/8, f16_scales.data(), &kernel,
          /*bias=*/nullptr, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qd8_f32_qb4w_f16_scales(
          input_channels, output_channels, input_channels, output_channels,
          block_size, /*kernel_zero_point=*/8, typed_f16_scales.data(), &kernel,
          /*bias=*/nullptr, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_qdu8_f32_qb4w_f16_scales(
          input_channels, output_channels, input_channels, output_channels,
          block_size, /*kernel_zero_point=*/8, typed_f16_scales.data(), &kernel,
          /*bias=*/nullptr, -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_F16, static_bias_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t fully_connected_op = nullptr;
  const uint16_t kernel = 0;
  const float bias = 0.0f;
  const size_t overflow_channels = (SIZE_MAX / sizeof(uint16_t)) + 1;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_f16(
          /*input_channels=*/1, /*output_channels=*/overflow_channels,
          /*input_stride=*/1, /*output_stride=*/1, &kernel, &bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(),
          XNN_FLAG_FP32_STATIC_BIASES,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_PF16, static_bias_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t fully_connected_op = nullptr;
  const uint16_t kernel = 0;
  const float bias = 0.0f;
  const size_t overflow_channels = (SIZE_MAX / sizeof(uint16_t)) + 1;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_pf16(
          /*input_channels=*/1, /*output_channels=*/overflow_channels,
          /*input_stride=*/1, /*output_stride=*/1, &kernel, &bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(),
          XNN_FLAG_FP32_STATIC_BIASES,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_F32_F16, static_bias_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t fully_connected_op = nullptr;
  const uint16_t kernel = 0;
  const uint16_t bias = 0;
  const size_t overflow_channels = (SIZE_MAX / sizeof(float)) + 1;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_f32_f16(
          /*input_channels=*/0, /*output_channels=*/overflow_channels,
          /*input_stride=*/1, /*output_stride=*/1, &kernel, &bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(),
          /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_PF32_F16, static_bias_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t fully_connected_op = nullptr;
  const uint16_t kernel = 0;
  const uint16_t bias = 0;
  const size_t overflow_channels = (SIZE_MAX / sizeof(float)) + 1;
  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_create_fully_connected_nc_pf32_f16(
          /*input_channels=*/0, /*output_channels=*/overflow_channels,
          /*input_stride=*/1, /*output_stride=*/1, &kernel, &bias,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(),
          /*flags=*/0,
          /*weights_cache=*/nullptr, &fully_connected_op));
  EXPECT_EQ(nullptr, fully_connected_op);
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(8)
      .output_channels(4)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(4)
      .input_channels(2)
      .output_channels(16)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(11)
      .input_channels(22)
      .output_channels(19)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC2W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)
      .use_weights_cache(true)
      .TestQD8F16QC2W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QB4W, bl_no_bias) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .has_bias(false)
          .batch_size(12)
          .output_channels(18)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQD8F16QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, BL_BIAS) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .has_bias(true)
          .batch_size(1)
          .output_channels(16)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQP8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, BL_NO_BIAS) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .has_bias(false)
          .batch_size(1)
          .output_channels(16)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQP8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, BL_BIAS_TRANSPOSE_W) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .transpose_weights(true)
          .has_bias(true)
          .batch_size(1)
          .output_channels(16)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQP8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, BL_NO_BIAS_TRANSPOSE_W) {
  for (size_t ic = 32; ic <= 256; ic *= 2) {
    for (size_t bs = 32; bs <= ic; bs = bs * 2) {
      FullyConnectedOperatorTester()
          .transpose_weights(true)
          .has_bias(false)
          .batch_size(1)
          .output_channels(16)
          .input_channels(ic)
          .block_size(bs)
          .kernel_zero_point(8)
          .TestQP8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .block_size(32)
      .kernel_zero_point(8)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .block_size(32)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .output_channels(19)
      .block_size(32)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .block_size(32)
      .input_stride(38)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(32)
      .block_size(32)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(32)
      .block_size(32)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(32)
      .block_size(32)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(32)
      .block_size(32)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(32)
      .block_size(32)
      .input_stride(38)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QB4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(32)
      .block_size(32)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QB4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QC8W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy requires even number
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmin(128)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .qmax(128)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(29)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .output_stride(29)
      .TestQP8F32QC8W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(8)
      .use_weights_cache(true)
      .TestQP8F32QC8W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC8W,
     weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(20)  // legacy doesn't support odd nc
      .kernel_zero_point(8)
      .use_weights_cache(true)
      .TestQP8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, DISABLED_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_fp32_weights) {
  FullyConnectedOperatorTester()
      .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmin(128)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .qmax(128)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .input_stride(28)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .output_stride(29)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, DISABLED_small_batch_transpose_fp32_weights) {
  FullyConnectedOperatorTester()
      .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, small_batch_without_bias) {
  FullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(22)
      .output_channels(19)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_BF16_F32,
     DISABLED_weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .use_weights_cache(true)
      .TestBF16F32();
}

TEST(FULLY_CONNECTED_NC_F32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  std::vector<float> kernel(100);
  std::vector<float> bias(10);
  ASSERT_EQ(xnn_status_success,
            xnn_create_fully_connected_nc_f32(
                /*input_channels=*/10, /*output_channels=*/10,
                /*input_stride=*/SIZE_MAX / 2, /*output_stride=*/10,
                kernel.data(), bias.data(),
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, /*weights_cache=*/nullptr, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_fully_connected_nc_f32(
                op, /*batch_size=*/1, /*threadpool=*/nullptr));
}

TEST(FULLY_CONNECTED_NC_F32, overflow_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  std::vector<float> kernel(100);
  std::vector<float> bias(10);
  ASSERT_EQ(xnn_status_success,
            xnn_create_fully_connected_nc_f32(
                /*input_channels=*/10, /*output_channels=*/10,
                /*input_stride=*/10, /*output_stride=*/SIZE_MAX / 2,
                kernel.data(), bias.data(),
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, /*weights_cache=*/nullptr, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_fully_connected_nc_f32(
                op, /*batch_size=*/1, /*threadpool=*/nullptr));
}

TEST(FULLY_CONNECTED_NC_F32, overflow_batch_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  std::vector<float> kernel(100);
  std::vector<float> bias(10);
  ASSERT_EQ(xnn_status_success,
            xnn_create_fully_connected_nc_f32(
                /*input_channels=*/10, /*output_channels=*/10,
                /*input_stride=*/100, /*output_stride=*/100,
                kernel.data(), bias.data(),
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, /*weights_cache=*/nullptr, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_fully_connected_nc_f32(
                op, /*batch_size=*/SIZE_MAX / 50, /*threadpool=*/nullptr));
}

TEST(FULLY_CONNECTED_NC_F32, overflow_create_weights_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  std::vector<float> kernel(10);
  std::vector<float> bias(10);
  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_create_fully_connected_nc_f32(
                /*input_channels=*/SIZE_MAX / 2, /*output_channels=*/10,
                /*input_stride=*/SIZE_MAX / 2, /*output_stride=*/10,
                kernel.data(), bias.data(),
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, /*weights_cache=*/nullptr, &op));
}

