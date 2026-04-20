#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# reduce 类 case 批量集成测试包装脚本。
# 复用 test-integration.sh，只负责提供默认 CASES。
#
# 用法:
#   bash integration/benchmark/scripts/local/test-integration-reduce.sh
#   FULL=0 bash integration/benchmark/scripts/local/test-integration-reduce.sh
#   BENCHMARK_LOG_DIR=/tmp/bench_reduce bash integration/benchmark/scripts/local/test-integration-reduce.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CASES="${CASES:-47_Sum_reduction_over_a_dimension,48_Mean_reduction_over_a_dimension,49_Max_reduction_over_a_dimension,51_Argmax_over_a_dimension,52_Argmin_over_a_dimension,53_Min_reduction_over_a_dimension,89_cumsum}"
export DEVICE="${DEVICE:-6}"
export PYPTO_TIMEOUT="${PYPTO_TIMEOUT:-7200}"

exec bash "${SCRIPT_DIR}/test-integration.sh"
