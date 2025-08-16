/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/torch.h>

#include "../calc.h"

namespace npu::tile_fwk::calc {

#define AXIS_TO_LAST -2

static torch::ScalarType FromDataType(DataType t) {
    switch (t) {
        case DT_INT8: return torch::kInt8;
        case DT_INT16: return torch::kInt16;
        case DT_INT32: return torch::kInt32;
        case DT_INT64: return torch::kInt64;
        case DT_FP16: return torch::kFloat16;
        case DT_FP32: return torch::kFloat32;
        case DT_BF16: return torch::kBFloat16;
        /* unsigned int are limited supported, use signed types for temp */
        case DT_UINT8: return torch::kInt8;
        case DT_UINT16: return torch::kInt16;
        case DT_UINT32: return torch::kInt32;
        case DT_UINT64: return torch::kInt64;
        case DT_BOOL: return torch::kBool;
        case DT_DOUBLE: return torch::kDouble;
        case DT_INT4:
        case DT_FP8:
        case DT_HF4:
        case DT_HF8:
        default: assert(0);
    }
    return torch::ScalarType::Undefined;
}

static at::Scalar From(const Element &elem) {
    switch (elem.GetDataType()) {
        case DT_BOOL:
        case DT_INT4:
        case DT_INT8:
        case DT_INT16:
        case DT_INT32:
        case DT_INT64: return at::Scalar(elem.GetSignedData());
        case DT_FP16:
        case DT_FP32:
        case DT_BF16:
        case DT_DOUBLE: return at::Scalar(elem.GetFloatData());
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
            // lower version of pytorch not support uint64 type, use int64 for temp
            return at::Scalar(static_cast<int64_t>(elem.GetUnsignedData()));
        case DT_FP8:
        case DT_HF4:
        case DT_HF8:
        default: assert(0);
    }
    return at::Scalar();
}

static std::vector<int64_t> ToShape64(const std::vector<int> &shape) {
    std::vector<int64_t> ret;
    for (auto x : shape)
        ret.emplace_back(x);
    return ret;
}

static torch::Tensor From(LogicalTensorDataPtr data) {
    RawTensorDataPtr raw = data->GetData();
    auto tensor = torch::from_blob(raw->data(), ToShape64(raw->GetShape()), FromDataType(raw->GetDataType()));
    auto view = tensor.as_strided(ToShape64(data->GetShape()), raw->GetStride(), data->GetStorageOffset());
    if (data->IsAxisCombine())
        view = view.transpose_(-1, AXIS_TO_LAST);
    return view;
}

torch::Tensor View(const torch::Tensor &self, const std::vector<int64_t> &shape, const std::vector<int64_t> &offset) {
    int64_t storageOffset = self.storage_offset();
    for (size_t dim = 0; dim < offset.size(); dim++) {
        storageOffset += self.stride(dim) * offset[dim];
    }
    return self.as_strided(shape, self.strides(), storageOffset);
}

void Dump(std::ostream &os, LogicalTensorDataPtr self) {
    os << From(self);
}

bool AllClose(LogicalTensorDataPtr self, LogicalTensorDataPtr other, double atol, double rtol) {
    return From(self).allclose(From(other), atol, rtol);
}

void Exp(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::exp_out(tout, From(self));
}

void Sqrt(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::sqrt_out(tout, From(self));
}

void Abs(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tout = From(out);
    torch::abs_out(tout, From(self));
}

#define DEFINE_BINARY_S_OPS(Name, op_out)                                                                 \
    void Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &scalar, bool reverse) { \
        auto tout = From(out);                                                                            \
        if (reverse) {                                                                                    \
            torch::full_out(tout, ToShape64(out->GetShape()), From(scalar));                              \
            torch::op_out(tout, From(self), tout);                                                        \
        } else {                                                                                          \
            torch::op_out(tout, From(self), From(scalar));                                                \
        }                                                                                                 \
    }

DEFINE_BINARY_S_OPS(AddS, add_out)
DEFINE_BINARY_S_OPS(SubS, sub_out)
DEFINE_BINARY_S_OPS(MulS, mul_out)
DEFINE_BINARY_S_OPS(DivS, div_out)

void Add(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::add_out(tout, From(self), From(other));
}

void Sub(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::sub_out(tout, From(self), From(other));
}

void Mul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::mul_out(tout, From(self), From(other));
}

void Div(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::div_out(tout, From(self), From(other));
}

void Cast(LogicalTensorDataPtr out, LogicalTensorDataPtr self, CastMode mode) {
    (void)mode;
    From(out) = From(self);
}

void Min(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::min_out(tout, From(self), From(other));
}

void Max(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) {
    auto tout = From(out);
    torch::max_out(tout, From(self), From(other));
}

void MinS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    torch::clamp_max_out(tout, From(self), From(elem));
}

void MaxS(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const Element &elem) {
    auto tout = From(out);
    torch::clamp_min_out(tout, From(self), From(elem));
}

static void MatmulSplitK(torch::Tensor &out, const torch::Tensor &lhs, const torch::Tensor &rhs, int64_t kstep) {
    auto shapeL = lhs.sizes().vec();
    auto shapeR = rhs.sizes().vec();
    auto offsetL = std::vector<int64_t>(shapeL.size(), 0);
    auto offsetR = std::vector<int64_t>(shapeR.size(), 0);
    int64_t kdimL = shapeL.size() - 1;
    int64_t kdimR = shapeR.size() - 0x2;
    int64_t k = shapeL[kdimL];

    for (int64_t offset = 0; offset < k; offset += kstep) {
        shapeL[kdimL] = std::min(kstep, k - offset);
        shapeR[kdimR] = std::min(kstep, k - offset);
        offsetL[kdimL] = offset;
        offsetR[kdimR] = offset;
        auto viewL = View(lhs, shapeL, offsetL);
        auto viewR = View(rhs, shapeR, offsetR);
        out.add_(torch::matmul(viewL, viewR));
    }
}

void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, bool atrans, bool btrans,
    bool acc, int64_t kstep) {
    auto tout = From(out);
    auto tself = From(self);
    auto tother = From(other);

    if (atrans) {
        tself.transpose_(-1, AXIS_TO_LAST);
    }
    if (btrans) {
        tother.transpose_(-1, AXIS_TO_LAST);
    }
    if (!acc) {
        tout.zero_();
    }
    if (tself.scalar_type() != tout.scalar_type()) {
        tself = tself.to(tout.scalar_type());
    }
    if (tother.scalar_type() != tout.scalar_type()) {
        tother = tother.to(tout.scalar_type());
    }
    if (!kstep) {
        tout.add_(torch::matmul(tself, tother));
    } else {
        MatmulSplitK(tout, tself, tother, kstep);
    }
}

void ExpandS(LogicalTensorDataPtr out, const Element &elem) {
    auto tout = From(out);
    torch::full_out(tout, ToShape64(out->GetShape()), From(elem));
}

void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    From(out) = From(self);
}

void Copy(LogicalTensorDataPtr out, LogicalTensorDataPtr self, bool trans) {
    if (trans) {
        From(out) = From(self).transpose_(-1, AXIS_TO_LAST);
    } else {
        From(out) = From(self);
    }
}

void RowSumExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    From(out) = torch::sum(From(self), {dim}, true);
}

void RowSumSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto tout = From(out);
    torch::sum_out(tout, From(self), {dim}, true);
}

void RowMinExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::min(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

void RowMaxExpand(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::max(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

void RowMinSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::min(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

void RowMaxSingle(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int dim) {
    auto ret = torch::max(From(self), dim, true);
    From(out) = std::get<0>(ret);
}

void Reshape(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    From(out) = torch::reshape(From(self), ToShape64(out->GetShape()));
}

void Transpose(LogicalTensorDataPtr out, LogicalTensorDataPtr self, int64_t dim0, int64_t dim1) {
    auto tout = From(out);
    torch::transpose_copy_out(tout, From(self), dim0, dim1);
}

void Permute(LogicalTensorDataPtr out, LogicalTensorDataPtr self, const std::vector<int64_t> &dim) {
    auto tout = From(out);
    torch::permute_copy_out(tout, From(self), dim);
}

void ReduceAcc(LogicalTensorDataPtr out, const std::vector<LogicalTensorDataPtr> &tdatas) {
    auto tout = From(out);
    std::vector<torch::Tensor> tensors;
    for (auto &tdata : tdatas) {
        tensors.push_back(From(tdata));
    }
    torch::sum_out(tout, torch::stack(tensors, 0), 0);
}

} // namespace npu::tile_fwk::calc