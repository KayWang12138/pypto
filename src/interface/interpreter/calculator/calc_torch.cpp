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

const char *Model() {
    return "torch";
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

#define DEFINE_BINARY_PAIR_OPS(Name, bop)                                                              \
    void Pair##Name(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other) { \
        auto big = self, small = other;                                                                \
        if (self->GetShape() < other->GetShape()) {                                                    \
            big = other, small = self;                                                                 \
        }                                                                                              \
        auto tout = From(out);                                                                         \
        std::vector<int64_t> offset(self->GetShape().size(), 0);                                       \
        auto tbig = View(tout, ToShape64(big->GetShape()), offset);                                    \
        tbig.copy_(From(big));                                                                         \
        auto tsmall = View(tout, ToShape64(small->GetShape()), offset);                                \
        torch::bop(tsmall, tsmall, From(small));                                                       \
    }

DEFINE_BINARY_PAIR_OPS(Sum, add_out)
DEFINE_BINARY_PAIR_OPS(Max, max_out)
DEFINE_BINARY_PAIR_OPS(Min, min_out)

std::vector<int64_t> GenAxesForTranspose(const int64_t offset, const std::vector<int64_t>& base) {
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < offset; i++) {
        axes.push_back(i);
    }
    for (auto x : base) {
        axes.push_back(x + offset);
    }
    return axes;
}

void FormatND2NZ(LogicalTensorDataPtr inputTensor) {
    auto inputData = From(inputTensor);
    auto oriShape = inputData.sizes();
    constexpr int minAxes = 2;
    if (oriShape.size() < minAxes) {
        return;
    }
    int64_t oriM = oriShape[oriShape.size() - minAxes];
    int64_t oriN = oriShape.back();

    std::vector<int64_t> oriBatch(oriShape.begin(), oriShape.end() - minAxes);
    int64_t batchNum = oriBatch.size();

    int64_t m0 = 16;
    constexpr int NZ_BLOCK_SIZE = 32;
    auto dtype = inputData.scalar_type();
    ASSERT(c10::elementSize(dtype) != 0);
    int64_t n0 = (dtype == at::ScalarType::Int) ? m0 : NZ_BLOCK_SIZE / c10::elementSize(dtype);
    ASSERT(n0 != 0);
    int64_t m1 = (oriM + m0 - 1) / m0;
    int64_t n1 = (oriN + n0 - 1) / n0;
    int64_t paddingM = m1 * m0 - oriM;
    int64_t paddingN = n1 * n0 - oriN;

    // Prepare padding vector
    std::vector<int64_t> padWidth;
    for (int64_t i = 0; i < batchNum; i++) {
        padWidth.push_back(0);
        padWidth.push_back(0);
    }
    padWidth.push_back(0);
    padWidth.push_back(paddingM);
    padWidth.push_back(0);
    padWidth.push_back(paddingN);

    auto paddedData = torch::constant_pad_nd(inputData, padWidth, 0);

    // Reshape and transpose
    std::vector<int64_t> newShape = oriBatch;
    newShape.push_back(m1);
    newShape.push_back(m0);
    newShape.push_back(n1);
    newShape.push_back(n0);

    auto reshaped = paddedData.reshape(newShape);
    std::vector<int64_t> axisToNZ = {2, 0, 1, 3};
    auto axes = GenAxesForTranspose(batchNum, axisToNZ);

    auto transposed = reshaped.permute(axes);
    auto restored = transposed.reshape({m1 * m0, n1 * n0});

    inputData.copy_(restored);
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

void MatMul(LogicalTensorDataPtr out, LogicalTensorDataPtr self, LogicalTensorDataPtr other, LogicalTensorDataPtr acc,
            MatMulSetParam &param) {
    auto tensorOut = From(out);
    auto outDtype = tensorOut.scalar_type();
    if (outDtype == at::ScalarType::Half) {
        outDtype = at::ScalarType::Float;
    }
    auto tensorSelf = From(self);
    auto tensorOther = From(other);
    if (acc) {
        tensorOut.copy_(From(acc));
    } else {
        tensorOut.zero_();
    }
    if (param.aTrans) {
        tensorSelf.transpose_(-1, AXIS_TO_LAST);
    }
    if (param.bTrans) {
        tensorOther.transpose_(-1, AXIS_TO_LAST);
    }
    if (tensorSelf.scalar_type() != outDtype) {
        tensorSelf = tensorSelf.to(outDtype);
    }
    if (tensorOther.scalar_type() != outDtype) {
        tensorOther = tensorOther.to(outDtype);
    }
    if (!param.kStep || param.kStep == self->GetShape(-1)) {
        tensorOut.add_(torch::matmul(tensorSelf, tensorOther));
    } else {
        MatmulSplitK(tensorOut, tensorSelf, tensorOther, param.kStep);
    }
    if (tensorOut.scalar_type() == at::ScalarType::Half) {
        tensorOut = tensorOut.to(at::ScalarType::Half);
    }
}

void ExpandS(LogicalTensorDataPtr out, const Element &elem) {
    auto tout = From(out);
    torch::full_out(tout, ToShape64(out->GetShape()), From(elem));
}

void Expand(LogicalTensorDataPtr out, LogicalTensorDataPtr self) {
    auto tself = From(self);
    if (self->GetShape(-1) != out->GetShape(-1) && self->GetShape(-1) != 1) {
        // possible block align
        tself = tself.slice(tself.dim() - 1, 0, 1);
    }
    From(out) = tself;
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