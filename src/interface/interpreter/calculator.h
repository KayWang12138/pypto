/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file calculator.h
 * \brief
 */
/*for flow Verify Tool */

#pragma once

#include <cfenv>
#include "interface/interpreter/thread_pool.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/inner/element.h"
#include "interface/operation/opcode.h"

namespace npu::tile_fwk {

#define CONFIG_PROC 64

template <typename T>
inline T Exp(T v) {
    return static_cast<T>(std::exp(static_cast<double>(v)));
}

template <typename T>
inline T Sqrt(T v) {
    return static_cast<T>(std::sqrt(static_cast<double>(v)));
}

inline int64_t Abs(int64_t v) {
    return std::abs(v);
}
inline uint64_t Abs(uint64_t v) {
    return v;
}
inline double Abs(double v) {
    return std::abs(v);
}

inline std::vector<int> ShapeToStride(const std::vector<int> &shape) {
    std::vector<int> stride(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    return stride;
}

template <typename DstType, typename SrcType, typename EnableDstType = void, typename EnableSrcType = void>
struct CalcCastByType {
    static inline DstType Call(SrcType t, CastMode castMode) {
        (void)castMode;
        return static_cast<DstType>(t);
    }
};

template <typename DstType, typename SrcType>
struct CalcCastByType<DstType, SrcType, typename std::enable_if<std::is_integral<DstType>::value>::type,
    typename std::enable_if<std::is_floating_point<SrcType>::value || std::is_same<SrcType, npu::tile_fwk::float16>::value ||
                            std::is_same<SrcType, npu::tile_fwk::bfloat16>::value>::type> {
    static inline float CalcCastToFloat(float v) { return v; }
    static inline double CalcCastToFloat(double v) { return v; }
    static inline float CalcCastToFloat(npu::tile_fwk::bfloat16 v) { return v; }
    static inline float CalcCastToFloat(npu::tile_fwk::float16 v) { return v; }

    static inline DstType Call(SrcType v, CastMode castMode) {
        DstType res;
        auto fv = CalcCastToFloat(v);
        switch (castMode) {
            case CastMode::CAST_NONE: res = static_cast<DstType>(std::llrint(fv)); break;
            case CastMode::CAST_RINT: res = static_cast<DstType>(std::llrint(fv)); break;
            case CastMode::CAST_ROUND: res = static_cast<DstType>(std::llround(fv)); break;
            case CastMode::CAST_FLOOR: res = static_cast<DstType>(std::floor(fv)); break;
            case CastMode::CAST_CEIL: res = static_cast<DstType>(std::ceil(fv)); break;
            case CastMode::CAST_TRUNC: res = static_cast<DstType>(std::trunc(fv)); break;
            case CastMode::CAST_ODD: ASSERT(false); break;
            default: ASSERT(false); break;
        }
        return res;
    }
};

struct CalcCastContext {
    CalcCastContext() {}
    CalcCastContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, CastMode castMode_, int indexBegin_,
        int indexEnd_)
        : ret(ret_), oper(oper_), castMode(castMode_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    CastMode castMode{CastMode::CAST_NONE};
    int indexBegin{0};
    int indexEnd{0};
};
template <typename DstDataType, typename SrcDataType>
struct CalcCastHandler {
    static void Entry(void *c) {
        auto [ret, oper, castMode, indexBegin, indexEnd] = *(CalcCastContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            auto src = oper->Get<SrcDataType>(i);
            auto dst = CalcCastByType<DstDataType, SrcDataType>::Call(src, castMode);
            ret->Get<DstDataType>(i) = dst;
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, DataType destType, CastMode castMode,
        util::ThreadPool *pool) {
        std::fesetround(FE_TONEAREST);
        ASSERT(ret->GetShape() == oper->GetShape());
        ASSERT(ret->GetDataType() == destType);

        std::vector<CalcCastContext> contextList;
        int count = (oper->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, oper, castMode, count * i, std::min(count * (i + 1), oper->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcVecDupContext {
    CalcVecDupContext() {}
    CalcVecDupContext(LogicalTensorData *ret_, const Element *oper_, int indexBegin_, int indexEnd_)
        : ret(ret_), oper(oper_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const Element *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcVecDupHandler {
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd] = *(CalcVecDupContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            ret->Get<DataType>(i) = oper->Cast<DataType>();
        }
    }
    static void Calc(LogicalTensorData *ret, const Element *oper, util::ThreadPool *pool) {
        std::vector<CalcVecDupContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, oper, count * i, std::min(count * (i + 1), ret->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcPermutateContext {
    CalcPermutateContext() {}
    CalcPermutateContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, int indexBegin_, int indexEnd_)
        : ret(ret_), oper(oper_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcPermutate {
    static_assert(opcode == Opcode::OP_RESHAPE, "invalid opcode");
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd] = *(CalcPermutateContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            auto operValue = oper->Get<DataType>(i);
            switch (opcode) {
                case Opcode::OP_RESHAPE: ret->Get<DataType>(i) = static_cast<DataType>(operValue); break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        ASSERT(ret->GetSize() == oper->GetSize());
        ASSERT(ret->GetDataType() == oper->GetDataType());

        std::vector<CalcPermutateContext> contextList;
        int count = (oper->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, oper, count * i, std::min(count * (i + 1), ret->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcUnaryContext {
    CalcUnaryContext() {}
    CalcUnaryContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, int indexBegin_, int indexEnd_)
        : ret(ret_), oper(oper_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcUnary {
    static_assert(opcode == Opcode::OP_EXP || opcode == Opcode::OP_SQRT || opcode == Opcode::OP_ABS ||
                      opcode == Opcode::OP_COPY_IN,
        "invalid opcode");
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd] = *(CalcUnaryContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            auto operValue = oper->Get<DataType>(i);
            switch (opcode) {
                case Opcode::OP_EXP:
                    ret->Get<DataType>(i) = static_cast<DataType>(Exp(static_cast<CalcType>(operValue)));
                    break;
                case Opcode::OP_SQRT:
                    ret->Get<DataType>(i) = static_cast<DataType>(Sqrt(static_cast<CalcType>(operValue)));
                    break;
                case Opcode::OP_ABS:
                    ret->Get<DataType>(i) = static_cast<DataType>(Abs(static_cast<CalcType>(operValue)));
                    break;
                case Opcode::OP_COPY_IN: ret->Get<DataType>(i) = operValue; break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        ASSERT(ret->GetShape() == oper->GetShape());
        ASSERT(ret->GetDataType() == oper->GetDataType());

        std::vector<CalcUnaryContext> contextList;
        int count = (oper->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, oper, count * i, std::min(count * (i + 1), oper->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcBinaryContext {
    CalcBinaryContext() {}
    CalcBinaryContext(LogicalTensorData *ret_, const LogicalTensorData *lhs_, const LogicalTensorData *rhs_,
        int indexBegin_, int indexEnd_)
        : ret(ret_), lhs(lhs_), rhs(rhs_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *lhs{nullptr};
    const LogicalTensorData *rhs{nullptr};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcBinary {
    static_assert(opcode == Opcode::OP_ADD || opcode == Opcode::OP_S_ADD || opcode == Opcode::OP_SUB ||
                      opcode == Opcode::OP_S_SUB || opcode == Opcode::OP_MUL || opcode == Opcode::OP_S_MUL ||
                      opcode == Opcode::OP_DIV || opcode == Opcode::OP_S_DIV || opcode == Opcode::OP_S_MAX ||
                      opcode == Opcode::OP_S_MIN || opcode == Opcode::OP_MAXIMUM || opcode == Opcode::OP_PAIRMAX ||
                      opcode == Opcode::OP_PAIRSUM || opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC ||
                      opcode == Opcode::OP_MUL_BRC || opcode == Opcode::OP_DIV_BRC,
        "invalid opcode");
    static void Entry(void *c) {
        auto [ret, lhs, rhs, indexBegin, indexEnd] = *(CalcBinaryContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            auto lhsValue = lhs->Get<DataType>(i);
            auto rhsValue = rhs->Get<DataType>(i);
            CalcType retValue{0};
            switch (opcode) {
                case Opcode::OP_ADD:
                case Opcode::OP_ADD_BRC:
                case Opcode::OP_PAIRSUM:
                case Opcode::OP_S_ADD:
                    retValue = static_cast<CalcType>(lhsValue) + static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_SUB:
                case Opcode::OP_SUB_BRC:
                case Opcode::OP_S_SUB:
                    retValue = static_cast<CalcType>(lhsValue) - static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_MUL:
                case Opcode::OP_MUL_BRC:
                case Opcode::OP_S_MUL:
                    retValue = static_cast<CalcType>(lhsValue) * static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_DIV:
                case Opcode::OP_DIV_BRC:
                case Opcode::OP_S_DIV:
                    retValue = static_cast<CalcType>(lhsValue) / static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_MAXIMUM:
                case Opcode::OP_PAIRMAX:
                case Opcode::OP_S_MAX:
                    retValue = std::max(static_cast<CalcType>(lhsValue), static_cast<CalcType>(rhsValue));
                    break;
                case Opcode::OP_S_MIN:
                    retValue = std::min(static_cast<CalcType>(lhsValue), static_cast<CalcType>(rhsValue));
                    break;
                default: ASSERT(false); break;
            }
            ret->Get<DataType>(i) = static_cast<DataType>(retValue);
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        ASSERT(ret->GetShape() == lhs->GetShape());
        ASSERT(ret->GetDataType() == lhs->GetDataType());
        ASSERT(ret->GetShape() == rhs->GetShape());
        ASSERT(ret->GetDataType() == rhs->GetDataType());

        std::vector<CalcBinaryContext> contextList;
        int count = (lhs->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, lhs, rhs, count * i, std::min(count * (i + 1), lhs->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcBinaryScalarContext {
    CalcBinaryScalarContext() {}
    CalcBinaryScalarContext(LogicalTensorData *ret_, const LogicalTensorData *lhs_,
        const Element *rhs_, bool reverse_, int indexBegin_, int indexEnd_)
        : ret(ret_), lhs(lhs_), rhs(rhs_), reverse(reverse_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *lhs{nullptr};
    const Element *rhs{nullptr};
    bool reverse{false};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcBinaryScalar {
    static_assert(opcode == Opcode::OP_ADDS || opcode == Opcode::OP_S_ADDS || opcode == Opcode::OP_SUBS ||
                      opcode == Opcode::OP_S_SUBS || opcode == Opcode::OP_MULS || opcode == Opcode::OP_S_MULS ||
                      opcode == Opcode::OP_DIVS || opcode == Opcode::OP_S_DIVS || opcode == Opcode::OP_S_MAXS ||
                      opcode == Opcode::OP_S_MINS,
        "invalid opcode");
    static void Entry(void *c) {
        auto [ret, lhs, rhs, reverse, indexBegin, indexEnd] = *(CalcBinaryScalarContext *)c;
        auto rhsValue = rhs->Cast<DataType>();
        for (int i = indexBegin; i < indexEnd; i++) {
            auto lhsValue = lhs->Get<DataType>(i);
            CalcType retValue{0};
            switch (opcode) {
                case Opcode::OP_ADDS:
                case Opcode::OP_S_ADDS:
                    retValue = static_cast<CalcType>(lhsValue) + static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_SUBS:
                case Opcode::OP_S_SUBS:
                    if (!reverse) {
                        retValue = static_cast<CalcType>(lhsValue) - static_cast<CalcType>(rhsValue);
                    } else {
                        retValue = static_cast<CalcType>(rhsValue) - static_cast<CalcType>(lhsValue);
                    }
                    break;
                case Opcode::OP_MULS:
                case Opcode::OP_S_MULS:
                    retValue = static_cast<CalcType>(lhsValue) * static_cast<CalcType>(rhsValue);
                    break;
                case Opcode::OP_DIVS:
                case Opcode::OP_S_DIVS:
                    if (!reverse) {
                        retValue = static_cast<CalcType>(lhsValue) / static_cast<CalcType>(rhsValue);
                    } else {
                        retValue = static_cast<CalcType>(rhsValue) / static_cast<CalcType>(lhsValue);
                    }
                    break;
                case Opcode::OP_S_MAXS:
                    if (!reverse) {
                        retValue = std::max(static_cast<CalcType>(lhsValue), static_cast<CalcType>(rhsValue));
                    } else {
                        retValue = std::max(static_cast<CalcType>(rhsValue), static_cast<CalcType>(lhsValue));
                    }
                    break;
                case Opcode::OP_S_MINS:
                    if (!reverse) {
                        retValue = std::min(static_cast<CalcType>(lhsValue), static_cast<CalcType>(rhsValue));
                    } else {
                        retValue = std::min(static_cast<CalcType>(rhsValue), static_cast<CalcType>(lhsValue));
                    }
                    break;
                default: ASSERT(false); break;
            }
            ret->Get<DataType>(i) = static_cast<DataType>(retValue);
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        ASSERT(ret->GetDataType() == lhs->GetDataType());
        ASSERT(ret->GetShape() == lhs->GetShape());
        ASSERT(ret->GetDataType() == rhs->GetDataType());

        std::vector<CalcBinaryScalarContext> contextList;
        int count = (lhs->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, lhs, rhs, reverse, count * i, std::min(count * (i + 1), lhs->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcReduceContext {
    CalcReduceContext() {}
    CalcReduceContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, int indexBegin_, int indexEnd_,
        int axis_, const std::vector<int> &retStride_, const std::vector<int> &operStride_)
        : ret(ret_),
          oper(oper_),
          indexBegin(indexBegin_),
          indexEnd(indexEnd_),
          axis(axis_),
          retStride(retStride_),
          operStride(operStride_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
    int axis{-1};
    std::vector<int> retStride;
    std::vector<int> operStride;
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcReduce {
    static_assert(
        opcode == Opcode::OP_ROWSUMLINE || opcode == Opcode::OP_ROWMAX_SINGLE || opcode == Opcode::OP_ROWSUM_SINGLE,
        "invalid opcode");
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd, axis, retStride, operStride] = *(CalcReduceContext *)c;
        int rowSize = oper->GetShape()[axis];
        for (int i = indexBegin; i < indexEnd; i++) {
            /*
             *  For oper shape <3 x 4 x 5>, oper stride [20, 5, 1]
             *      When axis == 0:
             *          return shape: <1 x 4 x 5>, returnStride: [20, 5, 1]
             *      When axis == 1:
             *          return shape: <3 x 1 x 5>, returnStride: [5, 5, 1]
             */
            int operHighDimIndex = axis == 0 ? 0 : (i / retStride[axis - 1] * operStride[axis - 1]);
            int operLowDimIndex = i % retStride[axis];

            int operBeginIndex = operHighDimIndex + operLowDimIndex;
            int operStep = operStride[axis];
            switch (opcode) {
                case Opcode::OP_ROWSUMLINE: {
                    CalcType val = static_cast<CalcType>(oper->Get<DataType>(operBeginIndex));
                    for (int k = 1; k < rowSize; k++) {
                        val = val + static_cast<CalcType>(oper->Get<DataType>(operBeginIndex + operStep * k));
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(val);
                } break;
                case Opcode::OP_ROWMAX_SINGLE: {
                    CalcType val = static_cast<CalcType>(oper->Get<DataType>(operBeginIndex));
                    for (int k = 1; k < rowSize; k++) {
                        val = std::max(val, static_cast<CalcType>(oper->Get<DataType>(operBeginIndex + operStep * k)));
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(val);
                } break;
                case Opcode::OP_ROWSUM_SINGLE: {
                    CalcType val = static_cast<CalcType>(oper->Get<DataType>(operBeginIndex));
                    for (int k = 1; k < rowSize; k++) {
                        val = val + static_cast<CalcType>(oper->Get<DataType>(operBeginIndex + operStep * k));
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(val);
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        auto retShape = oper->GetShape();
        retShape[axis] = 1;
        ASSERT(ret->GetDataType() == oper->GetDataType());
        ASSERT(ret->GetShape() == retShape);

        std::vector<int> retStride = ShapeToStride(retShape);
        std::vector<int> operStride = ShapeToStride(oper->GetShape());
        ASSERT(retStride[axis] == operStride[axis]);

        std::vector<CalcReduceContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(
                ret, oper, count * i, std::min(count * (i + 1), ret->GetSize()), axis, retStride, operStride);
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcBroadcastContext {
    CalcBroadcastContext() {}
    CalcBroadcastContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, int indexBegin_, int indexEnd_,
        int axis_, const std::vector<int> &retStride_, const std::vector<int> &operStride_)
        : ret(ret_),
          oper(oper_),
          indexBegin(indexBegin_),
          indexEnd(indexEnd_),
          axis(axis_),
          retStride(retStride_),
          operStride(operStride_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
    int axis{-1};
    std::vector<int> retStride;
    std::vector<int> operStride;
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcBroadcast {
    static_assert(opcode == Opcode::OP_EXPAND, "invalid opcode");
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd, axis, retStride, operStride] = *(CalcBroadcastContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            /*
             *  For return shape <3 x 4 x 5>, return stride [20, 5, 1]
             *      When axis == 0:
             *          oper shape: <1 x 4 x 5>, operStride: [20, 5, 1]
             *      When axis == 1:
             *          oper shape: <3 x 1 x 5>, operStride: [5, 5, 1]
             */
            int operHighDimIndex = axis == 0 ? 0 : (i / retStride[axis - 1] * operStride[axis - 1]);
            int operLowDimIndex = i % retStride[axis];

            int operIndex = operHighDimIndex + operLowDimIndex;
            switch (opcode) {
                case Opcode::OP_EXPAND: {
                    DataType val = oper->Get<DataType>(operIndex);
                    ret->Get<DataType>(i) = val;
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        auto retShape = oper->GetShape();
        ASSERT(retShape[axis] == 1);
        retShape[axis] = ret->GetShape()[axis];
        ASSERT(ret->GetDataType() == oper->GetDataType());
        ASSERT(ret->GetShape() == retShape);

        std::vector<int> retStride = ShapeToStride(retShape);
        std::vector<int> operStride = ShapeToStride(oper->GetShape());
        ASSERT(retStride[axis] == operStride[axis]);

        std::vector<CalcBroadcastContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(
                ret, oper, count * i, std::min(count * (i + 1), ret->GetSize()), axis, retStride, operStride);
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcTransposeAdjDimContext {
    CalcTransposeAdjDimContext() {}
    CalcTransposeAdjDimContext(LogicalTensorData *ret_, const LogicalTensorData *oper_, int indexBegin_,
        int indexEnd_, int axis_, const std::vector<int> &retStride_, const std::vector<int> &operStride_,
        const std::vector<int> &retShape_, const std::vector<int> &operShape_)
        : ret(ret_),
          oper(oper_),
          indexBegin(indexBegin_),
          indexEnd(indexEnd_),
          axis(axis_),
          retStride(retStride_),
          operStride(operStride_),
          retShape(retShape_),
          operShape(operShape_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *oper{nullptr};
    int indexBegin{0};
    int indexEnd{0};
    int axis{-1};
    std::vector<int> retStride;
    std::vector<int> operStride;
    std::vector<int> retShape;
    std::vector<int> operShape;
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcTransposeAdjDimHandler {
    static_assert(opcode == Opcode::OP_TRANSPOSE_MOVEOUT, "invalid opcode");
    static void Entry(void *c) {
        auto [ret, oper, indexBegin, indexEnd, axis, retStride, operStride, retShape, operShape] =
            *(CalcTransposeAdjDimContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            int operHighDimIndex = axis == 0 ? 0 : (i / retStride[axis - 1] * operStride[axis - 1]);
            int operLowDimIndex = i % retStride[axis + 1];

            int retMidDimIndex = (i - operHighDimIndex - operLowDimIndex) / operStride[axis + 1];
            int retMidRowIndex = retMidDimIndex / retShape[axis + 1];
            int retMidColIndex = retMidDimIndex % retShape[axis + 1];
            int operMidDimIndex = retMidColIndex * operShape[axis + 1] + retMidRowIndex;
            int operIndex = operHighDimIndex + operMidDimIndex * operStride[axis + 1] + operLowDimIndex;
            switch (opcode) {
                case Opcode::OP_TRANSPOSE_MOVEOUT: {
                    DataType val = oper->Get<DataType>(operIndex);
                    ret->Get<DataType>(i) = val;
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        ASSERT(0 <= axis);
        ASSERT(axis + 1 < static_cast<int>(ret->GetShape().size()));
        auto retShape = oper->GetShape();
        std::swap(retShape[axis], retShape[axis + 1]);
        ASSERT(ret->GetDataType() == oper->GetDataType());
        ASSERT(ret->GetShape() == retShape);

        std::vector<int> retStride = ShapeToStride(retShape);
        std::vector<int> operStride = ShapeToStride(oper->GetShape());

        std::vector<CalcTransposeAdjDimContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, oper, count * i, std::min(count * (i + 1), ret->GetSize()), axis, retStride,
                operStride, ret->GetShape(), oper->GetShape());
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcIndexCopyContext {
    CalcIndexCopyContext() {}
    CalcIndexCopyContext(LogicalTensorData *ret_, const LogicalTensorData *src_, std::vector<int> *indexList_,
        const LogicalTensorData *dst_, int indexBegin_, int indexEnd_, int axis_, const std::vector<int> &srcStride_,
        const std::vector<int> &dstStride_, const std::vector<int> &srcShape_, const std::vector<int> &dstShape_)
        : ret(ret_),
          src(src_),
          indexList(indexList_),
          dst(dst_),
          indexBegin(indexBegin_),
          indexEnd(indexEnd_),
          axis(axis_),
          srcStride(srcStride_),
          dstStride(dstStride_),
          srcShape(srcShape_),
          dstShape(dstShape_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *src{nullptr};
    std::vector<int> *indexList;
    const LogicalTensorData *dst{nullptr};
    int indexBegin{0};
    int indexEnd{0};
    int axis{-1};
    std::vector<int> srcStride;
    std::vector<int> dstStride;
    std::vector<int> srcShape;
    std::vector<int> dstShape;
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcIndexCopyHandler {
    static_assert(opcode == Opcode::OP_INDEX_OUTCAST, "invalid opcode");
    static void Entry(void *c) {
        auto [ret, src, indexList, dst, indexBegin, indexEnd, axis, srcStride, dstStride, srcShape, dstShape] =
            *(CalcIndexCopyContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            int dstHighDimIndex = i / dstStride[axis - 1] * dstStride[axis - 1];
            int dstLowDimIndex = i % dstStride[axis];
            int dstMidDimIndex = (i - dstHighDimIndex - dstLowDimIndex) / dstStride[axis];
            switch (opcode) {
                case Opcode::OP_INDEX_OUTCAST: {
                    DataType val;
                    if (indexList->at(dstHighDimIndex / dstStride[axis - 1]) == dstMidDimIndex) {
                        int srcIndex = dstHighDimIndex / dstStride[axis - 1] * srcStride[axis - 1] + dstLowDimIndex;
                        val = src->Get<DataType>(srcIndex);
                    } else {
                        val = dst->Get<DataType>(i);
                    }
                    ret->Get<DataType>(i) = val;
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *src, const LogicalTensorData *index,
        const LogicalTensorData *dst, int axis, util::ThreadPool *pool) {
        ASSERT(0 <= axis);
        ASSERT(axis + 1 < static_cast<int>(ret->GetShape().size()));

        ASSERT(ret->GetDataType() == src->GetDataType());
        auto shape = src->GetShape();
        ASSERT(shape[axis] == 1);
        shape[axis] = dst->GetShape()[axis];
        ASSERT(ret->GetShape() == shape);

        ASSERT(ret->GetDataType() == dst->GetDataType());
        ASSERT(ret->GetShape() == dst->GetShape());

        for (int i = 0; i < axis; i++) {
            ASSERT(ret->GetShape()[i] == index->GetShape()[i]);
        }

        std::vector<int> srcStride = ShapeToStride(src->GetShape());
        std::vector<int> dstStride = ShapeToStride(dst->GetShape());

        std::vector<int> indexData(index->GetSize());
        for (int i = 0; i < index->GetSize(); i++) {
            indexData[i] = index->GetElement(i).Cast<int32_t>();
        }

        std::vector<CalcIndexCopyContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, src, &indexData, dst, count * i, std::min(count * (i + 1), ret->GetSize()),
                axis, srcStride, dstStride, src->GetShape(), dst->GetShape());
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcMatMulContext {
    CalcMatMulContext() {}
    CalcMatMulContext(LogicalTensorData *ret_, const LogicalTensorData *lhs_, const LogicalTensorData *rhs_,
        int kStep_, const LogicalTensorData *acc_, int indexBegin_, int indexEnd_)
        : ret(ret_), lhs(lhs_), rhs(rhs_), kStep(kStep_), acc(acc_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const LogicalTensorData *lhs{nullptr};
    const LogicalTensorData *rhs{nullptr};
    int kStep{0};
    const LogicalTensorData *acc{nullptr};
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcMatMulHandler {
    static_assert(opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B || opcode == Opcode::OP_A_MUL_BT ||
                      opcode == Opcode::OP_A_MULACC_BT,
        "invalid opcode");
    static void Entry(void *c) {
        auto [ret, lhs, rhs, kStep, acc, indexBegin, indexEnd] = *(CalcMatMulContext *)c;
        int kSize = lhs->GetShape()[1];
        for (int i = indexBegin; i < indexEnd; i++) {
            switch (opcode) {
                case Opcode::OP_A_MUL_B:
                case Opcode::OP_A_MULACC_B: {
                    int lhsRowIndex = i / rhs->GetShape()[1];
                    int rhsColIndex = i % rhs->GetShape()[1];
                    int rhsColSize = rhs->GetShape()[1];
                    DataType val{0};
                    for (int kIndex = 0; kIndex < (kSize + kStep - 1) / kStep; kIndex++) {
                        CalcType valAcc{0};
                        for (int k = kIndex * kStep; k < std::min(kIndex * kStep + kStep, kSize); k++) {
                            auto lhsValue = lhs->Get<DataType>(lhsRowIndex * kSize + k);
                            auto rhsValue = rhs->Get<DataType>(k * rhsColSize + rhsColIndex);
                            CalcType lhsCalcValue = static_cast<CalcType>(lhsValue);
                            CalcType rhsCalcValue = static_cast<CalcType>(rhsValue);
                            valAcc = valAcc + lhsCalcValue * rhsCalcValue;
                        }
                        DataType valAccData = static_cast<DataType>(valAcc);
                        val = static_cast<DataType>(static_cast<CalcType>(val) + static_cast<CalcType>(valAccData));
                    }
                    if (opcode == Opcode::OP_A_MULACC_B) {
                        auto accValue = acc->Get<DataType>(i);
                        val = static_cast<DataType>(static_cast<CalcType>(val) + static_cast<CalcType>(accValue));
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(val);
                } break;
                case Opcode::OP_A_MUL_BT:
                case Opcode::OP_A_MULACC_BT: {
                    int lhsRowIndex = i / rhs->GetShape()[0];
                    int rhsRowIndex = i % rhs->GetShape()[0];
                    DataType val{0};
                    for (int kIndex = 0; kIndex < (kSize + kStep - 1) / kStep; kIndex++) {
                        CalcType valAcc{0};
                        for (int k = kIndex * kStep; k < std::min(kIndex * kStep + kStep, kSize); k++) {
                            auto lhsValue = lhs->Get<DataType>(lhsRowIndex * kSize + k);
                            auto rhsValue = rhs->Get<DataType>(rhsRowIndex * kSize + k);
                            CalcType lhsCalcValue = static_cast<CalcType>(lhsValue);
                            CalcType rhsCalcValue = static_cast<CalcType>(rhsValue);
                            valAcc = valAcc + lhsCalcValue * rhsCalcValue;
                        }
                        DataType valAccData = static_cast<DataType>(valAcc);
                        val = static_cast<DataType>(static_cast<CalcType>(val) + static_cast<CalcType>(valAccData));
                    }
                    if (opcode == Opcode::OP_A_MULACC_BT) {
                        auto accValue = acc->Get<DataType>(i);
                        val = static_cast<DataType>(static_cast<CalcType>(val) + static_cast<CalcType>(accValue));
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(val);
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        int kStep, const LogicalTensorData *acc, util::ThreadPool *pool) {
        ASSERT(ret->GetDataType() == lhs->GetDataType());
        ASSERT(ret->GetDataType() == rhs->GetDataType());

        ASSERT(ret->GetShape().size() == 0x2);
        ASSERT(lhs->GetShape().size() == 0x2);
        ASSERT(rhs->GetShape().size() == 0x2);

        if (opcode == Opcode::OP_A_MUL_B || opcode == Opcode::OP_A_MULACC_B) {
            ASSERT(ret->GetShape()[0] == lhs->GetShape()[0]);
            ASSERT(ret->GetShape()[1] == rhs->GetShape()[1]);
            ASSERT(lhs->GetShape()[1] == rhs->GetShape()[0]);
        } else if (opcode == Opcode::OP_A_MUL_BT || opcode == Opcode::OP_A_MULACC_BT) {
            ASSERT(ret->GetShape()[0] == lhs->GetShape()[0]);
            ASSERT(ret->GetShape()[1] == rhs->GetShape()[0]);
            ASSERT(lhs->GetShape()[1] == rhs->GetShape()[1]);
        } else {
            ASSERT(false);
        }

        if (acc != nullptr) {
            ASSERT(ret->GetDataType() == acc->GetDataType());
            ASSERT(ret->GetShape() == acc->GetShape());
        }

        std::vector<CalcMatMulContext> contextList;
        int count = (ret->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, lhs, rhs, kStep, acc, count * i, std::min(count * (i + 1), ret->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

struct CalcReduceAccContext {
    CalcReduceAccContext() {}
    CalcReduceAccContext(LogicalTensorData *ret_, const std::vector<LogicalTensorData *> *operList_,
        int indexBegin_, int indexEnd_)
        : ret(ret_), operList(operList_), indexBegin(indexBegin_), indexEnd(indexEnd_) {}

    LogicalTensorData *ret{nullptr};
    const std::vector<LogicalTensorData *> *operList;
    int indexBegin{0};
    int indexEnd{0};
};
template <Opcode opcode, typename DataType, typename CalcType>
struct CalcReduceAccHandler {
    static_assert(opcode == Opcode::OP_REDUCE_ACC, "invalid opcode");
    static void Entry(void *c) {
        auto [ret, operList, indexBegin, indexEnd] = *(CalcReduceAccContext *)c;
        for (int i = indexBegin; i < indexEnd; i++) {
            switch (opcode) {
                case Opcode::OP_REDUCE_ACC: {
                    CalcType v{0};
                    for (size_t k = 0; k < operList->size(); k++) {
                        auto operValue = operList->at(k)->Get<DataType>(i);
                        v = v + static_cast<CalcType>(operValue);
                    }
                    ret->Get<DataType>(i) = static_cast<DataType>(v);
                } break;
                default: ASSERT(false); break;
            }
        }
    }
    static void Calc(
        LogicalTensorData *ret, const std::vector<LogicalTensorData *> *operList, util::ThreadPool *pool) {
        for (size_t k = 0; k < operList->size(); k++) {
            ASSERT(ret->GetShape() == operList->at(k)->GetShape());
            ASSERT(ret->GetDataType() == operList->at(k)->GetDataType());
        }

        std::vector<CalcReduceAccContext> contextList;
        int count = (operList->at(0)->GetSize() + pool->GetThreadCount() - 1) / pool->GetThreadCount();
        for (int i = 0; i < pool->GetThreadCount(); i++) {
            contextList.emplace_back(ret, operList, count * i, std::min(count * (i + 1), operList->at(0)->GetSize()));
        }

        for (size_t i = 0; i < contextList.size(); i++) {
            pool->SubmitTask(&contextList[i], Entry);
        }
        pool->NotifyAll();
        pool->WaitForAll();
    }
};

class Calculator {
private:
    template <template <Opcode, typename, typename> typename T, Opcode opcode, typename... TyArgs>
    static void HandleDataType(DataType currDataType, TyArgs &&...args) {
        switch (currDataType) {
            case DT_INT8: T<opcode, int8_t, int64_t>::Calc(args...); break;
            case DT_INT16: T<opcode, int16_t, int64_t>::Calc(args...); break;
            case DT_INT32: T<opcode, int32_t, int64_t>::Calc(args...); break;
            case DT_INT64: T<opcode, int64_t, int64_t>::Calc(args...); break;
            case DT_FP16: T<opcode, npu::tile_fwk::float16, double>::Calc(args...); break;
            case DT_FP32: T<opcode, float, double>::Calc(args...); break;
            case DT_BF16: T<opcode, npu::tile_fwk::bfloat16, double>::Calc(args...); break;
            case DT_UINT8: T<opcode, uint8_t, uint64_t>::Calc(args...); break;
            case DT_UINT16: T<opcode, uint16_t, uint64_t>::Calc(args...); break;
            case DT_UINT32: T<opcode, uint32_t, uint64_t>::Calc(args...); break;
            case DT_UINT64: T<opcode, uint64_t, uint64_t>::Calc(args...); break;
            case DT_DOUBLE: T<opcode, double, double>::Calc(args...); break;
            default: ASSERT(false);
        }
    }

    template <typename srcDataType, typename... TyArgs>
    static void HandleDstDataTypeCast(DataType retDataType, TyArgs &&...args) {
        switch (retDataType) {
            case DT_INT8: CalcCastHandler<int8_t, srcDataType>::Calc(args...); break;
            case DT_INT16: CalcCastHandler<int16_t, srcDataType>::Calc(args...); break;
            case DT_INT32: CalcCastHandler<int32_t, srcDataType>::Calc(args...); break;
            case DT_INT64: CalcCastHandler<int64_t, srcDataType>::Calc(args...); break;
            case DT_FP16: CalcCastHandler<npu::tile_fwk::float16, srcDataType>::Calc(args...); break;
            case DT_FP32: CalcCastHandler<float, srcDataType>::Calc(args...); break;
            case DT_BF16: CalcCastHandler<npu::tile_fwk::bfloat16, srcDataType>::Calc(args...); break;
            case DT_UINT8: CalcCastHandler<uint8_t, srcDataType>::Calc(args...); break;
            case DT_UINT16: CalcCastHandler<uint16_t, srcDataType>::Calc(args...); break;
            case DT_UINT32: CalcCastHandler<uint32_t, srcDataType>::Calc(args...); break;
            case DT_UINT64: CalcCastHandler<uint64_t, srcDataType>::Calc(args...); break;
            case DT_DOUBLE: CalcCastHandler<double, srcDataType>::Calc(args...); break;
            default: ASSERT(false);
        }
    }

    template <typename... TyArgs>
    static void HandleDataTypeCast(DataType operDataType, DataType retDataType, TyArgs &&...args) {
        switch (operDataType) {
            case DT_INT8: HandleDstDataTypeCast<int8_t>(retDataType, args...); break;
            case DT_INT16: HandleDstDataTypeCast<int16_t>(retDataType, args...); break;
            case DT_INT32: HandleDstDataTypeCast<int32_t>(retDataType, args...); break;
            case DT_INT64: HandleDstDataTypeCast<int64_t>(retDataType, args...); break;
            case DT_FP16: HandleDstDataTypeCast<npu::tile_fwk::float16>(retDataType, args...); break;
            case DT_FP32: HandleDstDataTypeCast<float>(retDataType, args...); break;
            case DT_BF16: HandleDstDataTypeCast<npu::tile_fwk::bfloat16>(retDataType, args...); break;
            case DT_UINT8: HandleDstDataTypeCast<uint8_t>(retDataType, args...); break;
            case DT_UINT16: HandleDstDataTypeCast<uint16_t>(retDataType, args...); break;
            case DT_UINT32: HandleDstDataTypeCast<uint32_t>(retDataType, args...); break;
            case DT_UINT64: HandleDstDataTypeCast<uint64_t>(retDataType, args...); break;
            case DT_DOUBLE: HandleDstDataTypeCast<double>(retDataType, args...); break;
            default: ASSERT(false);
        }
    }

public:
    static void CalcCast(LogicalTensorData *ret, const LogicalTensorData *oper, DataType retDataType,
        CastMode castMode, util::ThreadPool *pool) {
        auto operDataType = oper->GetDataType();
        HandleDataTypeCast(operDataType, retDataType, ret, oper, retDataType, castMode, pool);
    }

    static void CalcVecDup(LogicalTensorData *ret, const Element *oper, util::ThreadPool *pool) {
        HandleDataType<CalcVecDupHandler, Opcode::OP_VEC_DUP>(ret->GetDataType(), ret, oper, pool);
    }

    static void CalcReshape(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        HandleDataType<CalcPermutate, Opcode::OP_RESHAPE>(oper->GetDataType(), ret, oper, pool);
    }

    static void CalcCopy(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        HandleDataType<CalcUnary, Opcode::OP_COPY_IN>(oper->GetDataType(), ret, oper, pool);
    }

    static void CalcExp(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        HandleDataType<CalcUnary, Opcode::OP_EXP>(oper->GetDataType(), ret, oper, pool);
    }

    static void CalcSqrt(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        HandleDataType<CalcUnary, Opcode::OP_SQRT>(oper->GetDataType(), ret, oper, pool);
    }

    static void CalcAbs(LogicalTensorData *ret, const LogicalTensorData *oper, util::ThreadPool *pool) {
        HandleDataType<CalcUnary, Opcode::OP_ABS>(oper->GetDataType(), ret, oper, pool);
    }

    static void CalcAdd(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_ADD>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcAddBrc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_ADD_BRC>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcPairSum(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_PAIRSUM>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcSub(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_SUB>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }
    static void CalcSubBrc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_SUB_BRC>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcMul(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_MUL>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcMulBrc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_MUL_BRC>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcDiv(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_DIV>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcDivBrc(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_DIV_BRC>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcMax(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_S_MAX>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcPairMax(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_PAIRMAX>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcMin(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        util::ThreadPool *pool) {
        HandleDataType<CalcBinary, Opcode::OP_S_MIN>(lhs->GetDataType(), ret, lhs, rhs, pool);
    }

    static void CalcAddS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_ADDS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcSubS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_SUBS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcMulS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_MULS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcDivS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_DIVS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcMaxS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_S_MAXS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcMinS(LogicalTensorData *ret, const LogicalTensorData *lhs, const Element *rhs,
        bool reverse, util::ThreadPool *pool) {
        HandleDataType<CalcBinaryScalar, Opcode::OP_S_MINS>(lhs->GetDataType(), ret, lhs, rhs, reverse, pool);
    }

    static void CalcRowSumLine(
        LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcReduce, Opcode::OP_ROWSUMLINE>(oper->GetDataType(), ret, oper, axis, pool);
    }

    static void CalcRowSumSingle(
        LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcReduce, Opcode::OP_ROWSUM_SINGLE>(oper->GetDataType(), ret, oper, axis, pool);
    }

    static void CalcRowMaxSingle(
        LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcReduce, Opcode::OP_ROWMAX_SINGLE>(oper->GetDataType(), ret, oper, axis, pool);
    }

    static void CalcExpand(
        LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcBroadcast, Opcode::OP_EXPAND>(oper->GetDataType(), ret, oper, axis, pool);
    }

    static void CalcTransposeAdjDim(
        LogicalTensorData *ret, const LogicalTensorData *oper, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcTransposeAdjDimHandler, Opcode::OP_TRANSPOSE_MOVEOUT>(
            oper->GetDataType(), ret, oper, axis, pool);
    }

    static void CalcIndexCopy(LogicalTensorData *ret, const LogicalTensorData *src,
        const LogicalTensorData *index, const LogicalTensorData *dst, int axis, util::ThreadPool *pool) {
        HandleDataType<CalcIndexCopyHandler, Opcode::OP_INDEX_OUTCAST>(
            dst->GetDataType(), ret, src, index, dst, axis, pool);
    }

    static void CalcMatMul(LogicalTensorData *ret, const LogicalTensorData *lhs, const LogicalTensorData *rhs,
        int kStep, util::ThreadPool *pool) {
        HandleDataType<CalcMatMulHandler, Opcode::OP_A_MUL_B>(lhs->GetDataType(), ret, lhs, rhs, kStep, nullptr, pool);
    }
    static void CalcMatMulAcc(LogicalTensorData *ret, const LogicalTensorData *lhs,
        const LogicalTensorData *rhs, int kStep, const LogicalTensorData *acc, util::ThreadPool *pool) {
        HandleDataType<CalcMatMulHandler, Opcode::OP_A_MULACC_B>(lhs->GetDataType(), ret, lhs, rhs, kStep, acc, pool);
    }

    static void CalcMatMulTrans(LogicalTensorData *ret, const LogicalTensorData *lhs,
        const LogicalTensorData *rhs, int kStep, util::ThreadPool *pool) {
        HandleDataType<CalcMatMulHandler, Opcode::OP_A_MUL_BT>(lhs->GetDataType(), ret, lhs, rhs, kStep, nullptr, pool);
    }
    static void CalcMatMulTransAcc(LogicalTensorData *ret, const LogicalTensorData *lhs,
        const LogicalTensorData *rhs, int kStep, const LogicalTensorData *acc, util::ThreadPool *pool) {
        HandleDataType<CalcMatMulHandler, Opcode::OP_A_MULACC_BT>(lhs->GetDataType(), ret, lhs, rhs, kStep, acc, pool);
    }

    static void CalcReduceAcc(
        LogicalTensorData *ret, const std::vector<LogicalTensorData *> *operList, util::ThreadPool *pool) {
        HandleDataType<CalcReduceAccHandler, Opcode::OP_REDUCE_ACC>(
            operList->at(0)->GetDataType(), ret, operList, pool);
    }
};
} // namespace npu::tile_fwk
