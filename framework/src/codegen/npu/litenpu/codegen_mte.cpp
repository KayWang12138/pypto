// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file codegen_mte.cpp
//  * \brief
//  */

// #include "codegen_op_litenpu.h"
// #include "codegen/symbol_mgr/codegen_symbol.h"
// #include "codegen/utils/codegen_utils.h"
// #include "securec.h"

// namespace npu::tile_fwk {
// const std::string TSTORE_CONF = "TStoreConfig";

// std::string CodeGenOpLiteNPU::GenMemL1CopyIn() const {
//     return GenMemCopyCube(false, 0);
// }

// std::string CodeGenOpLiteNPU::GenMemL0CCopyOut() const {
//     return GenMemCopyCube(true, 0);
// }

// std::string CodeGenOpLiteNPU::GenMemCopyCube(bool isLocalToGM, unsigned uf) const {
//     unsigned gmIdx = isLocalToGM ? 0 : 1;
//     bool isSpillToGm = operand[gmIdx] == SYMBOL_STACK_BASE;
//     return GenMemCopyVar(isLocalToGM, isSpillToGm, uf);
// }

// std::string CodeGenOpLiteNPU::PrintMemL1ToL0TileTensor() const {
//     bool isTrans = false;
//     if ((opCode == Opcode::OP_L1_TO_L0_BT) || (opCode == Opcode::OP_L1_TO_L0_AT)) {
//         isTrans = true;
//     }
//     std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
//     std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));

//     auto dynOffset = offsetFromAttr[ToUnderlying(MISOIdx::SRC0_IDX)];
//     size_t coordSize = rawShape[ToUnderlying(MISOIdx::SRC0_IDX)].size();
//     std::vector<std::string> l0Offset;
//     if (!dynOffset.empty()) {
//         ASSERT(dynOffset.size() == SHAPE_DIM2 || dynOffset.size() == SHAPE_DIM3)
//             << "GenMemL1ToL0 only support 2-dim or 3-dim!";
//         for (auto &srcOffset : dynOffset) {
//             l0Offset.push_back(SymbolicExpressionTable::BuildExpression(srcOffset));
//         }
//     } else {
//         for (size_t i = 0; i < coordSize; ++i) {
//             l0Offset.push_back("0");
//         }
//     }
//     // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coordCp = WrapParamByParentheses(l0Offset);
//     // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coord = PrintCoord(coordSize, coordCp);
//     std::ostringstream oss;
//     oss << tileOpName;
//     if (opCode != Opcode::OP_L1_TO_L0A_SCALE && opCode != Opcode::OP_L1_TO_L0B_SCALE) {
//         oss << WrapParamByAngleBrackets({std::to_string(isTrans)});
//     }
//     oss << WrapParamByParentheses({dstTensor, src0Tensor, coord});
//     oss << STMT_END;
//     return oss.str();
// }

// std::string CodeGenOpLiteNPU::GenMemL1ToL0() const {
//     if (isSupportLayout) {
//         return PrintMemL1ToL0TileTensor();
//     }

//     return "";
// }

// std::string CodeGenOpLiteNPU::GenMemUBTransfer(bool isCopyUBToGM) const {
//     unsigned gmIdx = isCopyUBToGM ? 0 : 1;
//     bool isSpillToGm = operand[gmIdx] == SYMBOL_STACK_BASE;
//     return GenMemCopyVar(isCopyUBToGM, isSpillToGm);
// }

// std::string CodeGenOpLiteNPU::GenUBCopyIn() const {
//     return GenMemUBTransfer(false);
// }

// std::string CodeGenOpLiteNPU::GenUBCopyOut() const {
//     return GenMemUBTransfer(true);
// }

// // In static shape scene, GM Offset is already calculated and added to GM Addr in host side, so TileOp do not need
// // GM offset
// std::string CodeGenOpLiteNPU::GenMemCopyVar(bool isCopyLocalToGM, bool isSpillToGm, unsigned uf) const {
//     unsigned gmIdx = isCopyLocalToGM ? 0 : 1;
//     unsigned localIdx = isCopyLocalToGM ? 1 : 0;
//     OperandType localType = operandType[localIdx];
//     std::vector<std::string> addrTypeHead(ID2);
//     addrTypeHead[gmIdx] = GetAddrTypeByOperandType(BUF_DDR);
//     addrTypeHead[localIdx] = GetAddrTypeByOperandType(localType);

//     std::vector<int64_t> gmShape = rawShape[gmIdx];
//     CODEGEN_LOGI("gmShape is %s", IntVecToStr(gmShape).c_str());
//     std::vector<int64_t> localRawShape = rawShape[localIdx];
//     CODEGEN_LOGI("localRawShape is %s", IntVecToStr(localRawShape).c_str());

//     std::vector<std::string> addrExpr(ID2);
//     addrExpr[localIdx] = sm->QueryVarNameByTensorMagic(operandWithMagic[localIdx]);
//     addrExpr[gmIdx] = isSpillToGm ? GenGMAddrExprWithOffset(GM_STACK_BASE) : GenGmParamVar(gmIdx);

//     std::vector<std::string> dataTypeExpr(ID2);
//     dataTypeExpr[gmIdx] = DataType2CCEStr(operandDtype[gmIdx]);
//     dataTypeExpr[localIdx] = DataType2CCEStr(operandDtype[localIdx]);

//     if (localType == BUF_L0C) {
//         return PrintMemCopyWithL0C({uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, localRawShape, dataTypeExpr});
//     } else if (localType == BUF_L1) {
//         return PrintMemCopyWithL1(
//             {isCopyLocalToGM, isSpillToGm, uf, gmIdx, localIdx, addrTypeHead, addrExpr, gmShape, localRawShape,
//              dataTypeExpr});
//     } else if (localType == BUF_UB) {
//         PrintMemCopyWithUBParam param = {gmIdx, localIdx, isSpillToGm, addrTypeHead, addrExpr, dataTypeExpr};
//         return PrintMemCopyWithUB(param);
//     }

//     ASSERT(OperErr::OPERAND_TYPE_UNSUPPORTED, false)
//         << "GenMemCopyVar: cannot support current localType!!!" << localType;
//     return {};
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const {
//     if (isSupportLayout) {
//         return PrintMemCopyWithL0CTileTensor(param);
//     }

//     return "";
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const {
//     if (isSupportLayout) {
//         return PrintMemCopyWithL1TileTensor(param);
//     }

//     return "";
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyWithL1TileTensor(const PrintMemCopyWithL1Param &param) const {
//     // if (param.isCopyLocalToGM) {
//     //     return PrintMemCopyOutWithL1TileTensor(param);
//     // }
//     // TODO...
//     (void)param;

//     return PrintMemCopyInWithL1TileTensor(param);
// }

// std::vector<std::string> CodeGenOpLiteNPU::GetGmOffsetForTileTensor(unsigned gmIdx, bool isSpillingToGM) const {
//     int dim = static_cast<int>(rawShape[gmIdx].size());
//     std::vector<std::string> gmOffsetExpr;
//     if (isSpillingToGM || functionType == FunctionType::STATIC) {
//         return std::vector<std::string>(dim, "0");
//     }

//     if (offsetFromAttr[gmIdx][ID0].IsValid()) {
//         return GenSymbolicArgument(offsetFromAttr[gmIdx]);
//     }

//     return {""}; // TODO...
//     // return GenGetParamMacroPacked(gmIdx, dim, PREFIX_STR_OFFSET);
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam &param) const {
//     std::vector<std::string> tileOpParamList =
//         GeTileOpParamForNormalCopyTileTensor(param.gmIdx, param.addrExpr[param.gmIdx], param.isSpillingToGM);
//     std::ostringstream oss;
//     oss << tileOpName;
//     oss << WrapParamByParentheses(tileOpParamList);
//     oss << STMT_END;
//     return oss.str();
// }

// std::string CodeGenOpLiteNPU::PrintTensorForCopyBetweenGM(
//     unsigned operandIdx, unsigned gmIdx, const std::string& gmVarName) const
// {
//     std::string tensor =
//         operandIdx == gmIdx ? sm->QueryTileTensorNameByBufVar(gmVarName) : QueryTileTensorNameByIdx(operandIdx);
//     return tensor;
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam &param) const {
//     std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
//     std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
//     // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);
//     std::string gmVarName = param.addrExpr[param.gmIdx];
//     std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
//     std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
//     int64_t reluMode = 0;
//     GetAttr(OP_ATTR_PREFIX + "relu_type", reluMode);
//     std::string src1Tensor = srcTensor;
//     int64_t nzValue = 0;
//     int64_t isAcc = 0;
//     auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape);
//     GetAttr(OP_ATTR_PREFIX + "atomic_add", isAcc);
//     GetAttr(OpAttributeKey::copyIsNZ, nzValue);
//     std::string nzVar = nzValue ? "CopyOutMode::NZ2NZ" : "CopyOutMode::NZ2ND";
//     std::vector<std::string> storeConfigList = {nzVar, std::to_string(isAcc), std::to_string(reluMode)};
//     std::string storeConfig = WrapParamByAngleBrackets(storeConfigList);

//     Element scaleValue = Element(DataType::DT_UINT64, 0);
//     if (!isAcc) {
//         GetAttr(OP_ATTR_PREFIX + "scale_value", scaleValue);
//     }

//     if ((!scaleValue.GetUnsignedData()) &&
//         ((operandDtype[param.localIdx] == DT_INT32) && (operandDtype[param.gmIdx] == DT_FP16))) {
//         src1Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC1_IDX));
//     }

//     std::vector<std::string> tileOpParamList = {dstTensor,
//                                                 srcTensor,
//                                                 src1Tensor,
//                                                 coord,
//                                                 outerValueStr,
//                                                 innerValueStr,
//                                                 std::to_string(scaleValue.GetUnsignedData())};
//     std::ostringstream oss;
//     oss << tileOpName << "<" << TSTORE_CONF << storeConfig << ">";
//     oss << PrintParams({"(", ")"}, tileOpParamList, ", ");
//     oss << STMT_END;
//     return oss.str();
// }

// std::vector<std::string> CodeGenOpLiteNPU::GeTileOpParamForNormalCopyTileTensor(
//     unsigned gmIdx, const std::string &gmVarName, bool isSpillingToGM) const {
//     std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(gmIdx, isSpillingToGM);
//     // e.g. ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
//     // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coord = PrintCoord(rawShape[gmIdx].size(), coordCp);

//     std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), gmIdx, gmVarName);
//     std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), gmIdx, gmVarName);
//     std::vector<std::string> tileOpParamList = {dstTensor, srcTensor, coord};
//     return tileOpParamList;
// }

// std::string CodeGenOpLiteNPU::PrintMemCopyInWithL1TileTensor(const PrintMemCopyWithL1Param &param) const {
//     std::vector<std::string> gmOffsetExpr = GetGmOffsetForTileTensor(param.gmIdx);
//     // constructor call parameter ((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coordCp = WrapParamByParentheses(gmOffsetExpr);
//     // e.g. Coord4Dim((RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 0)),(RUNTIME_COA_GET_PARAM_OFFSET(2, 136, 1)))
//     std::string coord = PrintCoord(rawShape[param.gmIdx].size(), coordCp);
//     std::string gmVarName = param.addrExpr[param.gmIdx];
//     std::string dstTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::DST_IDX), param.gmIdx, gmVarName);
//     std::string srcTensor = PrintTensorForCopyBetweenGM(ToUnderlying(MISOIdx::SRC0_IDX), param.gmIdx, gmVarName);
//     std::vector<std::string> tileOpParamList =
//         GeTileOpParamForNormalCopyTileTensor(param.gmIdx, gmVarName, param.isSpillingToGM);

//     auto [outerValueStr, innerValueStr] = GetOuterInnerValueStr(param.gmIdx, param.gmShape, param.isSpillingToGM);
//     if (opCode != Opcode::OP_L1_COPY_IN_A_SCALE && opCode != Opcode::OP_L1_COPY_IN_B_SCALE) {
//         tileOpParamList.insert(tileOpParamList.end(), {outerValueStr, innerValueStr});
//     }
//     int64_t copyInMode = -1;
//     if (opAttrs.count(OpAttributeKey::copyInMode)) {
//         copyInMode = AnyCast<int64_t>(opAttrs.at(OpAttributeKey::copyInMode));
//     }
//     auto cpMode = static_cast<Matrix::CopyInMode>(copyInMode);
//     int64_t nzValue = 0;
//     auto ret = GetAttr(OpAttributeKey::copyIsNZ, nzValue);
//     if (ret && nzValue) {
//         cpMode = Matrix::CopyInMode::NZ2NZ;
//     }
//     std::string cpModeStr = CopyInModeToString(cpMode);

//     int64_t paddingMode = 0;
//     GetAttr(OP_ATTR_PREFIX + "copy_in_l1_padding_mode", paddingMode);
//     std::string padModStr = PaddingModeToString(static_cast<Matrix::PaddingMode>(paddingMode));

//     std::ostringstream oss;
//     if (opCode == Opcode::OP_L1_COPY_IN_A_SCALE || opCode == Opcode::OP_L1_COPY_IN_B_SCALE) {
//         oss << tileOpName << WrapParamByAngleBrackets({cpModeStr}) << WrapParamByParentheses(tileOpParamList)
//             << STMT_END;
//     } else {
//         oss << tileOpName << WrapParamByAngleBrackets({cpModeStr, padModStr}) << WrapParamByParentheses(tileOpParamList)
//             << STMT_END;
//     }
//     return oss.str();
// }

// std::pair<std::string, std::string> CodeGenOpLiteNPU::GetOuterInnerValueStr(
//     unsigned gmIdx, const std::vector<int64_t>& gmShape, bool isSpillingToGM) const
// {
//     (void)gmIdx; // TODO...
//     int64_t outerValue = 0;
//     int64_t innerValue = 0;
//     GetAttr("op_attr_outer_value", outerValue);
//     GetAttr("op_attr_inner_value", innerValue);

//     bool useStaticShape = functionType == FunctionType::STATIC || isSpillingToGM;
//     // auto gmShapeExprByIndex = GenParamIdxExprByIndex(gmIdx, SHAPE_DIM2, PREFIX_STR_RAW_SHAPE);

//     // auto getValueStr = [useStaticShape, &gmShapeExprByIndex](
//                         //    int64_t value, size_t idx, int64_t shapeValue) -> std::string {
//     auto getValueStr = [useStaticShape](
//                            int64_t value, int64_t shapeValue) -> std::string {
//         if (value != 0) {
//             return std::to_string(value);
//         }
//         // return useStaticShape ? std::to_string(shapeValue) : gmShapeExprByIndex[idx];
//         return std::to_string(shapeValue);
//     };

//     // return {getValueStr(outerValue, 0, gmShape[0]), getValueStr(innerValue, 1, gmShape[1])};
//     return {getValueStr(outerValue, gmShape[0]), getValueStr(innerValue, gmShape[1])};
// }


// // When ub tensor spilling to GM occurred, the spilling unit is entire raw shape of ub tensor.
// // So ub offset is always zero under this scene, do not need to calculate anymore.
// std::string CodeGenOpLiteNPU::PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const {
//     if (isSupportLayout) {
//         return PrintMemCopyWithUBTileTensor(param);
//     }

//     return "";
// }

// std::string CodeGenOpLiteNPU::GenGMAddrExprWithOffset(const std::string& addrExpr) const
// {
//     // gm offset of spilling workspace is calculated by pass, the value is saved in dim 0.
//     int64_t gmOffset = 0;
//     // gmOffset Default to 0 when the attribute is not set
//     GetAttr(OpAttributeKey::workspaceBaseOffset, gmOffset);
//     std::ostringstream oss;
//     if (gmOffset == 0) {
//         oss << addrExpr;
//     } else {
//         oss << "((__gm__ uint8_t*)" << addrExpr << " + " << gmOffset << ")";
//     }

//     return oss.str();
// }
// } // namespace npu::tile_fwk
