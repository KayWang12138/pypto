/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_vf.cpp
 * \brief
 */

#include "codegen_vf.h"

#include <algorithm>

#include "interface/tensor/logical_tensor.h"
#include "interface/function/function.h"
#include "codegen/codegen_common.h"
#include "securec.h"

namespace npu::tile_fwk {
const std::unordered_set<Opcode> BinaryOps{Opcode::OP_ADD, Opcode::OP_SUB, Opcode::OP_MUL, Opcode::OP_DIV,
    Opcode::OP_PAIRMAX, Opcode::OP_PAIRMIN, Opcode::OP_PAIRSUM, Opcode::OP_EXPAND, Opcode::OP_FUSED_OP};
const std::unordered_set<Opcode> UnaryOps{
    Opcode::OP_EXP,
    Opcode::OP_NEG,
    Opcode::OP_SQRT,
    Opcode::OP_ABS,
    Opcode::OP_ROWMAX,
    Opcode::OP_ROWEXPSUM,
    Opcode::OP_ROWEXPMAX,
};

std::map<Opcode, const std::string> VFTileOpNameMap{
  // VF
    { Opcode::OP_MUL,  "vmul"},
    {Opcode::OP_SQRT, "vsqrt"},
    {Opcode::OP_ADDS, "vadds"},
    { Opcode::OP_ADD,  "vadd"},
    { Opcode::OP_SUB,  "vsub"},
    { Opcode::OP_EXP,  "vexp"},
    { Opcode::OP_VLD,  "vlds"},
    { Opcode::OP_VST,  "vsts"},
};

void VFCodeGen::GenCode(Function *func, const std::string &file) {
    path_ = file;
    std::string vfcodeList;
    for (auto &program : func->programs_) {
        std::string KernelName = program.second->GetMagicName();
        auto opList = program.second->Operations(false).DuplicatedOpList();
        std::string vfcode;
        magicToBufferId_.clear();
        AllocBufferId(program.second, opList);
        vfcode += GenVFHeader(KernelName, program.second->inCasts_, program.second->outCasts_);
        vfcode += GenVFBody(opList);
        vfcode += GenVFEnd();
        vfcodeList += vfcode;
    }
    if (vfcodeList.empty()) {
        return;
    }
    std::ofstream os;
    os.open(file);
    os << vfcodeList;
    isGenSuccess_ = !os.fail();
}

std::string VFCodeGen::GenVFBody(const std::vector<Operation *> &OpList) {
    std::string vfbody;
    for (auto &op : OpList) {
        std::string code = GenSingleOp(op);
        vfbody += "        ";
        vfbody += code;
    }
    return vfbody;
}

std::string VFCodeGen::GenSingleOp(Operation *op) {
    InitOpParm(op);
    auto opCode = op->GetOpcode();
    if (opCode == Opcode::OP_REG_ALLOC) {
        auto tensorMagic = (*op->GetOutCtrlOperations().begin())->GetOOperands()[ID0]->GetMagic();
        if (!magicToBufferId_.count(tensorMagic)) {
            ASSERT(false) << "magic has not allocated buffer. magic " << tensorMagic << " op " << op->GetOpcodeStr();
        }
        operand[ID0] = magicToBufferId_[tensorMagic];
        operandDtype[ID0] = (*op->GetOutCtrlOperations().begin())->GetOOperands()[ID0]->tensor->datatype;
        return GenRegAlloc();
    }
    auto iter = VFTileOpNameMap.find(opCode);
    if (iter == VFTileOpNameMap.end()) {
        return std::string{"NOT HANDLED OP: " + op->GetOpcodeStr()};
    }
    std::string tile_op_name = VFTileOpNameMap[opCode];

    auto iter_reg_binary = BinaryOps.find(opCode);
    if (iter_reg_binary != BinaryOps.end()) {
        return GenBinaryRegOp(tile_op_name);
    }

    auto iter_reg_unary = UnaryOps.find(opCode);
    if (iter_reg_unary != UnaryOps.end()) {
        return GenUnaryRegOp(tile_op_name);
    }

    if (opCode == Opcode::OP_VLD) {
        return GenVLD(tile_op_name);
    }

    if (opCode == Opcode::OP_VST) {
        return GenVST(tile_op_name);
    }

    return std::string{"NOT HANDLED OP: " + op->GetOpcodeStr()};
}

void VFCodeGen::InitOpParm(Operation *op) {
    auto opCode = op->GetOpcode();
    std::vector<int64_t> attrShape;
    std::vector<int64_t> attrOffset;
    bool useAttrShape = false;
    bool useAttrOffset = false;
    if (opCode == Opcode::OP_VLD || opCode == Opcode::OP_VST) {
        std::shared_ptr<CopyOpAttribute> attr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (!attr) {
            ASSERT(false) << ": missing OpAttr in copy op:" << op->GetOpcodeStr();
        }
        attrShape = attr->GetSpecifiedShape(1);
        if (opCode == Opcode::OP_VLD) {
            auto opImmList = attr->GetCopyInAttr().first;
            attrOffset.resize(opImmList.size());
            for (size_t i = 0; i < opImmList.size(); ++i) {
                attrOffset[i] = opImmList[i].GetSpecifiedValue().ConcreteValid() ?
                                    static_cast<int>(opImmList[i].GetSpecifiedValue()) :
                                    -1;
            }
        }
        if (opCode == Opcode::OP_VST) {
            auto opImmList = attr->GetCopyOutAttr().second;
            attrOffset.resize(opImmList.size());
            for (size_t i = 0; i < opImmList.size(); ++i) {
                attrOffset[i] = opImmList[i].GetSpecifiedValue().ConcreteValid() ?
                                    static_cast<int>(opImmList[i].GetSpecifiedValue()) :
                                    -1;
            }
        }
        useAttrShape = true;
        useAttrOffset = true;
    }

    int i = 0;
    for (const auto &tensor : op->GetOOperands()) {
        if (!magicToBufferId_.count(tensor->GetMagic())) {
            ASSERT(false) << "magic has not allocated buffer. magic " << tensor->GetMagic() << " op "
                          << op->GetOpcodeStr();
        }
        operand[i] = magicToBufferId_[tensor->GetMagic()];
        shape[i] = useAttrShape ? attrShape : tensor->shape;
        offset[i] = useAttrOffset ? attrOffset : tensor->offset;
        rawShape[i] = tensor->tensor->rawshape;
        originShape[i] = tensor->oriShape;
        operandDtype[i] = tensor->tensor->datatype;
        i++;
    }
    for (const auto &tensor : op->GetIOperands()) {
        if (!magicToBufferId_.count(tensor->GetMagic())) {
            ASSERT(false) << "magic has not allocated buffer. magic " << tensor->GetMagic() << " op "
                          << op->GetOpcodeStr();
        }
        operand[i] = magicToBufferId_[tensor->GetMagic()];
        shape[i] = useAttrShape ? attrShape : tensor->shape;
        offset[i] = useAttrOffset ? attrOffset : tensor->offset;
        rawShape[i] = tensor->tensor->rawshape;
        originShape[i] = tensor->oriShape;
        operandDtype[i] = tensor->tensor->datatype;
        i++;
    }
}

void VFCodeGen::AllocBufferId(Function *func, std::vector<Operation *> &opList) {
    int bufferId = 0;
    for (const auto &ele : func->inCasts_) {
        if (!magicToBufferId_.count(ele->GetMagic())) {
            magicToBufferId_[ele->GetMagic()] = bufferId;
            bufferId++;
        }
    }
    for (const auto &ele : func->outCasts_) {
        if (!magicToBufferId_.count(ele->GetMagic())) {
            magicToBufferId_[ele->GetMagic()] = bufferId;
        }
    }
    for (auto &op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            auto tensorMagic = (*op->GetOutCtrlOperations().begin())->GetOOperands()[ID0]->GetMagic();
            if (magicToBufferId_.count(tensorMagic)) {
                ASSERT(false) << "already allocated buffer or duplicate magic. tensor magic " << tensorMagic
                              << " op magic " << op->GetOpMagic() << " op " << op->GetOpcodeStr();
            }
            magicToBufferId_[tensorMagic] = bufferId;
            bufferId++;
        }
    }
}

std::string VFCodeGen::GenVarName(std::string loc, int id) {
    std::string VarName(loc + "Id" + std::to_string(id) + "Addr");
    return VarName;
}

std::string VFCodeGen::GenVFHeader(const std::string &KernelName, std::vector<std::shared_ptr<LogicalTensor>> &ubIn,
    std::vector<std::shared_ptr<LogicalTensor>> &ubOut) {
    std::string Line =
        std::string("#include \"TileOpImpl.h\" \n namespace TileOp {\nTILEOP void ") + KernelName + std::string("(");
    for (const auto &out : ubOut) {
        auto id = magicToBufferId_[out->GetMagic()];
        auto buffer = GenVarName("UB", id);
        std::string dtypeStr = DataType2CCEStr(out->tensor->datatype);
        Line += std::string("__ubuf__ ") + dtypeStr + std::string("* ") + std::string(buffer) + std::string(", ");
    }
    int i = 0;
    for (; i < static_cast<int>(ubIn.size() - 1); ++i) {
        auto id = magicToBufferId_[ubIn[i]->GetMagic()];
        auto buffer = GenVarName("UB", id);
        std::string dtypeStr = DataType2CCEStr(ubIn[i]->tensor->datatype);
        Line += std::string("__ubuf__ ") + dtypeStr + std::string("* ") + std::string(buffer) + std::string(", ");
    }
    auto id = magicToBufferId_[ubIn[i]->GetMagic()];
    auto buffer = GenVarName("UB", id);
    std::string dtypeStr = DataType2CCEStr(ubIn[i]->tensor->datatype);
    Line += std::string("__ubuf__ ") + dtypeStr + std::string("* ") + std::string(buffer);
    Line += std::string(") {\n    __VEC_SCOPE__\n    {\n        vector_bool allMask = pge_b8(PAT_ALL);\n");
    return Line;
}

void VFCodeGen::UpdateVarOffset(std::vector<std::string *> vars, std::vector<unsigned> operandIdxes) const {
    ASSERT(vars.size() == operandIdxes.size())
        << "vars size vs operandIdxes is not equal" << vars.size() << " vs. " << operandIdxes.size();

    for (size_t i = 0; i < vars.size(); ++i) {
        int resOffset{0};
        std::vector varOffset = offset[operandIdxes[i]];
        if (varOffset.empty()) {
            continue;
        }
        std::vector varRawShape = rawShape[operandIdxes[i]];
        ASSERT(!varRawShape.empty()) << "varRawShape is empty!!";
        ASSERT(varOffset.size() == varRawShape.size())
            << "varOffset " << IntVecToStr(varOffset) << ", size " << varOffset.size() << " vs varRawShape "
            << IntVecToStr(varRawShape) << ", size " << varRawShape.size() << " is not equal!!";
        int base = 1;
        for (int j = varOffset.size() - 1; j >= 0; j--) {
            resOffset += varOffset[j] * base;
            base *= varRawShape[j];
        }
        if (resOffset == 0) {
            continue;
        }
        ALOG_INFO("var: ", *vars[i], ", raw shape: ", IntVecToStr(varRawShape), ", offset: ", IntVecToStr(varOffset),
            ", resOffset: ", resOffset);
        ASSERT(vars[i]) << "var[" << i << "] is null!!";
        *vars[i] += " + " + std::to_string(resOffset);
    }
}

std::string VFCodeGen::GenVLD(const std::string &code) {
    std::string DName = GenVarName("REG", operand[ID0]);
    std::string S0Name = GenVarName("UB", operand[ID1]);
    UpdateVarOffset({&S0Name}, {1});

    std::ostringstream oss;
    oss << code << "(" << DName << ", " << S0Name << ", ELE_CNT_B16, NORM);\n";

    return oss.str();
}

std::string VFCodeGen::GenVST(const std::string &code) {
    std::string DName = GenVarName("UB", operand[ID0]);
    std::string S0Name = GenVarName("REG", operand[ID1]);
    UpdateVarOffset({&DName}, {0});

    std::ostringstream oss;
    oss << code << "(" << S0Name << ", " << DName << ", ELE_CNT_B16, NORM_B16, allMask);\n";

    return oss.str();
}

std::string VFCodeGen::GenRegAlloc() {
    std::string VarName = GenVarName("REG", operand[ID0]);
    std::string dtypeStr = DataType2VectorRegStr(operandDtype[ID0]);
    return dtypeStr + " " + VarName + ";\n";
}

std::string VFCodeGen::GenBinaryRegOp(const std::string &BinaryOp) {
    std::string DName = GenVarName("REG", operand[ID0]);
    std::string S0Name = GenVarName("REG", operand[ID1]);
    std::string S1Name = GenVarName("REG", operand[ID2]);

    std::ostringstream oss;
    oss << BinaryOp << "(" << DName << ", " << S0Name << ", " << S1Name << ", allMask);\n";

    return oss.str();
}

std::string VFCodeGen::GenUnaryRegOp(const std::string &UnaryOp) {
    std::string DName = GenVarName("REG", operand[ID0]);
    std::string S0Name = GenVarName("REG", operand[ID1]);

    std::ostringstream oss;
    oss << UnaryOp << "(" << DName << ", " << S0Name << ", allMask);\n";

    return oss.str();
}

std::string VFCodeGen::GenVFEnd() {
    return std::string{"    }\n}\n}\n"};
}

std::string VFCodeGen::GetVFHeaderForInclude() const {
    return "#include \"" + path_ + "\"\n";
}

} // namespace npu::tile_fwk
