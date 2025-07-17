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
 * \file vf_fusion.cpp
 * \brief
 */

#include "vf_fusion.h"
#include "queue"
#include "interface/utils/id_gen.h"
#include "../tensor_graph_pass/expand_function.h"
#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/tile_graph_pass/merge_view_assemble.h"
#include "add_alloc.h"
#include "passes/tile_graph_pass/generate_move_op.h"

namespace npu::tile_fwk {

bool VFFusionPass::NextOpCanFuse(Operation *curOp) const {
   // current only support single output
   if (curOp->GetOutputOperand(0)->GetConsumers().size() > 1) {
       return false;
   }
   auto nextOp = *(curOp->GetOutputOperand(0)->GetConsumers().begin());
   bool ret = false;

   // binary op
   if (BinaryOps.count(nextOp->GetOpcode())) {
       ret = true;
   }
   // unary op
   if (UnaryOps.count(nextOp->GetOpcode())) {
       ret = true;
   }
   // vectorscalar op
   if (VectorScalarOps.count(nextOp->GetOpcode())) {
       ret = true;
   }
   return ret;
}

void VFFusionPass::PatternMatch(Function *function, std::vector<std::vector<Operation *>> &fusedList,
   std::vector<std::set<Operation *>> &fusedSet) const {
   auto operations = function->Operations().DuplicatedOpList();
   std::set<Operation *> visitedOp;
   for (int i = 0; i < static_cast<int>(operations.size()); i++) {
       // find binary op
       auto start_op = operations.at(i);
       auto iter = BinaryOps.find(start_op->GetOpcode());
       if (iter == BinaryOps.end())
           continue;
       // fuse a series of unary/vectorscalar ops
       std::vector<Operation *> toBeFuedOps;
       std::set<Operation *> toBeFuedOpsSet;
       toBeFuedOps.push_back(start_op);
       toBeFuedOpsSet.insert(start_op);
       bool hasFused = false;
       std::queue<Operation *> bfsQue;
       bfsQue.push(start_op);
       while (!bfsQue.empty()) {
           auto curOp = bfsQue.front();
           bfsQue.pop();
           if (!NextOpCanFuse(curOp)) {
               break;
           }
           hasFused = true;
           if (visitedOp.count(curOp)) {
               hasFused = false;
               break;
           }
           visitedOp.insert(curOp);
           for (auto &consumer : curOp->GetOutputOperand(0)->GetConsumers()) {
               bfsQue.push(consumer);
               toBeFuedOps.emplace_back(consumer);
               toBeFuedOpsSet.insert(consumer);
           }
       }
       if (!hasFused)
           continue;
       fusedList.push_back(toBeFuedOps);
       fusedSet.push_back(toBeFuedOpsSet);
   }
}

void VFFusionPass::Fusion(Function *function) const {
   std::vector<std::vector<Operation *>> fusedList;
   std::vector<std::set<Operation *>> fusedSet;
   PatternMatch(function, fusedList, fusedSet);

   std::vector<std::shared_ptr<npu::tile_fwk::Function>> vfFuncList;
   std::map<std::string, Operation *> vfOpMap;
   // merge op, make vf func
   for (size_t i = 0; i < fusedList.size(); i++) {
       auto fusedOps = fusedList[i];
       std::vector<std::shared_ptr<LogicalTensor>> mergedInputs;
       auto &curFusedSet = fusedSet[i];
       for (auto &op : fusedOps) {
           for (auto &inputTensor : op->GetIOperands()) {
               if (!curFusedSet.count(*(inputTensor->GetProducers().begin()))) {
                   mergedInputs.push_back(inputTensor);
               }
           }
       }
       std::vector<std::shared_ptr<LogicalTensor>> mergedOutputs;
       for (auto &op : fusedOps) {
           for (auto &outTensor : op->GetOOperands()) {
               bool hasOut = false;
               for (auto &consumer : outTensor->GetConsumers()) {
                   if (!curFusedSet.count(consumer)) {
                       hasOut = true;
                   }
               }
               if (hasOut) {
                   mergedOutputs.push_back(outTensor);
               }
           }
       }
       int maxTensorMagic = -1;
       for (auto &tensor : mergedInputs) {
           maxTensorMagic = std::max(maxTensorMagic, tensor->GetMagic());
       }
       for (auto &tensor : mergedOutputs) {
           maxTensorMagic = std::max(maxTensorMagic, tensor->GetMagic());
       }

       auto &vfFusedOp = function->AddOperation(Opcode::OP_FUSED_OP, mergedInputs, mergedOutputs); // magic
       vfFusedOp.UpdateSubgraphID(fusedOps[0]->GetSubgraphID());
       vfFusedOp.opmagic = fusedOps[0]->opmagic;
       // sub function
       auto rootName = function->GetRawName();
       auto funcMagicName = rootName + "_vf_graph_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
       auto funcName = rootName + "_vf_graph_" + std::to_string(IdGen<IdType::FUNCTION>::Inst().CurId());
       auto vfFunc = std::make_shared<Function>(Program().GetInstance(), funcMagicName, funcName, function);
       vfFunc->magicSeed_ = maxTensorMagic + 1;

       vfFunc->SetFunctionType(FunctionType::STATIC);
       vfFunc->SetGraphType(GraphType::TENSOR_GRAPH);
       auto ubTileShapes = Program::GetInstance().GetTileShape().GetVecTileShapes();
       std::vector<int> vfTileShape(ubTileShapes.size(), 1);
       vfTileShape[ubTileShapes.size() - 1] = VL_B16;
       for (auto ele : fusedOps) {
           Program::GetInstance().GetTileShape().SetVecTileShapes(vfTileShape);
           vfFunc->AddOperation(ele->GetOpcode(), ele->GetIOperands(), ele->GetOOperands());
       }
       vfFunc->inCasts_ = mergedInputs;
       std::vector<std::shared_ptr<LogicalTensor>> newOutputs;
       for (auto &out : mergedOutputs) {
           auto out_view = std::make_shared<LogicalTensor>(*vfFunc, out->Datatype(), out->shape);
           out_view->SetMemoryTypeBoth(MEM_UB);
           auto &assembleOp = vfFunc->AddOperation(Opcode::OP_ASSEMBLE, {out}, {out_view});
           assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(out->offset));
           newOutputs.push_back(out_view);
       }
       vfFunc->outCasts_ = newOutputs;
       vfFuncList.push_back(vfFunc);
       vfOpMap.insert({funcMagicName, &vfFusedOp});
   }

   // vf graph isomorphism
   std::map<uint64_t, std::shared_ptr<npu::tile_fwk::Function>> mergedFuncMap;
   std::map<uint64_t, std::string> wrapperNameMap;
   for (auto &vfFunc : vfFuncList) {
       uint64_t funcHash = vfFunc->ComputeHash().GetHash();
       std::cout << "funcHash " << funcHash << std::endl;
       if (!mergedFuncMap.count(funcHash)) {
           auto vfWrapperName = "FusedOpVF" + std::to_string(IdGen<IdType::FUNCTION>::Inst().NewId());
           wrapperNameMap[funcHash] = vfWrapperName;
           vfOpMap[vfFunc->GetMagicName()]->SetAttribute("VF_WRAPPER_NAME", vfWrapperName);
           vfFunc->funcMagicName_ = vfWrapperName;
           mergedFuncMap[funcHash] = vfFunc;
       } else {
           vfOpMap[vfFunc->GetMagicName()]->SetAttribute("VF_WRAPPER_NAME", wrapperNameMap[funcHash]);
       }
   }
   // expand && memory && copy
   int i = 0;
   for (const auto &item : mergedFuncMap) {
       auto vfFunc = item.second;
       vfFunc->rootFunc_ = function;
       function->programs_.insert({i, vfFunc.get()});
       Program::GetInstance().InsertFuncToFunctionMap(vfFunc->GetMagicName(), vfFunc);
       ExpandFunction expandFunction;
       expandFunction.Run(*vfFunc, "", "");
       MergeViewAssemble mergeViewAssemble;
       mergeViewAssemble.Run(*vfFunc, "", "");
       // assign memory
       AssignMemory(vfFunc);
       AddCopyAndAlloc(vfFunc);
       i++;
       vfFunc->SetFunctionType(FunctionType::STATIC);
       vfFunc->SetGraphType(GraphType::LEAF_VF_GRAPH);
   }

   // remove fused ops
   for (auto &fusedOps : fusedList) {
       for (auto &op : fusedOps) {
           op->SetAsDeleted();
       }
   }
   function->EraseOperations(false);
}

void VFFusionPass::AssignMemory(const std::shared_ptr<npu::tile_fwk::Function> &vfFunc) const {
   // assign memory
   for (auto &operation : vfFunc->Operations()) {
       if (operation.GetOpcode() != Opcode::OP_VIEW) {
           for (size_t j = 0; j < operation.iOperand.size(); ++j) {
               auto &tensor = operation.iOperand[j];
               tensor->SetMemoryTypeBoth(MEM_VECTOR_REG);
           }
       } else {
           auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(operation.GetOpAttribute().get());
           viewOpAttribute->SetToType(MEM_VECTOR_REG);
       }
       if (operation.GetOpcode() != Opcode::OP_ASSEMBLE) {
           for (size_t j = 0; j < operation.oOperand.size(); ++j) {
               auto &tensor = operation.oOperand[j];
               tensor->SetMemoryTypeBoth(MEM_VECTOR_REG);
           }
       } else {
           auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(operation.GetOpAttribute().get());
           assembleOpAttribute->SetFromType(MEM_VECTOR_REG);
       }
   }
}
void VFFusionPass::AddCopyAndAlloc(const std::shared_ptr<npu::tile_fwk::Function> &function) const {
   GenerateMoveOp generateMoveOp;
   generateMoveOp.Run(*function, "", "");

   std::vector<Operation *> toBeAllocList;
   for (auto &op : function->Operations().DuplicatedOpList()) {
       bool needAlloc = false;
       for (auto out : op->GetOOperands()) {
           if (out->GetMemoryTypeOriginal() == MemoryType::MEM_VECTOR_REG) {
               needAlloc = true;
           }
       }
       if (needAlloc) {
           toBeAllocList.push_back(op);
       }
   }

   int maxOpMagic = -1;
   for (auto toBeAlloc : toBeAllocList) {
       for (auto &op : function->Operations()) {
           maxOpMagic = std::max(maxOpMagic, op.GetOpMagic());
       }
       auto &allocOp = function->AddOperation(Opcode::OP_REG_ALLOC, {}, {});
       allocOp.opmagic = maxOpMagic + 1;
       FunctionUtils::AddControlEdge(allocOp, *toBeAlloc);
   }
}

Status VFFusionPass::PostCheck(Function &function) {
   for (auto subFunc : function.programs_) {
       for (auto &op : subFunc.second->Operations()) {
           if (op.GetOpcodeStr().find("ALLOC") != std::string::npos) {
           } else {
               for (auto out : op.GetOOperands()) {
                   if (out->GetMemoryTypeOriginal() == MemoryType::MEM_VECTOR_REG) {
                       // check allocate
                       if (op.GetInCtrlOperations().empty()) {
                           ASSERT(false) << "has not allocate ndoe.";
                       }
                   }
               }
           }
       }
   }
   return SUCCESS;
}

} // namespace npu::tile_fwk
