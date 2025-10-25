# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_CaseFilterList
        ########################## STATIC ##########################
        # composite
        "TestCodegenLlama.TestCodegenLlama"
        # mte
        "TestCodegenScatterUpdate.TestScatterupdateDim2"
        "TestCodegenScatterUpdate.TestBatchMatmul"
        "TestCodegenScatterUpdate.TestScatterUpdate"
        "TestCodegenSpillOut.L1SpillOut"
        "TestCodegenSpillOut.UBSpillOut"
        # vector
        "TestCodegenAllGather.TestAllGatherTensorGraph"
        "TestCodegenAllGather.TestAllGatherOutTensorTensorGraph"
        "TestCodegenAllGather.TestAllGatherMatmulTensorGraph"
        "TestCodegenArgSort.TestArgSort"
        "TestCodegenDynBinary.TestCodegenAddDim2"
        "TestCodegenDynBinary.TestCodegenAddDim2SrcNotSameShape"
        "TestCodegenDynBinaryBrc.TestMulDynamic"
        "TestCodegenDynUna.TestAbsDynamic"
        "TestCodegenDynUna.TestDynExpand"
        "TestCodegenDynSort.TestDynBitSort"
        "TestCodegenDynSort.TestDynMrgSort"
        "TestCodegenDynSort.TestDynExtract"
        "TestCodegenDynBinary.TestAddsDynamic"
        "TestCodegenDynGather.TestGather"
        "TestCodegenDynScalar.TestScalarAdds"
        "TestCodegenDynScalar.TestScalarDivs"
        "TestCodegenBinary.TestCodegenAddDim2"
        "TestCodegenBinary.TestCodegenAddDim2ByJson"
        "TestCodegenBinary.TestCodegenAddDim3"
        "TestCodegenBinary.TestCodegenAddDim4"
        "TestCodegenBinary.TestCodegenAddSDim2"
        "TestCodegenBinary.TestCodegenAddSDim3"
        "TestCodegenBinary.TestCodegenAddSDim4"
        "TestCodegenBinary.TestCodegenAddMulDim4TileTensor"
        "TestCodegenGather.TestGather"
        "TestCodegenGather.TestGatherEle"
        "TestCodegenReduceScatter.TestReduceScatter"
        "TestCodegenReduceScatter.TestReduceScatterOneTensor"
        "TestCodegenScatter.TestScatter_"
        "TestCodegenUnary.RowMaxSingleDim2"
        "TestCodegenUnary.RowMaxSingleDim3"
        "TestCodegenUnary.RowMaxSingleDim4"
        "TestCodegenUnary.RowSumSingleDim2"
        "TestCodegenUnary.RowSumSingleDim3"
        "TestCodegenUnary.RowSumSingleDim4"
        "TestCodegenUnary.TransposeVnchwconvDim2"
        "TestCodegenUnary.TransposeVnchwconvDim4"
        "TestCodegenUnary.TransposeVnchwconvDim5"
        "TestCodegenUnary.RowMaxExpandDim2"
        "TestCodegenUnary.CastDim1"
        "TestCodegenUnary.ExpandDim4Axis0"
        "TestCodegenUnary.ExpandDim4Axis1"
        "TestCodegenUnary.ExpandDim2Axis0"
        "TestCodegenScalar.TestScalarOp"
        "TestCodegenScalar.TestQuant_32_1_7168"
        "TestCodegenScalar.TestPipeAll"
        "TestCodegenUnary.RowSumDim4Axis2"
        "TestCodegenUnary.RowSumDim3Axis1"
        "TestCodegenUnary.RowSumDim2Axis0"
        "TestCodegenUnary.TestVecDup"
        "TestCodegenUnary.TestVecDupUnaligned"

        # distributed
        "TestCodegenDispatch.TestMoeDispatchRoutingExpert"
        "TestCodegenDispatch.TestMoeDispatchSharedExpert"
        "TestCodegenMoeCombine.TestMoeCombine"
        "TestDistributedAllGather.TestAllGatherDyn"

        ########################## DYNAMIC ##########################
        # mte
        "TestCodegenDynCopy.L0CToOut"
        "TestCodegenDynCopy.L0CToOutUnalign"
        "TestCodegenDynCopy.L1CopyIn"
        "TestCodegenDynCopy.L1CopyInNZ"
        "TestCodegenDynCopy.L1CopyInNZWithValue"
        "TestCodegenDynCopy.UBCopyIn"
        "TestCodegenDynTransposeDataMove.TransposeDataMoveDim3"
        "TestCodegenDynTransposeDataMove.TransposeDataMoveDim4"
        "TestCodegenDynDataMove.TestDatamoveAlignDim4"
        "TestCodegenDynIndexOutCast.IndexOutCast"
        # dynamic support unaligned
        "TestCodegenDynIndexOutCast.DynIndexOutUnaligned"
        "TestCodegenDynBinary.TestGatherEle"
        "TestCodegenDynBinary.AddUnalignTileTensor"
        "TestCodegenDynDataMove.TestDatamoveUnalignDim3"
        "TestCodegenDynDataMove.TestDatamoveUnalignDim4"
        "TestCodegenDynVdup.TestDynVdupUnaligned"
        "TestCodegenDynRowSumLine.TestOperationRowSumLine"
        "TestCodegenDynVnchwconv.TransposeDynVnchwconvDim3"
        "TestCodegenDynSpillOut.UBSpillOut"
        "TestCodegenDynSpillOut.L1SpillOut"
        # composite
        # "TestCodegenDynFFN.FFNDynamicTest"
        "TestCodegenDynFFN.FFNQuantDynamicTest"
        "TestCodegenDynPa.PaHighThroughputDviewLargeDynamicValidShape"
        "TestCodegenStaticUnderDyn.TestStaticFuncUnderDyn"
)

set(_Target ${PTO_Fwk_UTestNamePrefix}_codegen)

file(GLOB_RECURSE _Sources src/*.cpp)
set(_Include "include")

PTO_Fwk_UTest_AddCaseLib(
        TARGET ${_Target}
        SOURCES ${_Sources}
        PRIVATE_INCLUDE_DIRECTORIES ${_Include}
        PUBLIC_LINK_LIBRARIES tile_fwk_codegen tile_fwk_interface
        GTEST_FILTER_LIST ${_CaseFilterList}
)
