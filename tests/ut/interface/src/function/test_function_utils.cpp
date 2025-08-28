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
 * \file test_function_utils.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/configs/config_storage.h"
#include "passes/pass_utils/pass_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"

using namespace npu::tile_fwk;

class FunctionUtilsTest : public testing::Test {
public:
    static void SetUpTestCase() {
        std::cout << "FunctionUtilsTest SetUpTestCase" << std::endl;
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }

    static void TearDownTestCase() {
        std::cout << "FunctionUtilsTest TearDownTestCase" << std::endl;
    }

    void SetUp() override {
        std::cout << "FunctionUtilsTest SetUp" << std::endl;
        Program::GetInstance().Reset();
            }

    void TearDown() override {
        std::cout << "FunctionUtilsTest TearDown" << std::endl;
        Program::GetInstance().Reset();
    }
};

TEST_F(FunctionUtilsTest, TestCloneOperation) {
    std::vector<int64_t> shape{8, 16};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    Program::GetInstance().GetTileShape().SetVecTileShapes(shape);
    FUNCTION("main", FunctionType::STATIC) {
        output = AddS(input, Element(DT_FP32, 1.0));
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_main");
    ASSERT_NE(func, nullptr);
    for (const auto &op : func->Operations(false)) {
        if (op.GetOpcode() == Opcode::OP_ADDS) {
            std::vector<std::shared_ptr<LogicalTensor>> ioperands;
            std::vector<std::shared_ptr<LogicalTensor>> ooperands;
            for (auto iOperand : op.GetIOperands()) {
                std::shared_ptr<LogicalTensor> tensor = iOperand->Clone(*func, true);
                ioperands.push_back(tensor);
            }
            for (auto oOperand : op.GetOOperands()) {
                std::shared_ptr<LogicalTensor> tensor = oOperand->Clone(*func, true);
                ooperands.push_back(tensor);
            }
            Operation &opClone = op.CloneOperation(*func, ioperands, ooperands);
            EXPECT_EQ(op.GetOOperands()[0]->GetShape(), opClone.GetOOperands()[0]->GetShape());
            break;
        }
    }
}

TEST_F(FunctionUtilsTest, TestRemoveOperationCase1) {
    Program::GetInstance().GetTileShape().SetVecTileShapes({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("M1") {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Exp(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_M1");
    ASSERT_NE(func, nullptr);

    Operation *first_exp = nullptr;
    size_t alloc_count1 = 0;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
        std::cout << "input ctrl operation:";
        for (const auto &in_ctrl_op : op.GetInCtrlOperations()) {
            if (in_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << in_ctrl_op->opmagic << ", " << in_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << "output ctrl operation:";
        for (const auto &out_ctrl_op : op.GetOutCtrlOperations()) {
            if (out_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << out_ctrl_op->opmagic << ", " << out_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_EXP && first_exp == nullptr) {
            first_exp = &op;
        }
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            ASSERT(!op.IsDeleted());
            op.SetAsDeleted();
        }
        if (op.GetOpcode() == Opcode::OP_UB_ALLOC) {
            alloc_count1++;
        }
    }
    std::cout << "Alloc1 op size:" << alloc_count1 << std::endl;

    ASSERT(!first_exp->IsDeleted());
    first_exp->SetAsDeleted();
    func->EraseOperations(true);

    std::cout << "===============after remove first exp====================" << std::endl;
    size_t exp_op_count = 0;
    size_t alloc_count2 = 0;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
        std::cout << "input ctrl operation:";
        for (const auto in_ctrl_op : op.GetInCtrlOperations()) {
            if (in_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << in_ctrl_op->opmagic << ", " << in_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << "output ctrl operation:";
        for (const auto out_ctrl_op : op.GetOutCtrlOperations()) {
            if (out_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << out_ctrl_op->opmagic << ", " << out_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_EXP) {
            exp_op_count++;
        }
        if (op.GetOpcode() == Opcode::OP_UB_ALLOC) {
            alloc_count2++;
        }
    }
    std::cout << "Alloc2 op size:" << alloc_count2 << std::endl;
    EXPECT_EQ(exp_op_count, 3);

    func->EraseOperations(true);

    std::cout << "===============after remove sqrt ops====================" << std::endl;
    size_t sqrt_op_count = 0;
    size_t alloc_count3 = 0;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (op.GetOpcode() == Opcode::OP_MUL) {
                    EXPECT_EQ(item_op->GetOpcode(), Opcode::OP_COPY_IN);
                }
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
        std::cout << "input ctrl operation:";
        for (const auto in_ctrl_op : op.GetInCtrlOperations()) {
            if (in_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << in_ctrl_op->opmagic << ", " << in_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << "output ctrl operation:";
        for (const auto out_ctrl_op : op.GetOutCtrlOperations()) {
            if (out_ctrl_op == nullptr) {
                continue;
            }
            std::cout << "(" << out_ctrl_op->opmagic << ", " << out_ctrl_op->GetOpcodeStr() << ") ";
        }
        std::cout << std::endl << std::endl;
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_op_count++;
        }
        if (op.GetOpcode() == Opcode::OP_UB_ALLOC) {
            alloc_count3++;
        }
    }
    std::cout << "Alloc3 op size:" << alloc_count1 << std::endl;
    EXPECT_EQ(sqrt_op_count, 0);
}

TEST_F(FunctionUtilsTest, TestRemoveOperationCase2) {
    Program::GetInstance().GetTileShape().SetVecTileShapes({16, 16});
    std::vector<int64_t> shape{16, 16};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("M2", FunctionType::STATIC) {
        Tensor in0 = Reciprocal(input);
        Tensor in1 = Exp(in0);
        Tensor in2 = Sqrt(in1);
        Tensor in3 = Exp(in2);

        Tensor in4 = Abs(in0);
        Tensor in5 = Sqrt(in4);
        Tensor in6 = Exp(in5);
        output = Mul(in3, in6);
    }

    Function *func = Program::GetInstance().GetFunctionByRawName("TENSOR_M2");
    ASSERT_NE(func, nullptr);

    bool isFirstRemoveOp = true;
    for (auto &op : func->Operations()) {
        bool add_output_op = false;
        if (op.GetOpcode() == Opcode::OP_EXP && isFirstRemoveOp) {
            ASSERT(!op.IsDeleted());
            op.SetAsDeleted();
            isFirstRemoveOp = false;
            add_output_op = true;
        }
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (add_output_op) {
                    op.SetAsDeleted();
                    isFirstRemoveOp = false;
                }
            }
        }
        std::cout << std::endl;
    }

    func->EraseOperations(true);
    std::cout << "===============after remove one set of exp and sqrt====================" << std::endl;
    Operation *exp_op = nullptr;
    Operation *sqrt_op = nullptr;
    for (auto &op : func->Operations()) {
        bool is_exp_re = false;
        bool is_exp_mul = false;
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (op.GetOpcode() == Opcode::OP_EXP && item_op->GetOpcode() == Opcode::OP_RECIPROCAL) {
                    is_exp_re = true;
                }
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (op.GetOpcode() == Opcode::OP_EXP && item_op->GetOpcode() == Opcode::OP_MUL) {
                    is_exp_mul = true;
                }
            }
        }
        std::cout << std::endl;
        if (is_exp_re && is_exp_mul) {
            exp_op = &op;
        }
        if (op.GetOpcode() == Opcode::OP_SQRT) {
            sqrt_op = &op;
        }
    }
    ASSERT_NE(exp_op, nullptr);
    ASSERT_NE(sqrt_op, nullptr);

    FunctionUtils::RelinkOperationInput(exp_op, 0, sqrt_op, 0);
    std::cout << "===============after relink input of exp====================" << std::endl;
    size_t exp_bw_sqrt_mul = 0;
    for (auto &op : func->Operations()) {
        bool is_exp_sqrt = false;
        bool is_exp_mul = false;
        std::cout << "Op:" << op.opmagic<< " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (op.GetOpcode() == Opcode::OP_EXP && item_op->GetOpcode() == Opcode::OP_SQRT) {
                    is_exp_sqrt = true;
                }
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
                if (op.GetOpcode() == Opcode::OP_EXP && item_op->GetOpcode() == Opcode::OP_MUL) {
                    is_exp_mul = true;
                }
            }
        }
        std::cout << std::endl;
        if (is_exp_sqrt && is_exp_mul) {
            exp_bw_sqrt_mul++;
        }
    }
    ASSERT_EQ(exp_bw_sqrt_mul, 2);
}
