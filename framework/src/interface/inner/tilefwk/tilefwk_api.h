/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tilefwk_api.h
 * \brief
 */

#pragma once

#include <functional>
#include <string>
#include <vector>
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

/*************************************************************************************************************
* 功能说明: TileFwk 环境初始化
* 输入参数:
  SocVersion： 在线场景如torch，torchnpu只调用该接口但可以不传递SocVersion；接口内调用rts接口获取SocVersion。
               离线场景(未来)，用户离线编译时需传递SocVersion。
* 输出参数: 无
* 返 回 值: 0表示成功，非零表示失败。
* 备 注: 调用者不传递SocVersion时函数内通过runtime接口获取并设置
  **********************************************************************************************************/
int32_t TileFwkInit(const std::string &socVersion = "Ascend910B1");

/*************************************************************************************************************
* 功能说明: capture TileFwk function
* 输入参数:
  funcName  函数名
  opArgs  函数参数声明， 要保证和 TileFwkRun 传入的args的顺序一致
* 输出参数: 无
* 返 回 值: 0表示成功，非零表示失败。
  **********************************************************************************************************/
int32_t TileFwkBeginFunction(const std::string &funcName, const std::vector<std::reference_wrapper<Tensor>> &opArgs);

/*************************************************************************************************************
 * 功能说明: finish capture TileFwk function
 * 输入参数:
 *     isWaitTaskFinished: whether waiting task finished
 * 输出参数: 无
 * 返 回 值: 0表示成功，非零表示失败。
 * 备 注: 和 TileFwkBeginFunction 配对使用
 **********************************************************************************************************/
int32_t TileFwkEndFunction(const bool isWaitTaskFinished = false);

/*************************************************************************************************************
 * 功能说明: 用户构图后完成图优化、codegen、buildkernel
 * 输入参数: 无
 * 输出参数: 无
 * 返 回 值: 编译产物handle
 **********************************************************************************************************/
void *TileFwkCompile();

/*************************************************************************************************************
* 功能说明: 返回给前端workspace大小，客户申请内存
* 输入参数:
    handle： compile返回值
* 输出参数:
    workspaceSize： workspace大小
* 返 回 值: 0表示成功，非零表示失败。
  **********************************************************************************************************/
int32_t TileFwkGetWorkspaceSize(const void *handle, uint64_t *workspaceSize);

/*************************************************************************************************************
* 功能说明: handle资源回收
* 输入参数:
    handle：compile返回值
* 输出参数: 无
* 返 回 值: 无
  **********************************************************************************************************/
void TileFwkFreeHandle(const void *handle);

/*************************************************************************************************************
 * 功能说明: TileFwk 环境资源释放
 * 输出参数: 无as
 * 返 回 值: 无
 * 备 注: 和 TileFwkInit 配对使用
 **********************************************************************************************************/
void TileFwkFinalize();

/*************************************************************************************************************
 * 功能说明: 设置cube tile shape
 * 输入参数: matrix m/k/n
 * 输出参数: 无
 * 返 回 值: 无
  **********************************************************************************************************/
void TileFwkSetCubeTileShapes(const std::array<int64_t, 2> &m, const std::vector<int64_t> &k, const std::array<int64_t, 2> &n);

/*************************************************************************************************************
* 功能说明: 设置vector shape
* 输入参数:
    tileShape ： vec tile shape
* 输出参数: 无
* 返 回 值: 无
  **********************************************************************************************************/
void TileFwkSetVecTileShapes(const std::vector<int64_t> &tileShape);

/*************************************************************************************************************
* 功能说明: Tensor 赋值操作 dst = src
* 输入参数:
    src ： 源Tensor对象
* 输出参数:
     dst :  目的Tensor对象
* 返 回 值: 无
  **********************************************************************************************************/
void TileFwkAssign(Tensor &dst, const Tensor &src);

bool TileOpCompile(const std::string &opType, const uint64_t configKey, const std::string &kernelName,
    const std::string &dumpPath);

extern "C" bool TileFwkCompileFatbin(const char *opType, const char *socVersion, const char *dumpPath,
    const char *kernelName);

namespace Distributed {
/*************************************************************************************************************
* 功能说明: 设置Distributed通信tile shape
* 输入参数:
    row : 源Tensor的第一维度，按照矩阵理解表示行
    col : 源Tensor的第二维度，按照矩阵理解表示列
    rank: 按照通信域内rank维度做切分，即每个切块对应到多少卡
* 输出参数: 无
* 返 回 值: 无
  **********************************************************************************************************/
void TileFwkSetDistTileShapes(std::array<int, 0x3> row, std::array<int, 0x3> col, std::array<int, 0x3> rank);

/*************************************************************************************************************
* 功能说明: 指定本卡的rankId。若未指定rankId，则在前端表达生成动态图，运行时判断rankId做动态下发处理。
            若指定rankId，则在前端表达中根据rankId做切图和生成代码，优先生成静态图。
            注意，该函数前需要在SetDistTileShapes之后调用
* 输入参数:
    rankId : 本卡的rankId序号
* 输出参数: 无
* 返 回 值: 无
  **********************************************************************************************************/
void TileFwkSpecifyStaticRankId(int rankId);
} // namespace Distributed
} // namespace npu::tile_fwk
