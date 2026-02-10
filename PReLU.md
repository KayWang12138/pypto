## 接口定义
'''
Tensor prelu(Tensor input, Tensor weight)
'''

## 功能描述
输入张量 input 中的每个元素，根据其符号（正/负）选择不同的计算路径：
- 当元素为正时，输出该元素本身。
- 当元素为负时，输出该元素与权重 weight 相乘的结果。
输入张量 input的第二维 和权重张量 weight 必须具有相同的形状。
假设输入张量 input 的形状为 (N, C, ...)，权重张量 weight 的形状为 (C,)，则输出张量的形状与输入张量相同。
weight 需要满足约束
‘’‘
weight 非空
weight与input的输入类型一致
weight为一维
weight的元素数量必须与input的第二维相同
’‘’

**输出**
输出张量 output 的每个元素根据其符号和权重计算得到：
- 当 input[i, j, ...] 为正时，output[i, j, ...] = input[i, j, ...]
- 当 input[i, j, ...] 为负时，output[i, j, ...] = input[i, j, ...] * weight[j]

## 方案分析
**TILEOP接口定义**
axis 参数代表个左填充扩充5维后，channel轴所在的位置，控制函数走不同的实现
'''
template <typename T, typename T1, typename T2, typename T3, int axis>
TILEOP void TPrelu(T1 dst, T2 src, T3 weight, T3 tmp)
'''

**tileFunc**
'''
void TiledPReLUOperation(
    Function &function, const TileShape &tileShape, size_t cur, Input &input, Input &weight, const LogicalTensorPtr &result) {
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto weightTile = weight.tensor.GetStorage()->View(function, weight.tileInfo.shape, weight.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        int axis = 5 - (cur + 1) + 1;
        constexpr size_t ALIGN_SIZE = 32;
        int64_t tmpSize = ALIGN_SIZE / 2;
        if (axis == 4) {
            tmpSize = (input.tileInfo.shape[cur] + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        }
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT16, tmpShape);
        function.AddOperation(Opcode::OP_PRELU, {tile, weightTile}, {resultTile, tmpTensor});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }
    auto &vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        if (i == 1) {
            weight.tileInfo.shape[0] = std::min(weight.tensor.GetShape()[cur] - i, vecTile[cur]);
            weight.tileInfo.offset[0] = i;
        }
        TiledPReLUOperation(function, tileShape, cur + 1, input, result);
    }
}
'''

**axis=4**
这种情况，对应输入为两维即(N, C)，权重为(C,)，输出为(N, C)，计算src[N,:] * weight[:],调用pto::TPRELU实现计算
'''
for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TPRELU(dstTile, srcTile, weightTile, tmpTile);
            }
        }
    }
'''

**axis=2,3**
这种情况，对应输入为三维即(N, C, H, W)，权重为(C,)，输出为(N, C, H, W)，计算src[N, C, :, :] * weight[C],需要循环控制H, W。调用PTO::TLRELU实现计算

'''
for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto negaative_slope = weightTile[n2Index];
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TLRELU(dstTile, srcTile, negaative_slope);
            }
        }
    }
'''