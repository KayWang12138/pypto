# Tanh

## 描述
    计算输入张量的双曲正切值。
    输出张量的每个元素都是输入张量对应元素的双曲正切值。
    数学公式为：
    $$
    \text{tanh}(x) = \frac{\text{exp}(x) - \text{exp}(-x)}{\text{exp}(x) + \text{exp}(-x)}
    $$

    输入张量的每个元素都将被映射到范围 [-1, 1] 内。

## 实现方案
    ### 前端operation
    前端operation使用unary op实现，op type为TANH。
    其中TileTanhOperation单独实现，当输入数据类型为fp16、bf16时，创建一个tmpBuffer，buffer的大小为输入的最后两维shape的两倍，且最后一维需要32字节对齐。输入数据类型为fp32时，buffer的大小32B。
    codegen直接使用unarywithtmp实现，tmpBuffer的使用在unarywithtmp中已经实现，不需要额外处理。

    ### opcode
    对应opcode为OP_TANH。注册时应保证内存不复用，指定OpAttributeKey::excludeBufferReuse属性。

    ### 后端tileop
    tileop内计算如果输入数据类型为fp16、bf16，先将输入数据转换为fp32，计算完成后再将结果转换为fp16、bf16。
    如果输入数据类型为fp32，直接计算。
    输入数据类型为fp16、bf16时，由于在申请tmpBuffer时申请了两块tmp，在实际取地址时按照最后两位的TileShape进行取地址。
    ‘’‘
    // 极小值 epsilon，防止分母为 0
    static constexpr float EPSILON = 1.1754943508222875e-38f;
    inline float tanh_stable(float x) {
        const float abs_x = std::abs(x);

        // 计算 exp(-2*|x|)
        const float exp_neg_2absx = std::exp(-2.0f * abs_x);

        // 分子: x * (1 - exp(-2*|x|)))
        const float numerator = x * (1.0f - exp_neg_2absx);

        // 分母: (1 + exp(-2*|x|))) * (|x| + epsilon)
        const float denominator = (1.0f + exp_neg_2absx) * (abs_x + EPSILON);

        return numerator / denominator;
    }
    ’‘’
    具体的计算过程参考这段c++实现，其中EPSILON为一个极小值，防止分母为0。
