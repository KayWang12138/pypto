template <typename T, typename T2, unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexOutcast(T dst, T src, T2 src1)
{
   constexpr auto expectSize = 5; // 所有张量都是 5D

    // === src0: data [1, B, S, 1, D] ===
    const auto uLayout = src.GetLayout();
    auto uShape1 = uLayout.template GetShapeDim<1, expectSize>(); // B
    auto uShape2 = uLayout.template GetShapeDim<2, expectSize>(); // S
    auto uShape4 = uLayout.template GetShapeDim<4, expectSize>(); // D

    // Stride for src0
    auto uStride1 = uLayout.template GetStrideDim<1, expectSize>(); // stride of B
    auto uStride2 = uLayout.template GetStrideDim<2, expectSize>(); // stride of S
    auto uStride4 = uLayout.template GetStrideDim<4, expectSize>(); // stride of D (should be 1)

    // === src1: indices [1, 1, 1, B, S] ===
    const auto iLayout = src1.GetLayout();
    auto iShape3 = iLayout.template GetShapeDim<3, expectSize>(); // B
    auto iShape4 = iLayout.template GetShapeDim<4, expectSize>(); // S

    auto iStride3 = iLayout.template GetStrideDim<3, expectSize>(); // stride of B
    auto iStride4 = iLayout.template GetStrideDim<4, expectSize>(); // stride of S (should be 1)

    // === dst: [1, N, K, 1, D] ===
    const auto dLayout = dst.GetLayout();
    auto dShape1 = dLayout.template GetShapeDim<1, expectSize>(); // N (logical rows)
    auto dShape2 = dLayout.template GetShapeDim<2, expectSize>(); // K = 32 or 128 (physical block size)
    auto dShape4 = dLayout.template GetShapeDim<4, expectSize>(); // D

    auto dStride1 = dLayout.template GetStrideDim<1, expectSize>(); //  This is the stride for one logical row (GmShape3 equivalent)
    auto dStride4 = dLayout.template GetStrideDim<4, expectSize>(); // should be 1

    using SrcDtype = typename T::Type;
    using IdxDtype = typename T2::Type;

    // === 提取对齐后的 raw shape（TileShape 是 5D）===
    constexpr auto src0rawShape2 = TileOp::GetTensorTileShapeDim<T, 2, 5>();   // S_32aligned (dim2)
    constexpr auto src0rawShape4 = TileOp::GetTensorTileShapeDim<T, 4, 5>();   // D_32aligned (dim4)
    constexpr auto src1rawShape4 = TileOp::GetTensorTileShapeDim<T2, 4, 5>();  // S_32aligned for index (dim4)

    if (uShape1 == 0 || uShape2 == 0 || uShape4 == 0 || iShape3 == 0 || iShape4 == 0) {
        return;
    }

    if constexpr (cacheMode == 2) {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
        __ubuf__ IdxDtype* idxBase = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
        __gm__ SrcDtype* dstBase   = reinterpret_cast<__gm__ SrcDtype*>(dst.GetAddr());

        unsigned B = iShape3;
        unsigned S = iShape4;
        unsigned D = uShape4;

        constexpr unsigned S_32aligned = src0rawShape2; //S 维的对齐长度（物理 UB 中每 B 块占多少 S）

        for (unsigned b = 0; b < B; ++b) {
            for (unsigned s = 0; s < S; ++s) {
                // 读取 index: src1[0,0,0,b,s]
                IdxDtype idx_val_raw = idxBase[b * iStride3 + s * iStride4];
                unsigned idx_val = static_cast<unsigned>(idx_val_raw);

                // 计算 src0 地址: [0, b, s, 0, 0]
                uint64_t srcOffset = b * uStride1 + s * uStride2; 
                // 构造 src tile: 1 x D
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, 1, D, pto::BLayout::RowMajor>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcBase + srcOffset));

                // 写入 dst: dst[0, idx_val, 0, 0, 0]
                __gm__ SrcDtype* scatterAddr = dstBase + idx_val * dStride1; // dStride1 = padded_D

                struct DummyGm { using Type = SrcDtype; };
                using ScatterShape5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                using ScatterStride5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                constexpr auto out_shape = ScatterShape5D{1, 1, 1, 1, D};
                constexpr auto out_stride = ScatterStride5D{0, 0, 0, 0, 1};

                auto dstGlobal = PtoGlobal<DummyGm, ScatterShape5D, ScatterStride5D, false>(
                    scatterAddr, out_shape, out_stride).Data();

                pto::TSTORE(dstGlobal, srcTile.Data());
            }

            // 修复点 4: padding 跳转必须基于 **物理 UB layout**
            constexpr unsigned D_32aligned = src0rawShape4;
            idxBase += (S_32aligned - S);
            srcBase += (S_32aligned - S) * D_32aligned;
        }
        return;
    }
    unsigned B = uShape1;   // = iShape3
    unsigned S = uShape2;   // = iShape4
    unsigned D = uShape4;   // logical D

    constexpr unsigned S_32aligned = src0rawShape2; // S aligned in UB
    constexpr unsigned D_32aligned = src0rawShape4; // D aligned in UB

    // UB 起始地址
    __ubuf__ SrcDtype* src0_base = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1_base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__  SrcDtype* dst_base   = reinterpret_cast<__gm__  SrcDtype*>(dst.GetAddr());

    // 按 B 维度遍历
    for (unsigned b = 0; b < B; ++b) {
        __ubuf__ SrcDtype* cur_src0 = src0_base + b * (S_32aligned * D_32aligned);
        __ubuf__ IdxDtype* cur_src1 = src1_base + b * S_32aligned;

        for (unsigned s = 0; s < S; ++s) {
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

            IdxDtype curValue = cur_src1[s];

            if constexpr (cacheMode == 1) { // PA_NZ
                T2 blockCount = curValue / blockSize;
                T2 index_in_block = curValue % blockSize;
                __gm__ SrcDtype* new_dst = dst_base +
                    (blockCount * blockSize + index_in_block) * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, 1, /*W*/D, pto::BLayout::RowMajor>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                struct DummyGm { using Type = SrcDtype; };
                using ScatterShape5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                using ScatterStride5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                constexpr auto out_shape = ScatterShape5D{1, 1, 1, 1, D};
                constexpr auto out_stride = ScatterStride5D{0, 0, 0, 0, 1};

                auto dstGlobal = PtoGlobal<DummyGm, ScatterShape5D, ScatterStride5D, false>(
                    new_dst, out_shape, out_stride).Data();

                pto::TSTORE(dstGlobal, srcTile.Data());
            } else { // cacheMode == 0 或其他：普通 scatter
                unsigned row_id = static_cast<unsigned>(curValue);
                __gm__ SrcDtype* new_dst = dst_base + row_id * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, 1, D, pto::BLayout::RowMajor>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                struct DummyGm { using Type = SrcDtype; };
                using ScatterShape5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                using ScatterStride5D = Std::tuple<size_t, size_t, size_t, size_t, size_t>;
                constexpr auto out_shape = ScatterShape5D{1, 1, 1, 1, D};
                constexpr auto out_stride = ScatterStride5D{0, 0, 0, 0, 1};

                auto dstGlobal = PtoGlobal<DummyGm, ScatterShape5D, ScatterStride5D, false>(
                    new_dst, out_shape, out_stride).Data();

                pto::TSTORE(dstGlobal, srcTile.Data());
            }
        }
        cur_src1 += S_32aligned;   // skip padded S dimension
        src0_base += S_32aligned  * D_32aligned;  // skip entire B block in src0
    }
}