#define OP_TILE_OP_FMOD TFmod
template <typename T0, typename T1, typename T2>
TILEOP void TFmod(T0 dst, T1 src0, T2 src1) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using Src0TileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using Src1TileDefine =
        pto::Tile<pto::TileType::Vec, typename T2::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    Src0TileDefine src0Tile(dstShape3, dstShape4);
    Src1TileDefine src1Tile(dstShape3, dstShape4);
    using DstType = typename T0::Type;
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + dstOffset * dstTypeSize));
                if constexpr (std::is_same_v<DstType, float>) {
                    using Src0TileTmp =
                        pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Src0TileTmp src0TileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(src0TileTmp, (uint64_t)(src1.GetAddr() + (dstOffset + dstTileH * dstTileW * 2) * dstTypeSize));
                    pto::TMOV(src0TileTmp, src0Tile);
                    pipe_barrier(PIPE_ALL);
                    pto::TDIV(dstTile, src0Tile, src1Tile);
                    pipe_barrier(PIPE_ALL);
                    using TruncedTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    TruncedTileDefine truncedTileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(truncedTileTmp, (uint64_t)(src1.GetAddr() + (dstOffset + dstTileH * dstTileW * 3) * dstTypeSize));
                    pto::TCVT(truncedTileTmp, dstTile, pto::RoundMode::CAST_TRUNC);
                    pipe_barrier(PIPE_ALL);
                    pto::TMUL(dstTile, truncedTileTmp, src1Tile);
                    pipe_barrier(PIPE_ALL);
                    pto::TSUB(dstTile, src0TileTmp, dstTile);
                } else if constexpr (std::is_same_v<DstType, half> || std::is_same_v<DstType, bfloat16_t>) {
                    using Fp32TmpTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Fp32TmpTileDefine dstTileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src0TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src1TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine truncedTileTmp(dstShape3, dstShape4);
                    auto fp32TmpTileOffset = ((dstOffset + dstTileH * dstTileW) * dstTypeSize) + (dstTileH * dstTileW * sizeof(float));
                    pto::TASSIGN(dstTileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset));
                    pto::TASSIGN(src0TileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset + (dstTileH * dstTileW * sizeof(float))));
                    pto::TASSIGN(src1TileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset + (dstTileH * dstTileW * sizeof(float) * 2)));
                    pto::TASSIGN(truncedTileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset + (dstTileH * dstTileW * sizeof(float) * 4)));
                    pto::TCVT(dstTileTmp, dstTile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src0TileTmp, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src1TileTmp, src1Tile, pto::RoundMode::CAST_NONE);

                    pipe_barrier(PIPE_V);
                    pto::TDIV(dstTileTmp, src0TileTmp, src1TileTmp);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(truncedTileTmp, dstTileTmp, pto::RoundMode::CAST_TRUNC);
                    pipe_barrier(PIPE_V);
                    pto::TMUL(dstTileTmp, truncedTileTmp, src1TileTmp);
                    pipe_barrier(PIPE_V);
                    pto::TSUB(src1TileTmp, src0TileTmp, dstTileTmp);
                    pipe_barrier(PIPE_V);
                    pto::TCVT(dstTile, src1TileTmp, pto::RoundMode::CAST_NONE);
                }
            }
        }
    }
}


#define OP_TILE_OP_FMODS TFmodS
template <typename Scalar, typename T0, typename T1>
TILEOP void TFmodS(T0 dst, T1 src0, Scalar src1) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    using DstTileDefine =
        pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using Src0TileDefine =
        pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    Src0TileDefine src0Tile(dstShape3, dstShape4);
    using DstType = typename T0::Type;
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + dstOffset * dstTypeSize));

                if constexpr (std::is_same_v<DstType, float>) {
                    using Src0TileTmp =
                        pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Src0TileTmp src0TileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(src0TileTmp, (uint64_t)(src0.GetAddr() + (dstOffset + dstTileH * dstTileW * 2) * dstTypeSize));
                    pto::TMOV(src0TileTmp, src0Tile);
                    pto::TDIVS(dstTile, src0Tile, src1);
                    using TruncedTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    TruncedTileDefine truncedTileTmp(dstShape3, dstShape4);
                    pto::TASSIGN(truncedTileTmp, (uint64_t)(src0.GetAddr() + (dstOffset + dstTileH * dstTileW * 3) * dstTypeSize));
                    pto::TCVT(truncedTileTmp, dstTile, pto::RoundMode::CAST_TRUNC);
                    pto::TMULS(dstTile, truncedTileTmp, src1);
                    pto::TSUB(dstTile, src0TileTmp, dstTile);
                } else if constexpr (std::is_same_v<DstType, half> || std::is_same_v<DstType, bfloat16_t>) {
                    float scalarSrc1 = static_cast<float>(src1);
                    using Fp32TmpTileDefine =
                        pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    Fp32TmpTileDefine dstTileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine src0TileTmp(dstShape3, dstShape4);
                    Fp32TmpTileDefine truncedTileTmp(dstShape3, dstShape4);
                    auto fp32TmpTileOffset = ((dstOffset + dstTileH * dstTileW) * dstTypeSize) + (dstTileH * dstTileW * sizeof(float));
                    pto::TASSIGN(dstTileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset));
                    pto::TASSIGN(src0TileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset + (dstTileH * dstTileW * sizeof(float))));
                    pto::TASSIGN(truncedTileTmp, (uint64_t)(src1.GetAddr() + fp32TmpTileOffset + (dstTileH * dstTileW * sizeof(float) * 3)));
                    pto::TCVT(dstTileTmp, dstTile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(src0TileTmp, src0Tile, pto::RoundMode::CAST_NONE);

                    pto::TDIVS(dstTileTmp, src0TileTmp, scalarSrc1);
                    pto::TCVT(truncedTileTmp, dstTileTmp, pto::RoundMode::CAST_TRUNC);
                    pto::TMULS(dstTileTmp, truncedTileTmp, scalarSrc1);
                    pto::TSUB(dstTileTmp, src0TileTmp, dstTileTmp);
                    pto::TCVT(dstTile, dstTileTmp, pto::RoundMode::CAST_NONE);

                }
            }
        }
    }
}