template <typename DstTensor, typename SrcTensor>
TILEOP void TPad(DstTensor dst, SrcTensor src, float padValue, LoopVar srcValidRow, LoopVar srcValidCol) {
    constexpr auto dstShapeSize = Std::tuple_size<typename DstTensor::Shape>::value;
    constexpr auto srcShapeSize = Std::tuple_size<typename SrcTensor::Shape>::value;
    static_assert(srcShapeSize == dstShapeSize, "Pad: Src and Dst rank mismatch");

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    using SrcDtype = typename SrcTensor::Type;
    using DstDtype = typename DstTensor::Type;

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<DstTensor, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<DstTensor, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<SrcTensor, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<SrcTensor, 4, 5>();

    using DstTileType = pto::Tile<pto::TileType::Vec, DstDtype, 1, dstTileW, pto::BLayout::RowMajor, -1, -1,
                                  pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
    using SrcTileType = pto::Tile<pto::TileType::Vec, SrcDtype, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;

    if(dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar hIndex = 0; hIndex < dstShape3; ++hIndex) {
                    DstTileType dstTile(1, dstShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 + hIndex * dstStride3;
                    auto dstAddr = dst.GetAddr() + dstOffs et * sizeof(DstDtype);
                    pto::TASSIGN(dstTile, dstAddr);

                    if (hIndex < srcValidRow && srcValidCol > 0) {
                        SrcTileType srcTile(1, srcValidCol);
                        auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 + hIndex * srcStride3;
                        auto srcAddr = src.GetAddr() + srcOffset * sizeof(SrcDtype);
                        pto::TASSIGN(srcTile, srcAddr);

                        pto::TFILLPAD(dstTile, srcTile);
                    } else {
                        pto::TEXPANDS(dstTile, static_cast<DstDtype>(padValue));
                    }
                    (void)padValue;
                }
            }
        }
    }
}