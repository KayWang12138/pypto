template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskPerAxisImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t srcAxisLen,
    const uint32_t bucketSize)
{
    const auto paddingLen = AlignUp(bucketSize / ONE_BYTE_BIT_SIZE, ONE_BLK_SIZE);
    const auto localMaskTmpOffset = paddingLen;
    const auto localScalarOffset = sizeof(half) * bucketSize;
    LocalTensor<uint8_t> localMask = sharedTmpBuffer;
    LocalTensor<half> localMaskTmp = sharedTmpBuffer.ReinterpretCast<half>();
    SetVectorMask<half, MaskMode::COUNTER>(0, srcAxisLen);
    CastMaskToHalfImpl<U>(localMaskTmp, mask);
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    constexpr auto loopSize = ONE_REPEAT_BYTE_SIZE / sizeof(half);
    const auto repeatTime = DivCeil(srcAxisLen, loopSize);

    if constexpr (!reverse) {
        CompareScalar<half, uint8_t, false>(localMask, localMaskTmp, static_cast<half>(0), CMPMODE::EQ,
            MASK_PLACEHOLDER, repeatTime, unaryParams);
    } else {
        CompareScalar<half, uint8_t, false>(localMask, localMaskTmp, static_cast<half>(0), CMPMODE::NE,
            MASK_PLACEHOLDER, repeatTime, unaryParams);
    }
    PipeBarrier<PIPE_V>();
    Select(dst, localMask, src0, 1, binaryParams);
}