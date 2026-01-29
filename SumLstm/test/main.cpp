/**
 * @file main.cpp
 *
 * SumLstm Custom Operator Test
 * Uses aclnn interface to call the SumLstm operator
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_sum_lstm.h"

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);
    return SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return SUCCESS;
}

template <typename T>
bool ReadBinaryFile(const std::string &filename, std::vector<T> &data, size_t expectedSize)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    data.resize(expectedSize);
    file.read(reinterpret_cast<char *>(data.data()), expectedSize * sizeof(T));
    file.close();
    return true;
}

template <typename T>
bool WriteBinaryFile(const std::string &filename, const std::vector<T> &data)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
    file.close();
    return true;
}

void DestroyResources(std::vector<void *> tensors, std::vector<void *> deviceAddrs, aclrtStream stream,
                      int32_t deviceId, void *workspaceAddr = nullptr)
{
    for (uint32_t i = 0; i < tensors.size(); i++) {
        if (tensors[i] != nullptr) {
            aclDestroyTensor(reinterpret_cast<aclTensor *>(tensors[i]));
        }
        if (i < deviceAddrs.size() && deviceAddrs[i] != nullptr) {
            aclrtFree(deviceAddrs[i]);
        }
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main(int argc, char **argv)
{
    // 1. Initialize device/stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return FAILED);

    // 2. Define shapes and parameters
    const int64_t batch = 2;
    const int64_t seqLen = 4;
    const int64_t hiddenDim = 64;
    const int64_t gatedDim = 4 * hiddenDim;

    std::vector<int64_t> states4dShape = {batch, seqLen, gatedDim};
    std::vector<int64_t> prevCellShape = {batch, seqLen, hiddenDim};
    std::vector<int64_t> weightShape = {hiddenDim};

    // Attributes
    float alpha = 1.0f;
    float epsCell = 1e-6f;
    float epsState = 1e-6f;
    bool useFastGelu = true;

    // 3. Prepare host data
    int64_t states4dSize = GetShapeSize(states4dShape);
    int64_t prevCellSize = GetShapeSize(prevCellShape);
    int64_t weightSize = GetShapeSize(weightShape);

    std::vector<aclFloat16> states4dHostData(states4dSize);
    std::vector<aclFloat16> z4_4dHostData(states4dSize);
    std::vector<aclFloat16> prevCellHostData(prevCellSize);
    std::vector<aclFloat16> wCellHostData(weightSize);
    std::vector<aclFloat16> bCellHostData(weightSize);
    std::vector<aclFloat16> wStateHostData(weightSize);
    std::vector<aclFloat16> bStateHostData(weightSize);
    std::vector<aclFloat16> outStateHostData(prevCellSize, aclFloatToFloat16(0.0f));
    std::vector<aclFloat16> outCellHostData(prevCellSize, aclFloatToFloat16(0.0f));

    // Try to read from input files, otherwise generate test data
    if (!ReadBinaryFile("input_states_4d.bin", states4dHostData, states4dSize)) {
        LOG_PRINT("Generating test input data...\n");
        for (int64_t i = 0; i < states4dSize; ++i) {
            states4dHostData[i] = aclFloatToFloat16(0.1f * (i % 10 - 5));
        }
    }
    if (!ReadBinaryFile("input_z4_4d.bin", z4_4dHostData, states4dSize)) {
        for (int64_t i = 0; i < states4dSize; ++i) {
            z4_4dHostData[i] = aclFloatToFloat16(0.05f * (i % 10 - 5));
        }
    }
    if (!ReadBinaryFile("input_prev_cell.bin", prevCellHostData, prevCellSize)) {
        for (int64_t i = 0; i < prevCellSize; ++i) {
            prevCellHostData[i] = aclFloatToFloat16(0.1f * (i % 10 - 5));
        }
    }
    if (!ReadBinaryFile("input_w_cell.bin", wCellHostData, weightSize)) {
        for (int64_t i = 0; i < weightSize; ++i) {
            wCellHostData[i] = aclFloatToFloat16(1.0f);
        }
    }
    if (!ReadBinaryFile("input_b_cell.bin", bCellHostData, weightSize)) {
        for (int64_t i = 0; i < weightSize; ++i) {
            bCellHostData[i] = aclFloatToFloat16(0.0f);
        }
    }
    if (!ReadBinaryFile("input_w_state.bin", wStateHostData, weightSize)) {
        for (int64_t i = 0; i < weightSize; ++i) {
            wStateHostData[i] = aclFloatToFloat16(1.0f);
        }
    }
    if (!ReadBinaryFile("input_b_state.bin", bStateHostData, weightSize)) {
        for (int64_t i = 0; i < weightSize; ++i) {
            bStateHostData[i] = aclFloatToFloat16(0.0f);
        }
    }

    // 4. Create device tensors
    void *states4dDeviceAddr = nullptr;
    void *z4_4dDeviceAddr = nullptr;
    void *prevCellDeviceAddr = nullptr;
    void *wCellDeviceAddr = nullptr;
    void *bCellDeviceAddr = nullptr;
    void *wStateDeviceAddr = nullptr;
    void *bStateDeviceAddr = nullptr;
    void *outStateDeviceAddr = nullptr;
    void *outCellDeviceAddr = nullptr;

    aclTensor *states4dTensor = nullptr;
    aclTensor *z4_4dTensor = nullptr;
    aclTensor *prevCellTensor = nullptr;
    aclTensor *wCellTensor = nullptr;
    aclTensor *bCellTensor = nullptr;
    aclTensor *wStateTensor = nullptr;
    aclTensor *bStateTensor = nullptr;
    aclTensor *outStateTensor = nullptr;
    aclTensor *outCellTensor = nullptr;

    std::vector<void *> tensors;
    std::vector<void *> deviceAddrs;

    ret = CreateAclTensor(states4dHostData, states4dShape, &states4dDeviceAddr, ACL_FLOAT16, &states4dTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(states4dTensor); deviceAddrs.push_back(states4dDeviceAddr);

    ret = CreateAclTensor(z4_4dHostData, states4dShape, &z4_4dDeviceAddr, ACL_FLOAT16, &z4_4dTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(z4_4dTensor); deviceAddrs.push_back(z4_4dDeviceAddr);

    ret = CreateAclTensor(prevCellHostData, prevCellShape, &prevCellDeviceAddr, ACL_FLOAT16, &prevCellTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(prevCellTensor); deviceAddrs.push_back(prevCellDeviceAddr);

    ret = CreateAclTensor(wCellHostData, weightShape, &wCellDeviceAddr, ACL_FLOAT16, &wCellTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(wCellTensor); deviceAddrs.push_back(wCellDeviceAddr);

    ret = CreateAclTensor(bCellHostData, weightShape, &bCellDeviceAddr, ACL_FLOAT16, &bCellTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(bCellTensor); deviceAddrs.push_back(bCellDeviceAddr);

    ret = CreateAclTensor(wStateHostData, weightShape, &wStateDeviceAddr, ACL_FLOAT16, &wStateTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(wStateTensor); deviceAddrs.push_back(wStateDeviceAddr);

    ret = CreateAclTensor(bStateHostData, weightShape, &bStateDeviceAddr, ACL_FLOAT16, &bStateTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(bStateTensor); deviceAddrs.push_back(bStateDeviceAddr);

    ret = CreateAclTensor(outStateHostData, prevCellShape, &outStateDeviceAddr, ACL_FLOAT16, &outStateTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(outStateTensor); deviceAddrs.push_back(outStateDeviceAddr);

    ret = CreateAclTensor(outCellHostData, prevCellShape, &outCellDeviceAddr, ACL_FLOAT16, &outCellTensor);
    CHECK_RET(ret == SUCCESS, DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);
    tensors.push_back(outCellTensor); deviceAddrs.push_back(outCellDeviceAddr);

    // 5. Call aclnnSumLstm
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 获取 workspace 大小
    ret = aclnnSumLstmGetWorkspaceSize(states4dTensor, z4_4dTensor, prevCellTensor,
                                        wCellTensor, bCellTensor, wStateTensor, bStateTensor,
                                        alpha, epsCell, epsState, useFastGelu,
                                        outStateTensor, outCellTensor,
                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumLstmGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId); return FAILED);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }

    // 执行一次
    ret = aclnnSumLstm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumLstm failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // Benchmark: 每次迭代重新获取 executor
    const int benchmarkRuns = 10;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmarkRuns; ++i) {
        ret = aclnnSumLstmGetWorkspaceSize(states4dTensor, z4_4dTensor, prevCellTensor,
                                            wCellTensor, bCellTensor, wStateTensor, bStateTensor,
                                            alpha, epsCell, epsState, useFastGelu,
                                            outStateTensor, outCellTensor,
                                            &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumLstmGetWorkspaceSize failed in benchmark. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
        ret = aclnnSumLstm(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSumLstm benchmark failed. ERROR: %d\n", ret);
                  DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);
    }
    aclrtSynchronizeStream(stream);
    auto endTime = std::chrono::high_resolution_clock::now();
    double avgTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count() / benchmarkRuns;

    // 6. Copy results back to host
    std::vector<aclFloat16> outStateResult(prevCellSize);
    std::vector<aclFloat16> outCellResult(prevCellSize);

    ret = aclrtMemcpy(outStateResult.data(), prevCellSize * sizeof(aclFloat16), outStateDeviceAddr,
                      prevCellSize * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out_state failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    ret = aclrtMemcpy(outCellResult.data(), prevCellSize * sizeof(aclFloat16), outCellDeviceAddr,
                      prevCellSize * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out_cell failed. ERROR: %d\n", ret);
              DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr); return FAILED);

    // 7. Save outputs to binary files
    WriteBinaryFile("output_state.bin", outStateResult);
    WriteBinaryFile("output_cell.bin", outCellResult);

    // 8. Print results
    LOG_PRINT("=== SumLstm NPU Execution ===\n");
    LOG_PRINT("Input: states_4d=[%ld,%ld,%ld], prev_cell=[%ld,%ld,%ld]\n",
              batch, seqLen, gatedDim, batch, seqLen, hiddenDim);
    LOG_PRINT("Output: out_state=[%ld,%ld,%ld], out_cell=[%ld,%ld,%ld]\n",
              batch, seqLen, hiddenDim, batch, seqLen, hiddenDim);
    LOG_PRINT("Benchmark: %d runs\n", benchmarkRuns);
    LOG_PRINT("NPU Average Time: %.4f ms\n", avgTimeMs);

    LOG_PRINT("\nout_state first 10 values: ");
    for (int64_t i = 0; i < 10 && i < prevCellSize; i++) {
        LOG_PRINT("%.4f ", aclFloat16ToFloat(outStateResult[i]));
    }
    LOG_PRINT("\n");

    LOG_PRINT("out_cell first 10 values: ");
    for (int64_t i = 0; i < 10 && i < prevCellSize; i++) {
        LOG_PRINT("%.4f ", aclFloat16ToFloat(outCellResult[i]));
    }
    LOG_PRINT("\n");

    LOG_PRINT("\nOutputs saved to output_state.bin and output_cell.bin\n");

    // 9. Cleanup
    DestroyResources(tensors, deviceAddrs, stream, deviceId, workspaceAddr);

    return SUCCESS;
}
