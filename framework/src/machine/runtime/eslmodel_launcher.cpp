#ifdef BUILD_WITH_CANN

#include <thread>
#include "machine/runtime/eslmodel_launcher.h"
#include "machine/runtime/device_launcher.h"
#include "interface/utils/op_info_manager.h"



extern "C" int DynTileFwkBackendKernelServer(void *targ);
namespace npu::tile_fwk::dynamic {

int EslModelLaunchAicore(aclrtStream aicoreStream, void *kernel, DeviceKernelArgs *kernelArgs) {
    rtArgsEx_t rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void *> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    return rtKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim_, &rtArgs, nullptr, aicoreStream, &cfg);
}

int DynamicKernelLaunchEsl(DeviceKernelArgs *kArgs, aclrtStream aicoreStream, void *kernel) {
    auto *devProg = (dynamic::DevAscendProgram *)(kArgs->cfgdata);
    devProg->devArgs.nrAic = 32;
    devProg->devArgs.nrAiv = 64;
    auto ret = EslModelLaunchAicore(aicoreStream, kernel, kArgs);
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    for (size_t k = 0; k < inputDataList.size(); k++) {
        auto &inputData = inputDataList[k];
        if (inputData) {
            memcpy_s(inputData->GetDevPtr(), inputData->size(), (uint8_t *)inputData->data(), inputData->size());
        }
    }
    for (size_t k = 0; k < outputDataList.size(); k++) {
        auto &outputData = outputDataList[k];
        if (outputData) {
            memcpy_s(outputData->GetDevPtr(), outputData->size(), (uint8_t *)outputData->data(), outputData->size());
        }
    }
    devProg->devArgs.enableEslModel = true;
    size_t shmSize = dynamic::DEVICE_TASK_CTRL_POOL_SIZE + dynamic::DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    (void)memset_s(reinterpret_cast<void*>(devProg->devArgs.runtimeDataRingBufferAddr), shmSize, 0, shmSize);
    int threadNum = static_cast<int>(devProg->devArgs.nrAicpu);
    threadNum = (devProg->devArgs.enableCtrl == 1) ? threadNum : threadNum + 1;
    std::vector<std::thread> aicpus(threadNum);
    std::atomic<int> idx{0};
    std::this_thread::sleep_for(std::chrono::seconds(10));
    for (int i = 0; i < threadNum; i++) {
        aicpus[i] = std::thread([&]() {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            std::string name = "aicput" + std::to_string(tidx);
            pthread_setname_np(pthread_self(), name.c_str());
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            (void)DynTileFwkBackendKernelServer(kArgs);
        });
    }
    for (int i = 0; i < threadNum; i++) {
        if (aicpus[i].joinable()) {
            aicpus[i].join();
        }
    }
    EslModelMemoryUtils::UnmapAllMappings();
    return 0;
}

void ExchangeCaputerMode(const bool &isCapture) {
    if (isCapture) {
        aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
        aclmdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", mode);
    }
}

int EslModelLauncher::EslModelLaunchDeviceTensorData(Function *function,
    const std::vector<DeviceTensorData> &inDevList, const std::vector<DeviceTensorData> &outDevList,
    rtStream_t aicpuStream, aclrtStream aicoreStream, void *kernel, const DeviceLauncherConfig &config) {
    MACHINE_LOGI("Kernel Launch");
    bool isCapture = false;

    DeviceLauncher::SetCaptureStream(aicoreStream, aicpuStream, isCapture);

    if (isCapture) {
        DeviceLauncher::ChangeCaptureModelRelax();
    }

    auto rc = aclInit(nullptr);
    if (rc != 0 && rc != ACL_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }

    auto dynAttr = function->GetDyndevAttribute();
    DeviceKernelArgs kArgs;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
    EslModelMemoryUtils eslMemoryUtils;
 	DeviceLauncher::DeviceInitDistributedContext(eslMemoryUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(eslMemoryUtils, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(eslMemoryUtils, kArgs, inDevList, outDevList, dynAttr->disableL2List);
    // DeviceLauncher::RegisterKernelBin(function->GetDyndevAttribute()->kernelBinary);
    ExchangeCaputerMode(isCapture);

    auto rc = DynamicKernelLaunchEsl(&kArgs, aicoreStream, kernel);
    if (rc < 0) {
        return rc;
    }
    rc = rtStreamSynchronize(aicoreStream);
    return rc;
}

int EslModelLauncher::EslModelRunOnce(Function *function, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = machine::GetRA()->GetScheStream();
    auto aicoreStream = machine::GetRA()->GetStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    EslModelMemoryUtils devMemoryHugePage(true);
    EslModelMemoryUtils devMemoryNotHugePage(false);
    Function *function = Program::GetInstance().GetLastFunction();
 	std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(devMemoryHugePage, inputDataList, outputDataList);
    int rc = EslModelLaunchDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream, aicoreStream, kernel, config);
    
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        DeviceLauncher::CopyFromDev(devMemoryNotHugePage, inputDataList);
    }
    return rc;
}
}
#endif