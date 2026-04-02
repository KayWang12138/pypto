#ifdef BUILD_WITH_CANN

#include <thread>
#include "machine/runtime/eslmodel_launcher.h"
#include "machine/runtime/device_launcher.h"
#include "interface/utils/op_info_manager.h"



extern "C" int DynTileFwkBackendKernelServer(void *targ);
namespace npu::tile_fwk::dynamic {

int DynamicKernelLaunchEsl(DeviceKernelArgs *kArgs) {
    auto *devProg = (dynamic::DevAscendProgram *)(kArgs->cfgdata);
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

int EslModelLauncher::EslModelRunOnce(Function *function, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    EslModelMemoryUtils devMemoryHugePage(true);
    EslModelMemoryUtils devMemoryNotHugePage(false);
 	std::tie(inputDeviceDataList, outputDeviceDataList) = DeviceLauncher::BuildInputOutputFromHost(devMemoryHugePage, inputDataList, outputDataList);
    int rc = EslModelLaunchDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, config);
    DeviceLauncher::CopyFromDev(devMemoryNotHugePage, outputDataList);
    if (HasInplaceArgs(function) || outputDataList.size() == 0) {
        DeviceLauncher::CopyFromDev(devMemoryNotHugePage, inputDataList);
    }
    // devMemoryNotHugePage.Free(devCtrlCache);
    return rc;
}

int EslModelLauncher::EslModelLaunchDeviceTensorData(Function *function,
    const std::vector<DeviceTensorData> &inDevList, const std::vector<DeviceTensorData> &outDevList,
    const DeviceLauncherConfig &config) {
    MACHINE_LOGI("Kernel Launch");
    auto dynAttr = function->GetDyndevAttribute();
    DeviceKernelArgs kArgs;
    EslModelMemoryUtils devMemory;
    DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(config);
 	DeviceLauncher::DeviceInitDistributedContext(devMemory, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(devMemory, kArgs, dynAttr->devProgBinary, nullptr, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(devMemory, kArgs, inDevList, outDevList, dynAttr->disableL2List);
    DeviceLauncher::DeviceRunCacheKernelSet(function, (uint8_t *)kArgs.cfgdata);
    DeviceLauncher::RegisterKernelBin(function->GetDyndevAttribute()->kernelBinary);
    // if (rc < 0) {
    //     MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "Register kernel bin failed.");
    //     return rc;
    // }
    auto rc = DynamicKernelLaunchEsl(&kArgs);
    if (rc < 0) {
        return rc;
    }
    return rc;
}

int EslModelLauncher::EslModelLaunchAicore(aclrtStream aicoreStream, void *kernel, rtArgsEx_t &rtArgs, rtTaskCfgInfo_t &rtTaskCfg) {
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    auto blockDim = dynamic::GetCfgBlockdim();
    auto ret = rtKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    return ret;
}
}
#endif