#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include <fstream>
#include <cstdlib>
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
using namespace std;

class TEST_TILEFWKERFACE_UTest : public testing::Test {};


TEST_F(TEST_TILEFWKERFACE_UTest, staticFwkServel) {
  std::cout << "staticFwkServel start" << std::endl;
  npu::tile_fwk::DeviceArgs devArgs;
  devArgs.aicpuSoLen = 1;
  StaticPyptoKernelServer(&devArgs);
  std::cout << "staticFwkServel end" << std::endl;
}

TEST_F(TEST_TILEFWKERFACE_UTest, staticFwkServel01) {
  std::cout << "staticFwkServel start" << std::endl;
  npu::tile_fwk::DeviceArgs devArgs;
  devArgs.aicpuSoLen = 0;
  StaticPyptoKernelServer(&devArgs);
  std::cout << "staticFwkServel end" << std::endl;
}

TEST_F(TEST_TILEFWKERFACE_UTest, dynFwkServel) {
  std::cout << "dync FwkServel start" << std::endl;
  DynPyptoKernelServer(nullptr);
  std::cout << "dync FwkServel end" << std::endl;
}


TEST_F(TEST_TILEFWKERFACE_UTest, dynFwkServelInit) {
  std::cout << "dynFwkServelInit start" << std::endl;
  npu::tile_fwk::AstKernelArgs astArgs;
  npu::tile_fwk::DeviceArgs devArgs;
  devArgs.aicpuSoLen = 1;
  astArgs.cfgdata = reinterpret_cast<int64_t *>(&devArgs);
  DynPyptoKernelServerInit(&astArgs);
  std::cout << "dynFwkServelInit end" << std::endl;
}

TEST_F(TEST_TILEFWKERFACE_UTest, dynFwkServelInit01) {
  std::cout << "dynFwkServelInit start" << std::endl;
  npu::tile_fwk::AstKernelArgs astArgs;
  npu::tile_fwk::DeviceArgs devArgs;
  devArgs.aicpuSoLen = 0;
  astArgs.cfgdata = reinterpret_cast<int64_t *>(&devArgs);
  DynPyptoKernelServerInit(&astArgs);
  std::cout << "dynFwkServelInit end" << std::endl;
}
