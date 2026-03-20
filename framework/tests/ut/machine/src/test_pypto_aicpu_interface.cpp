#include <gtest/gtest.h>
#include "machine/device/machine_interface/pypto_aicpu_interface.h"

using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void *args);

TEST(PyptoAicpuInterfaceTest, NullInitArgReturnsError) {
    EXPECT_EQ(DynPyptoKernelServerNull(nullptr), 1U);
}

TEST(PyptoAicpuInterfaceTest, ExecuteBeforeLoadReturnsError) {
    EXPECT_EQ(DynPyptoKernelServer(nullptr), 1U);
    EXPECT_EQ(DynPyptoKernelServerInit(nullptr), 1U);
}
