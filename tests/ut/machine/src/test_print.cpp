/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "tilefwk/aicore_print.h"

TEST(TestAicorePrint, simple) {
    uint8_t data[4096];
    AicoreLogger logger, logger1;
    logger.Init(data, 4096);
    logger1.Init(data, 4096);

    aicore_printf(logger.context(), "test int %d\n", 1);
    aicore_printf(logger.context(), "test float %f\n", 1.2);
    aicore_printf(logger.context(), "test char %c\n", 'a');
    aicore_printf(logger.context(), "test pointer %p\n", &logger);
    aicore_printf(logger.context(), "test string %s\n", "hello world");
    aicore_printf(logger.context(), "test normal\n");
    aicore_printf(logger.context(), "test normal %%\n");

    char buf[512];
    while (logger1.read(buf, 512)) {
        printf("%s", buf);
    }

    aicore_printf(logger.context(), "test1 int %d\n", 1);
    aicore_printf(logger.context(), "test1 float %f\n", 1.2);
    aicore_printf(logger.context(), "test1 char %c\n", 'a');
    aicore_printf(logger.context(), "test1 pointer %p\n", &logger);
    aicore_printf(logger.context(), "test1 string %s\n", "hello world");
    aicore_printf(logger.context(), "test1 normal\n");
    aicore_printf(logger.context(), "test1 normal %%\n");

    while (logger1.read(buf, 512)) {
        printf("%s", buf);
    }

}

TEST(TestAicorePrint, batch) {
    uint8_t data[16384];
    AicoreLogger logger, logger1;
    logger.Init(data, 16384);
    logger1.Init(data, 16384);

    for (int i = 0; i < 100; i++) {
        aicore_printf(logger.context(), "test int %d\n", i);
        aicore_printf(logger.context(), "test float %f\n", 1.2);
        aicore_printf(logger.context(), "test char %c\n", 'a');
        aicore_printf(logger.context(), "test pointer %p\n", &logger);
        aicore_printf(logger.context(), "test string %s\n", "hello world");
        aicore_printf(logger.context(), "test normal\n");
    }

    char buf[512];
    while (logger1.read(buf, 512)) {
        printf("%s", buf);
    }

    aicore_printf(logger.context(), "test1 int %d\n", 1);
    aicore_printf(logger.context(), "test1 float %f\n", 1.2);
    aicore_printf(logger.context(), "test1 char %c\n", 'a');
    aicore_printf(logger.context(), "test1 pointer %p\n", &logger);
    aicore_printf(logger.context(), "test1 string %s\n", "hello world");
    aicore_printf(logger.context(), "test1 normal\n");
    aicore_printf(logger.context(), "test1 normal %%\n");

    while (logger1.read(buf, 512)) {
        printf("%s", buf);
    }
}
