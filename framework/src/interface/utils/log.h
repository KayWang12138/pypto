/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log.h
 * \brief
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <cstdio>
#include "securec.h"

#define ALOG_DEBUG
#define ALOG_INFO
#define ALOG_WARN
#define ALOG_ERROR
#define ALOG_FATAL
#define ALOG_EVENT

#define ALOG_DEBUG_F(fmt, args...)
#define ALOG_INFO_F(fmt, args...)
#define ALOG_WARN_F(fmt, args...)
#define ALOG_ERROR_F(fmt, args...)
#define ALOG_EVENT_F(fmt, args...)

