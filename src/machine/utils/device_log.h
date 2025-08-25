/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_log.h
 * \brief
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <ctime>
#include <cassert>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "machine/utils/device_switch.h"
#ifdef __DEVICE__
#include "toolchain/slog.h"
#endif
namespace npu::tile_fwk {

#define DEV_IF_NONDEVICE                                                        \
    if constexpr (!IsDeviceMode())

#if DEBUG_PLOG && defined(__DEVICE__)
#define GET_TID() syscall(__NR_gettid)
const std::string TILE_FWK_DEVICE_MACHINE = "AI_CPU";

bool IsLogDEnable();
bool IsLogIEnable();
bool IsLogWEnable();
bool IsLogEEnable();

#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogDEnable()) {                                                  \
        dlog_debug(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogIEnable()) {                                                   \
        dlog_info(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogWEnable()) {                                                   \
        dlog_warn(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (IsLogEEnable()) {                                                  \
        dlog_error(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while(false)

#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...) D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...) D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

inline int CheckDebug() {
    return CheckLogLevel(AICPU, DLOG_DEBUG);
}

#define DEV_ASSERT_MSG(expr, fmt, args...)                                                   \
    do {                                                                                     \
        if (!(expr)) {                                                                       \
            DEV_ERROR(fmt, ##args);                                                          \
            assert(0);                                                                       \
        }                                                                                    \
    } while (0)

#define DEV_ASSERT(expr)                                                       \
    do {                                                                       \
        if (!(expr)) {                                                         \
            assert(0);                                                         \
        }                                                                      \
    } while (0)

#define DEV_DEBUG_ASSERT(expr)                                                 \
    do {                                                                       \
        if (!(expr)) {                                                         \
            DEV_ERROR("assert failed: %s, %d", #expr, __FILE__, __LINE__);     \
            assert(0);                                                         \
        }                                                                      \
    } while (0)
    
#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...) DEV_ASSERT_MSG(expr, fmt, ##args)

#define DEV_MEM_DUMP(fmt, args...)

#else

constexpr int LOG_LEVEL_DEBUG = 0;
constexpr int LOG_LEVEL_INFO = 1;
constexpr int LOG_LEVEL_WARN = 2;
constexpr int LOG_LEVEL_ERROR = 3;

static const char *g_levelName[] = {"DEBUG", "INFO", "WARN", "ERROR"};

class DeviceLogger {
public:
    explicit DeviceLogger(int level = LOG_LEVEL_INFO) : level_(level){};

    int Level() const { return level_; }

    void Log(int level, const char *file, int line, const char *fmt, ...) const __attribute__((format(printf, 5, 6))) {
        if (level < level_) {
            return;
        }

        struct timeval tv;
        gettimeofday(&tv, nullptr);
        const char *fileName = strrchr(file, '/');
        if (fileName != nullptr) {
            file = fileName + 1;
        }
        va_list ap;
        va_start(ap, fmt);
        if (fp_) {
            fprintf(fp_, "%ld.%06ld [%s] %s:%d", tv.tv_sec, tv.tv_usec, g_levelName[level], file, line);
            vfprintf(fp_, fmt, ap);
            fprintf(fp_, "\n");
        } else {
            printf("%ld.%06ld [%s] %s:%d", tv.tv_sec, tv.tv_usec, g_levelName[level], file, line);
            vprintf(fmt, ap);
            printf("\n");
        }
        va_end(ap);
        Flush();
    }

    void Flush() const {
        if (fp_) {
            fflush(fp_);
        }
    }

    void SetLogFile(const char *logfile) {
        if (strcmp(logfile, logfile_.c_str()) == 0) {
            return;
        }

        if (fp_) {
            fclose(fp_);
        }

        fp_ = fopen(logfile, "wb+");
        logfile_ = logfile;
    }

    ~DeviceLogger() {
        if (fp_) {
            fclose(fp_);
        }
    }

private:
    FILE *fp_{nullptr};
    std::string logfile_;
    int level_;
};

inline DeviceLogger &GetLogger(const char *logfile = nullptr, int level = LOG_LEVEL_DEBUG) {
    thread_local DeviceLogger devLogger(level);
    if (logfile != nullptr) {
        devLogger.SetLogFile(logfile);
    }
    return devLogger;
}
#ifdef CONFIG_BAREMETAL

#define DEV_DEBUG(fmt, args...)
#define DEV_INFO(fmt, args...)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...)
#define DEV_MEM_DUMP(fmt, args...)

#else

#if defined(DEBUG_SWITCH) && DEBUG_SWITCH

#define DEV_DEBUG(fmt, args...) GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args)
#define DEV_INFO(fmt, args...) GetLogger().Log(LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##args)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...) GetLogger().Log(LOG_LEVEL_WARN, __FILE__, __LINE__, fmt, ##args)

#else

#define DEV_DEBUG(fmt, args...)
#define DEV_INFO(fmt, args...)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...)

#endif // DEBUG_SWITCH

#if DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE
#define DEV_MEM_DUMP(fmt, args...) GetLogger().Log(LOG_LEVEL_DEBUG, "/memdump", 0, "[WsMem Statistics] " fmt, ##args)
#else
#define DEV_MEM_DUMP(fmt, args...)
#endif // DEBUG_MEM_DUMP_LEVEL != DEBUG_MEM_DUMP_DISABLE

#endif // CONFIG_BAREMETAL

#define DEV_ASSERT_MSG(expr, fmt, args...)                                                   \
    do {                                                                                     \
        if (!(expr)) {                                                                       \
            GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, "%s :" fmt, #expr, ##args); \
            GetLogger().Flush();                                                             \
            assert(0);                                                                       \
        }                                                                                    \
    } while (0)

#define DEV_ASSERT(expr)                                                       \
    do {                                                                       \
        if (!(expr)) {                                                         \
            GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, "%s", #expr); \
            GetLogger().Flush();                                               \
            assert(0);                                                         \
        }                                                                      \
    } while (0)

#if DEBUG_SWITCH
#define DEV_DEBUG_ASSERT(expr) DEV_ASSERT(expr)
#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...) DEV_ASSERT_MSG(expr, fmt, ##args)
#else
#define DEV_DEBUG_ASSERT(expr)
#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...)
#endif // DEBUG_SWITCH
#endif // DEBUG_PLOG
} // namespace npu::tile_fwk