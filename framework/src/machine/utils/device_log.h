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
#include "securec.h"
#include "machine/utils/device_switch.h"
#ifdef __DEVICE__
#include "dlog_pub.h"
#endif
namespace npu::tile_fwk {

#define DEV_IF_NONDEVICE                                                        \
    if constexpr (!IsDeviceMode())

#define DEV_IF_DEVICE                                                           \
    if constexpr (IsDeviceMode())

#define DEV_IF_DEBUG                                                        \
    if (IsDebugMode())

#define DEV_IF_VERBOSE_DEBUG                                            \
    if constexpr (IsCompileVerboseLog())

#if ENABLE_TMP_LOG == 0
#define DEBUG_PLOG 1
#else
#define DEBUG_PLOG 0
#endif/*DEBUG_PLOG*/

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
    (void)logfile;
    (void)level;
    thread_local DeviceLogger devLogger(level);
#if ENABLE_TMP_LOG || !defined(__DEVICE__)
    if (logfile != nullptr) {
        devLogger.SetLogFile(logfile);
    }
#endif
    return devLogger;
}

typedef enum {
    LOG_TYPE_SCHEDULER,    // 调度器日志
    LOG_TYPE_CONTROLLER,   // 控制器日志
    LOG_TYPE_PREFETCH      // 预取日志
} LogType;

// 创建日志文件
inline void CreateLogFile(LogType type, int threadIdx) {
    (void)type;
    (void)threadIdx;
#if ENABLE_TMP_LOG || !defined(__DEVICE__)
    char logfile[256];
    switch (type) {
        case LOG_TYPE_SCHEDULER:
            (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_sch%d.txt", threadIdx);
            break;
        case LOG_TYPE_CONTROLLER:
            (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_ctrl.txt");
            break;
        case LOG_TYPE_PREFETCH:
            (void)sprintf_s(logfile, sizeof(logfile), "/tmp/tile_fwk_aicpu_prefetch.txt");
            break;
        default:
            return;
    }
    GetLogger(logfile);
#endif
}

#if DEBUG_PLOG && defined(__DEVICE__)
#define GET_TID() syscall(__NR_gettid)
const std::string TILE_FWK_DEVICE_MACHINE = "AI_CPU";

extern bool g_isLogDEnable;
extern bool g_isLogIEnable;
extern bool g_isLogWEnable;
extern bool g_isLogEEnable;

inline bool IsDebugMode() {
    return g_isLogDEnable;
}

void InitLogSwitch();

#define D_DEV_LOGD(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (g_isLogDEnable) {                                                  \
        dlog_debug(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while (false)

#define D_DEV_LOGI(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (g_isLogIEnable) {                                                   \
        dlog_info(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGW(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
      if (g_isLogWEnable) {                                                   \
        dlog_warn(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);   \
      }                                                                               \
  } while(false)

#define D_DEV_LOGE(MODE_NAME, fmt, ...)                                               \
  do {                                                                                \
    if (g_isLogEEnable) {                                                  \
        dlog_error(AICPU, "%lu %s\n" #fmt , GET_TID(), __FUNCTION__, ##__VA_ARGS__);  \
      }                                                                               \
  } while(false)

#define DEV_DEBUG(fmt, args...) D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_INFO(fmt, args...) D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...) D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_VERBOSE_DEBUG(fmt, args...)                                  \
  do {                                                                  \
    if constexpr (IsCompileVerboseLog())  {                          \
        D_DEV_LOGD(TILE_FWK_DEVICE_MACHINE, fmt, ##args);               \
    }                                                                   \
  } while(0)


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

inline constexpr bool IsCompileVerboseLog() {
#if ENABLE_COMPILE_VERBOSE_LOG
    return true;
#else
    return false;
#endif
}

#else

inline bool IsDebugMode() {
    return true;
}

#ifdef CONFIG_BAREMETAL

#define DEV_DEBUG(fmt, args...)
#define DEV_INFO(fmt, args...)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...)
#define DEV_MEM_DUMP(fmt, args...)

#else

#if ENABLE_TMP_LOG
inline constexpr bool IsCompileVerboseLog() {
    return true;
}

#define DEV_DEBUG(fmt, args...) GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args)
#define DEV_INFO(fmt, args...) GetLogger().Log(LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##args)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...) GetLogger().Log(LOG_LEVEL_WARN, __FILE__, __LINE__, fmt, ##args)
#define DEV_VERBOSE_DEBUG(fmt, args...)  GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args)
#else
inline constexpr bool IsCompileVerboseLog() {
    return false;
}

#define DEV_DEBUG(fmt, args...)
#define DEV_INFO(fmt, args...)
#define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#define DEV_WARN(fmt, args...)
#define DEV_VERBOSE_DEBUG(fmt, args...)
#endif

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


#define DEV_DEBUG_ASSERT(expr) DEV_ASSERT(expr)
#define DEV_DEBUG_ASSERT_MSG(expr, fmt, args...) DEV_ASSERT_MSG(expr, fmt, ##args)
#endif // DEBUG_PLOG
} // namespace npu::tile_fwk