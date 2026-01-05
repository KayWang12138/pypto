# PyPTO 日志机制分析报告

> **适用对象：** 框架开发者、维护者、性能优化人员  
> **分析时间：** 2026年1月  
> **分析方法：** Sequential Thinking 系统性分析  
> **学习目标：** 理解当前日志架构、识别问题、制定改进方案

---

## 目录

1. [日志模块概览](#1-日志模块概览)
2. [日志架构详解](#2-日志架构详解)
3. [问题识别](#3-问题识别)
4. [问题汇总与优先级](#4-问题汇总与优先级)
5. [短期改进方案](#5-短期改进方案)
6. [中期改进方案](#6-中期改进方案)
7. [长期改进方案](#7-长期改进方案)
8. [实施路线图](#8-实施路线图)

---

## 1. 日志模块概览

### 1.1 C++ Framework 层

PyPTO Framework 层存在 **4 套独立的日志系统**：

| 日志模块 | 文件位置 | 命名空间 | 职责 |
|---------|---------|---------|------|
| **主日志系统** | `framework/src/interface/utils/log.h` | `npu::tile_fwk` | 编译器/接口层日志 |
| **Pass日志** | `framework/src/passes/pass_log/pass_log.h` | `npu::tile_fwk` | Pass优化阶段日志 |
| **设备日志** | `framework/src/machine/utils/device_log.h` | `npu::tile_fwk` | 设备端运行时日志 |
| **代价模型日志** | `framework/src/cost_model/simulation/base/ModelLogger.h` | `CostModel` | 代价模型仿真日志 |

### 1.2 Python 层

| 日志方式 | 文件位置 | 使用方式 |
|---------|---------|---------|
| **Python logging** | `python/pypto/_controller.py` | `logging.basicConfig()` + `logging.error()` |
| **诊断系统** | `python/pypto/frontend/parser/diagnostics.py` | 自定义 `DiagnosticLevel` + 源码上下文 |
| **异常处理** | `python/pypto/frontend/parser/error.py` | `ParserError` + `RenderedParserError` |

### 1.3 日志使用统计

```
C++ 层日志使用情况:
├── ALOG_* 宏使用: 341 处（分布在 20 个文件）
├── DEV_* 宏使用: ~100 处
├── MLOG_* 宏使用: ~50 处
└── printf/std::cout: 25 个文件中存在

Python 层日志使用情况:
├── logging.* 调用: 4 处
├── raise Exception: 279 处（32 个文件）
└── DiagnosticLevel: 1 个文件专用
```

---

## 2. 日志架构详解

### 2.1 主日志系统（`log.h`）

```cpp
// 日志级别定义
enum class LoggerLevel {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL,
    EVENT,
    NONE,
};

// 核心组件
class LoggerManager {  // 单例模式
    std::mutex logMtx;
    LoggerLevel level{LoggerLevel::ERROR};  // 默认ERROR级别
    bool stdEnabled{true};
    StdLogger stdLogger;
    std::unordered_map<std::string, std::unique_ptr<FileLogger>> fileLoggerDict;
    std::unordered_map<std::string, std::shared_ptr<LineLogger>> lineLoggerDict;
    
    static LoggerManager &GetManager();
    static void ResetLevel(LoggerLevel l);
    static void FileLoggerRegister(const std::string &filepath, bool append);
};

// 日志类
class Logger {
    std::stringstream ss;      // 普通输出
    std::stringstream ssRich;  // 带颜色输出
    LoggerLevel level;
    bool enableLog;
    
    Logger(LoggerLevel levelIn, const std::string &func, int line);
    ~Logger();  // 析构时输出日志
    
    template<typename T> Logger& Log(T&& val);
    template<typename T> Logger& operator<<(T&& val);
};

// 宏定义
#define ALOG_DEBUG ALOG_LEVEL(npu::tile_fwk::LoggerLevel::DEBUG)
#define ALOG_INFO  ALOG_LEVEL(npu::tile_fwk::LoggerLevel::INFO)
#define ALOG_WARN  ALOG_LEVEL(npu::tile_fwk::LoggerLevel::WARN)
#define ALOG_ERROR ALOG_LEVEL(npu::tile_fwk::LoggerLevel::ERROR)
#define ALOG_FATAL ALOG_LEVEL(npu::tile_fwk::LoggerLevel::FATAL)
#define ALOG_EVENT ALOG_LEVEL(npu::tile_fwk::LoggerLevel::EVENT)

// printf 风格宏
#define ALOG_DEBUG_F(fmt, args...) ALOG_F(DEBUG, fmt, ##args)
#define ALOG_INFO_F(fmt, args...)  ALOG_F(INFO, fmt, ##args)
```

**特点**：
- 支持标准输出、文件输出、行缓存输出
- 支持终端颜色（TTY_RED, TTY_GREEN 等）
- 线程安全（使用 mutex）
- 默认日志级别为 ERROR

### 2.2 设备日志系统（`device_log.h`）

```cpp
// 简单的整数级别常量
constexpr int LOG_LEVEL_DEBUG = 0;
constexpr int LOG_LEVEL_INFO = 1;
constexpr int LOG_LEVEL_WARN = 2;
constexpr int LOG_LEVEL_ERROR = 3;

// 日志类型枚举
enum class LogType {
    LOG_TYPE_SCHEDULER,    // 调度器日志
    LOG_TYPE_CONTROLLER,   // 控制器日志
    LOG_TYPE_PREFETCH      // 预取日志
};

// 设备日志类
class DeviceLogger {
    FILE *fp_{nullptr};
    std::string logfile_;
    int level_;
    
    void Log(int level, const char *file, int line, const char *fmt, ...);
    void SetLogFile(const char *logfile);
};

// 条件编译控制
#if DEBUG_PLOG && defined(__DEVICE__)
    // 设备端使用华为 dlog 系统
    #define DEV_DEBUG(fmt, args...) dlog_debug(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##args)
    #define DEV_INFO(fmt, args...)  dlog_info(AICPU, ...)
    #define DEV_WARN(fmt, args...)  dlog_warn(AICPU, ...)
    #define DEV_ERROR(fmt, args...) dlog_error(AICPU, ...)
#else
    // 主机端使用本地 Logger
    #define DEV_DEBUG(fmt, args...) GetLogger().Log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##args)
    #define DEV_INFO(fmt, args...)  GetLogger().Log(LOG_LEVEL_INFO, __FILE__, __LINE__, fmt, ##args)
    #define DEV_WARN(fmt, args...)  GetLogger().Log(LOG_LEVEL_WARN, __FILE__, __LINE__, fmt, ##args)
    #define DEV_ERROR(fmt, args...) GetLogger().Log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##args)
#endif

// 断言宏
#define DEV_ASSERT(expr) ...
#define DEV_ASSERT_MSG(expr, fmt, args...) ...
```

**特点**：
- 通过编译宏区分设备端和主机端
- 设备端使用华为 dlog 系统
- 支持 backtrace 打印
- 与主日志系统完全独立

### 2.3 代价模型日志（`ModelLogger.h`）

```cpp
namespace CostModel {

// 注意：枚举顺序与主日志系统不同！
enum class LoggerLevel {
    NONE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL,
    DEFAULT = INFO,
};

class LoggerManager {
    LoggerLevel level{LoggerLevel::DEFAULT};  // 默认 INFO 级别
    std::map<int, CostModel::LoggerLevel> levelMap = {
        {1, LoggerLevel::DEBUG},
        {2, LoggerLevel::INFO},
        {3, LoggerLevel::WARN},
        {4, LoggerLevel::ERROR},
        {5, LoggerLevel::FATAL},
    };
};

// 宏定义
#define MLOG_DEBUG CostModel::Logger(CostModel::LoggerLevel::DEBUG, __func__, __LINE__)
#define MLOG_INFO  CostModel::Logger(CostModel::LoggerLevel::INFO, __func__, __LINE__)
#define MLOG_WARN  CostModel::Logger(CostModel::LoggerLevel::WARN, __func__, __LINE__)
#define MLOG_ERROR CostModel::Logger(CostModel::LoggerLevel::ERROR, __func__, __LINE__)
#define MLOG_FATAL CostModel::Logger(CostModel::LoggerLevel::FATAL, __func__, __LINE__)
}
```

**特点**：
- 独立的命名空间 `CostModel`
- 枚举值顺序与主日志系统不同（NONE=0 vs DEBUG=0）
- 默认日志级别为 INFO（与主系统 ERROR 不同）
- 代码结构与主日志系统相似但独立维护

### 2.4 Pass 日志（`pass_log.h`）

```cpp
// 依赖主日志系统
#include "interface/utils/log.h"

// 元素类型枚举
enum class Elements {
    Operation,
    Tensor,
    Function,
    Graph,
    Config,
    Manager
};

// Pass 日志宏（需要定义 MODULE_NAME）
#define APASS_LOG_F(lvl, MODULE_NAME, opName, fmt, args...) \
    ALOG_F(lvl, "[%s][%s][" #lvl "]: " fmt, MODULE_NAME, opName, ##args)

#define APASS_LOG_DEBUG_F(opEnum, fmt, args...) APASS_LOG_F(DEBUG, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_INFO_F(opEnum, fmt, args...)  APASS_LOG_F(INFO, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_WARN_F(opEnum, fmt, args...)  APASS_LOG_F(WARN, MODULE_NAME, toString(opEnum), fmt, ##args)
#define APASS_LOG_ERROR_F(opEnum, fmt, args...) APASS_LOG_F(ERROR, MODULE_NAME, toString(opEnum), fmt, ##args)
```

**特点**：
- 基于主日志系统扩展
- 需要每个文件定义 `MODULE_NAME` 宏
- 提供 backtrace 函数用于操作追踪

### 2.5 Python 层日志

```python
# _controller.py
import logging
logging.basicConfig(level=logging.DEBUG)  # 硬编码 DEBUG 级别

# 使用示例
logging.error("Record function %s failed: %s", name, e)
logging.error("Record loop function %s failed: %s", name, e)

# error.py - 异常处理
PTO_BACKTRACE_ENV_VAR = "PTO_BACKTRACE"

class ParserError(Exception):
    def __init__(self, node: doc.AST, msg: Union[str, Exception]):
        ...

# diagnostics.py - 诊断系统
class DiagnosticLevel(enum.IntEnum):
    BUG = 10
    ERROR = 20
    WARNING = 30
    INFO = 40
    DEBUG = 50
```

### 2.6 日志级别控制

```cpp
// program.cpp - 通过环境变量控制主日志级别
void GetEnv(const char * const envName, std::string &envValue);

Program::Program() {
    std::string envLogLevel;
    GetEnv("GLOBAL_LOG_LEVEL", envLogLevel);
    if (!envLogLevel.empty()) {
        int32_t logLevel = std::stoi(envLogLevel);
        LoggerManager::GetManager().ResetLevel(static_cast<LoggerLevel>(logLevel));
    }
}
```

---

## 3. 问题识别

### 3.1 问题 1：多套独立的日志系统，缺乏统一

**问题描述**：
- 存在 4 套独立的 C++ 日志系统
- 各系统有不同的日志级别枚举、不同的默认级别
- 代码维护困难，日志格式不一致
- 无法统一控制日志级别

**证据**：

| 日志系统 | 枚举定义 | 默认级别 |
|---------|---------|---------|
| `log.h` | `DEBUG=0, INFO, WARN, ERROR, FATAL, EVENT, NONE` | ERROR |
| `ModelLogger.h` | `NONE=0, DEBUG, INFO, WARN, ERROR, FATAL` | INFO |
| `device_log.h` | `int` 常量 `0, 1, 2, 3` | DEBUG/ERROR(条件编译) |

### 3.2 问题 2：日志级别控制机制不完善

**问题描述**：
- 环境变量 `GLOBAL_LOG_LEVEL` 只影响 `LoggerManager`
- 设备日志通过编译宏控制，无运行时控制
- Python 层日志级别硬编码为 DEBUG

**证据**：

```python
# _controller.py:25 - 硬编码 DEBUG 级别
logging.basicConfig(level=logging.DEBUG)
```

```cpp
// device_log.h:173-178 - 编译时决定日志开关
#if ENABLE_TMP_LOG
static inline bool IsLogEnableDebug() { return true; }
#else
static inline bool IsLogEnableDebug() { return false; }
#endif
```

### 3.3 问题 3：日志格式不一致

**问题描述**：不同日志系统输出格式完全不同，难以统一分析。

| 日志系统 | 格式示例 |
|---------|---------|
| `log.h` | `2025-01-05 12:00:00.123 I \| message` |
| `device_log.h` | `1704456000.123456 [INFO] file.cpp:42 message` |
| `ModelLogger.h` | 无时间戳，直接输出消息 |
| Python logging | `DEBUG:root:message` |

### 3.4 问题 4：设备端日志与主机端日志不统一

**问题描述**：
- 设备端使用华为 `dlog` 系统
- 主机端使用自定义 `Logger`
- 日志无法统一收集和分析
- 调试时需要查看多个日志输出

### 3.5 问题 5：Python 层日志使用不规范

**问题描述**：
- 只有 3 处使用 `logging`，其他地方直接 `raise`
- 缺少统一的日志配置
- 无法方便地调整日志级别

**证据**：

| 文件 | 代码 |
|------|------|
| `_controller.py:318` | `logging.error("Record function %s failed: %s", name, e)` |
| `_controller.py:413` | `logging.error("Record loop function %s failed: %s", name, e)` |
| `error.py:80` | `logging.info("note: run with...")` |

### 3.6 问题 6：Pass 日志需要手动定义 MODULE_NAME

**问题描述**：每个使用 Pass 日志的文件都需要定义 `MODULE_NAME` 宏，容易遗漏或不一致。

### 3.7 问题 7：日志文件管理功能缺失

**缺失功能**：
- 日志轮转（Log Rotation）
- 日志压缩
- 日志清理
- 异步写入

### 3.8 问题 8：错误处理与日志混合

**问题描述**：存在多种断言方式，断言失败信息可能不会记录到日志。

```cpp
// 多种断言方式
DEV_ASSERT(expr);           // device_log.h
DEV_ASSERT_MSG(expr, fmt);  // device_log.h
ASSERT(expr);               // tilefwk/error.h (另一套)
```

### 3.9 问题 9：缺少结构化日志

**问题描述**：所有日志都是文本格式，缺少：
- JSON 格式输出
- 日志字段标注
- 日志追踪 ID（trace ID）
- 性能指标日志

### 3.10 问题 10：环境变量控制分散

| 环境变量 | 用途 | 定义位置 |
|---------|------|---------|
| `GLOBAL_LOG_LEVEL` | 主日志级别 | `program.cpp` |
| `TILEFWK_CONFIG_PATH` | 配置文件路径 | `config_manager.cpp` |
| `PTO_BACKTRACE` | Python 异常回溯 | `error.py` |
| `ASCEND_HOME_PATH` | 设备路径 | 多处使用 |

---

## 4. 问题汇总与优先级

| 优先级 | 问题 | 影响范围 | 修复难度 |
|-------|------|---------|---------|
| **P0** | 多套独立日志系统 | 全项目 | 高 |
| **P0** | 日志级别控制不统一 | 调试困难 | 中 |
| **P1** | Python/C++ 日志不协调 | 跨语言调试 | 中 |
| **P1** | 日志格式不一致 | 日志分析 | 中 |
| **P1** | 设备端/主机端日志分离 | 问题定位 | 高 |
| **P2** | Pass 日志需要手动定义 MODULE_NAME | 代码维护 | 低 |
| **P2** | 缺少日志文件管理 | 生产环境 | 中 |
| **P2** | 缺少结构化日志 | 日志分析 | 中 |
| **P3** | 环境变量控制分散 | 文档维护 | 低 |
| **P3** | 断言与日志分离 | 错误追踪 | 低 |

---

## 5. 短期改进方案

### 5.1 统一 Python 日志配置

**目标**：创建统一的日志配置模块，支持环境变量控制。

**实施方案**：

创建 `python/pypto/logging_config.py`：

```python
#!/usr/bin/env python3
# coding: utf-8
"""PyPTO 统一日志配置模块。

通过环境变量控制日志级别和输出格式。

环境变量:
    PYPTO_LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    PYPTO_LOG_FORMAT: 日志格式 (simple, detailed, json)
    PYPTO_LOG_FILE: 日志文件路径（可选）
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import Optional

# 默认配置
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "detailed"

# 日志格式模板
LOG_FORMATS = {
    "simple": "%(levelname)s: %(message)s",
    "detailed": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    "json": None,  # 使用自定义 JSON 格式化器
}


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器。"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class PyptoLogger:
    """PyPTO 日志管理器。"""
    
    _initialized = False
    _logger_cache: dict = {}
    
    @classmethod
    def setup(cls, 
              level: Optional[str] = None,
              format_type: Optional[str] = None,
              log_file: Optional[str] = None) -> None:
        """初始化日志系统。
        
        Parameters
        ----------
        level : str, optional
            日志级别，默认从环境变量 PYPTO_LOG_LEVEL 读取
        format_type : str, optional
            日志格式，默认从环境变量 PYPTO_LOG_FORMAT 读取
        log_file : str, optional
            日志文件路径，默认从环境变量 PYPTO_LOG_FILE 读取
        """
        if cls._initialized:
            return
            
        # 从环境变量或参数获取配置
        level = level or os.environ.get("PYPTO_LOG_LEVEL", DEFAULT_LOG_LEVEL)
        format_type = format_type or os.environ.get("PYPTO_LOG_FORMAT", DEFAULT_LOG_FORMAT)
        log_file = log_file or os.environ.get("PYPTO_LOG_FILE")
        
        # 获取日志级别
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # 创建根日志器
        root_logger = logging.getLogger("pypto")
        root_logger.setLevel(log_level)
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 创建格式化器
        if format_type == "json":
            formatter = JsonFormatter()
        else:
            fmt = LOG_FORMATS.get(format_type, LOG_FORMATS["detailed"])
            formatter = logging.Formatter(fmt)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 添加文件处理器（如果指定）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str = "pypto") -> logging.Logger:
        """获取日志器。
        
        Parameters
        ----------
        name : str
            日志器名称
            
        Returns
        -------
        logging.Logger
            日志器实例
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._logger_cache:
            logger = logging.getLogger(name)
            cls._logger_cache[name] = logger
        
        return cls._logger_cache[name]
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """动态设置日志级别。
        
        Parameters
        ----------
        level : str
            日志级别
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger("pypto").setLevel(log_level)


# 便捷函数
def get_logger(name: str = "pypto") -> logging.Logger:
    """获取 PyPTO 日志器。"""
    return PyptoLogger.get_logger(name)


def setup_logging(**kwargs) -> None:
    """初始化日志系统。"""
    PyptoLogger.setup(**kwargs)


# 模块导入时自动初始化
PyptoLogger.setup()
```

**修改 `_controller.py`**：

```python
# 移除原有的 logging 配置
# import logging
# logging.basicConfig(level=logging.DEBUG)

# 使用统一日志配置
from .logging_config import get_logger

_logger = get_logger("pypto.controller")

# 使用示例
_logger.error("Record function %s failed: %s", name, e)
```

### 5.2 统一环境变量文档

**目标**：创建环境变量参考文档。

创建 `docs/note/05-debugging/05-environment-variables.md`（见附录）。

### 5.3 添加日志级别映射

**目标**：在主日志系统中添加字符串到枚举的映射。

修改 `framework/src/interface/utils/log.h`：

```cpp
// 添加在 LoggerLevel 枚举定义后
inline LoggerLevel ParseLogLevel(const std::string& levelStr) {
    static const std::unordered_map<std::string, LoggerLevel> levelMap = {
        {"DEBUG", LoggerLevel::DEBUG},
        {"INFO", LoggerLevel::INFO},
        {"WARN", LoggerLevel::WARN},
        {"WARNING", LoggerLevel::WARN},
        {"ERROR", LoggerLevel::ERROR},
        {"FATAL", LoggerLevel::FATAL},
        {"EVENT", LoggerLevel::EVENT},
        {"NONE", LoggerLevel::NONE},
    };
    
    std::string upper = levelStr;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    auto it = levelMap.find(upper);
    return (it != levelMap.end()) ? it->second : LoggerLevel::ERROR;
}

inline const char* LogLevelToString(LoggerLevel level) {
    static const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL", "EVENT", "NONE"};
    return names[static_cast<int>(level)];
}
```

---

## 6. 中期改进方案

### 6.1 创建统一日志门面（Logging Facade）

**目标**：创建统一的日志接口层，屏蔽底层实现差异。

**架构设计**：

```
┌─────────────────────────────────────────────────────────────┐
│                     统一日志门面 (LogFacade)                  │
├─────────────────────────────────────────────────────────────┤
│  LOG_DEBUG(fmt, ...)  LOG_INFO(fmt, ...)  LOG_ERROR(fmt, ...)│
│  LOG_WARN(fmt, ...)   LOG_FATAL(fmt, ...)                   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   主日志系统     │  │   设备日志系统   │  │  代价模型日志   │
│   (log.h)       │  │ (device_log.h) │  │ (ModelLogger.h)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**实现方案**：

创建 `framework/src/interface/utils/log_facade.h`：

```cpp
/**
 * @file log_facade.h
 * @brief 统一日志门面 - 提供跨模块一致的日志接口
 */

#pragma once

#include <string>
#include <cstdarg>
#include <memory>

namespace npu::tile_fwk {

/**
 * @brief 统一日志级别枚举
 */
enum class UnifiedLogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5,
    OFF = 6
};

/**
 * @brief 日志输出目标
 */
enum class LogTarget {
    CONSOLE = 1 << 0,
    FILE = 1 << 1,
    SYSTEM = 1 << 2,  // 系统日志（如 dlog）
    ALL = CONSOLE | FILE | SYSTEM
};

/**
 * @brief 日志上下文信息
 */
struct LogContext {
    const char* file;
    int line;
    const char* function;
    const char* module;
    
    LogContext(const char* f, int l, const char* fn, const char* m = nullptr)
        : file(f), line(l), function(fn), module(m) {}
};

/**
 * @brief 日志后端接口
 */
class ILogBackend {
public:
    virtual ~ILogBackend() = default;
    virtual void Log(UnifiedLogLevel level, const LogContext& ctx, const std::string& message) = 0;
    virtual void SetLevel(UnifiedLogLevel level) = 0;
    virtual UnifiedLogLevel GetLevel() const = 0;
    virtual void Flush() = 0;
};

/**
 * @brief 日志门面单例
 */
class LogFacade {
public:
    static LogFacade& Instance();
    
    // 注册日志后端
    void RegisterBackend(const std::string& name, std::shared_ptr<ILogBackend> backend);
    void UnregisterBackend(const std::string& name);
    
    // 设置全局日志级别
    void SetGlobalLevel(UnifiedLogLevel level);
    UnifiedLogLevel GetGlobalLevel() const;
    
    // 设置特定模块的日志级别
    void SetModuleLevel(const std::string& module, UnifiedLogLevel level);
    
    // 日志输出
    void Log(UnifiedLogLevel level, const LogContext& ctx, const char* fmt, ...);
    void LogV(UnifiedLogLevel level, const LogContext& ctx, const char* fmt, va_list args);
    
    // 刷新所有后端
    void Flush();
    
    // 从环境变量初始化
    void InitFromEnv();
    
private:
    LogFacade();
    ~LogFacade();
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 便捷宏定义
#define LOG_CTX npu::tile_fwk::LogContext(__FILE__, __LINE__, __FUNCTION__)
#define LOG_CTX_M(module) npu::tile_fwk::LogContext(__FILE__, __LINE__, __FUNCTION__, module)

#define ULOG_TRACE(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::TRACE, LOG_CTX, fmt, ##__VA_ARGS__)
#define ULOG_DEBUG(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::DEBUG, LOG_CTX, fmt, ##__VA_ARGS__)
#define ULOG_INFO(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::INFO, LOG_CTX, fmt, ##__VA_ARGS__)
#define ULOG_WARN(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::WARN, LOG_CTX, fmt, ##__VA_ARGS__)
#define ULOG_ERROR(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::ERROR, LOG_CTX, fmt, ##__VA_ARGS__)
#define ULOG_FATAL(fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::FATAL, LOG_CTX, fmt, ##__VA_ARGS__)

// 带模块名的日志宏
#define ULOG_DEBUG_M(module, fmt, ...) \
    npu::tile_fwk::LogFacade::Instance().Log(npu::tile_fwk::UnifiedLogLevel::DEBUG, LOG_CTX_M(module), fmt, ##__VA_ARGS__)

} // namespace npu::tile_fwk
```

创建 `framework/src/interface/utils/log_facade.cpp`：

```cpp
/**
 * @file log_facade.cpp
 * @brief 统一日志门面实现
 */

#include "log_facade.h"
#include "log.h"

#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <algorithm>

namespace npu::tile_fwk {

struct LogFacade::Impl {
    std::mutex mutex_;
    UnifiedLogLevel globalLevel_{UnifiedLogLevel::INFO};
    std::unordered_map<std::string, std::shared_ptr<ILogBackend>> backends_;
    std::unordered_map<std::string, UnifiedLogLevel> moduleLevels_;
    
    bool ShouldLog(UnifiedLogLevel level, const std::string& module) const {
        if (level < globalLevel_) return false;
        
        if (!module.empty()) {
            auto it = moduleLevels_.find(module);
            if (it != moduleLevels_.end() && level < it->second) {
                return false;
            }
        }
        return true;
    }
};

LogFacade& LogFacade::Instance() {
    static LogFacade instance;
    return instance;
}

LogFacade::LogFacade() : impl_(std::make_unique<Impl>()) {
    InitFromEnv();
}

LogFacade::~LogFacade() = default;

void LogFacade::RegisterBackend(const std::string& name, std::shared_ptr<ILogBackend> backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->backends_[name] = std::move(backend);
}

void LogFacade::UnregisterBackend(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->backends_.erase(name);
}

void LogFacade::SetGlobalLevel(UnifiedLogLevel level) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->globalLevel_ = level;
    
    // 同步到所有后端
    for (auto& [name, backend] : impl_->backends_) {
        backend->SetLevel(level);
    }
}

UnifiedLogLevel LogFacade::GetGlobalLevel() const {
    return impl_->globalLevel_;
}

void LogFacade::SetModuleLevel(const std::string& module, UnifiedLogLevel level) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->moduleLevels_[module] = level;
}

void LogFacade::Log(UnifiedLogLevel level, const LogContext& ctx, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    LogV(level, ctx, fmt, args);
    va_end(args);
}

void LogFacade::LogV(UnifiedLogLevel level, const LogContext& ctx, const char* fmt, va_list args) {
    std::string module = ctx.module ? ctx.module : "";
    
    if (!impl_->ShouldLog(level, module)) {
        return;
    }
    
    // 格式化消息
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    std::string message(buffer);
    
    // 输出到所有后端
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    for (auto& [name, backend] : impl_->backends_) {
        backend->Log(level, ctx, message);
    }
}

void LogFacade::Flush() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    for (auto& [name, backend] : impl_->backends_) {
        backend->Flush();
    }
}

void LogFacade::InitFromEnv() {
    // 从 PYPTO_LOG_LEVEL 环境变量读取日志级别
    const char* levelEnv = std::getenv("PYPTO_LOG_LEVEL");
    if (levelEnv) {
        std::string levelStr(levelEnv);
        std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(), ::toupper);
        
        static const std::unordered_map<std::string, UnifiedLogLevel> levelMap = {
            {"TRACE", UnifiedLogLevel::TRACE},
            {"DEBUG", UnifiedLogLevel::DEBUG},
            {"INFO", UnifiedLogLevel::INFO},
            {"WARN", UnifiedLogLevel::WARN},
            {"WARNING", UnifiedLogLevel::WARN},
            {"ERROR", UnifiedLogLevel::ERROR},
            {"FATAL", UnifiedLogLevel::FATAL},
            {"OFF", UnifiedLogLevel::OFF},
        };
        
        auto it = levelMap.find(levelStr);
        if (it != levelMap.end()) {
            impl_->globalLevel_ = it->second;
        }
    }
    
    // 兼容旧的 GLOBAL_LOG_LEVEL 环境变量
    const char* globalLevelEnv = std::getenv("GLOBAL_LOG_LEVEL");
    if (globalLevelEnv) {
        int level = std::atoi(globalLevelEnv);
        if (level >= 0 && level <= 6) {
            impl_->globalLevel_ = static_cast<UnifiedLogLevel>(level);
        }
    }
}

} // namespace npu::tile_fwk
```

### 6.2 实现日志轮转

**目标**：支持日志文件大小限制和自动轮转。

创建 `framework/src/interface/utils/rotating_file_logger.h`：

```cpp
/**
 * @file rotating_file_logger.h
 * @brief 支持日志轮转的文件日志器
 */

#pragma once

#include "log_facade.h"
#include <string>
#include <fstream>
#include <mutex>
#include <queue>

namespace npu::tile_fwk {

/**
 * @brief 日志轮转配置
 */
struct RotatingConfig {
    std::string baseFilename;     // 基础文件名
    size_t maxFileSize;           // 单文件最大大小（字节）
    size_t maxFileCount;          // 最大文件数量
    bool compress;                // 是否压缩旧文件
    
    RotatingConfig()
        : maxFileSize(10 * 1024 * 1024)  // 默认 10MB
        , maxFileCount(5)
        , compress(false)
    {}
};

/**
 * @brief 支持轮转的文件日志后端
 */
class RotatingFileBackend : public ILogBackend {
public:
    explicit RotatingFileBackend(const RotatingConfig& config);
    ~RotatingFileBackend() override;
    
    void Log(UnifiedLogLevel level, const LogContext& ctx, const std::string& message) override;
    void SetLevel(UnifiedLogLevel level) override { level_ = level; }
    UnifiedLogLevel GetLevel() const override { return level_; }
    void Flush() override;
    
private:
    void RotateIfNeeded();
    void DoRotate();
    std::string GetRotatedFilename(int index) const;
    void CompressFile(const std::string& filename);
    
    RotatingConfig config_;
    UnifiedLogLevel level_{UnifiedLogLevel::INFO};
    std::ofstream file_;
    size_t currentSize_{0};
    std::mutex mutex_;
};

/**
 * @brief 异步日志后端（提高性能）
 */
class AsyncLogBackend : public ILogBackend {
public:
    explicit AsyncLogBackend(std::shared_ptr<ILogBackend> backend, size_t queueSize = 8192);
    ~AsyncLogBackend() override;
    
    void Log(UnifiedLogLevel level, const LogContext& ctx, const std::string& message) override;
    void SetLevel(UnifiedLogLevel level) override;
    UnifiedLogLevel GetLevel() const override;
    void Flush() override;
    
private:
    void WorkerThread();
    
    struct LogEntry {
        UnifiedLogLevel level;
        LogContext ctx;
        std::string message;
    };
    
    std::shared_ptr<ILogBackend> backend_;
    std::queue<LogEntry> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{true};
};

} // namespace npu::tile_fwk
```

### 6.3 添加 JSON 格式输出

**目标**：支持结构化 JSON 日志输出，便于日志分析工具处理。

创建 `framework/src/interface/utils/json_logger.h`：

```cpp
/**
 * @file json_logger.h
 * @brief JSON 格式日志后端
 */

#pragma once

#include "log_facade.h"
#include <string>

namespace npu::tile_fwk {

/**
 * @brief JSON 日志格式化器
 */
class JsonFormatter {
public:
    /**
     * @brief 格式化日志为 JSON 字符串
     * 
     * 输出格式:
     * {
     *   "timestamp": "2025-01-05T12:00:00.123Z",
     *   "level": "INFO",
     *   "module": "pass",
     *   "file": "pass_manager.cpp",
     *   "line": 42,
     *   "function": "RunPass",
     *   "message": "Starting optimization pass",
     *   "extra": { ... }
     * }
     */
    static std::string Format(
        UnifiedLogLevel level,
        const LogContext& ctx,
        const std::string& message,
        const std::unordered_map<std::string, std::string>& extra = {}
    );
    
    static std::string EscapeJson(const std::string& str);
    static std::string GetIsoTimestamp();
};

/**
 * @brief JSON 日志后端
 */
class JsonLogBackend : public ILogBackend {
public:
    explicit JsonLogBackend(std::shared_ptr<ILogBackend> wrappedBackend);
    
    void Log(UnifiedLogLevel level, const LogContext& ctx, const std::string& message) override;
    void SetLevel(UnifiedLogLevel level) override;
    UnifiedLogLevel GetLevel() const override;
    void Flush() override;
    
    // 添加额外字段
    void AddField(const std::string& key, const std::string& value);
    void RemoveField(const std::string& key);
    
private:
    std::shared_ptr<ILogBackend> wrapped_;
    std::unordered_map<std::string, std::string> extraFields_;
    std::mutex mutex_;
};

} // namespace npu::tile_fwk
```

### 6.4 统一 Pass 日志

**目标**：自动获取模块名，无需手动定义 MODULE_NAME。

修改 `framework/src/passes/pass_log/pass_log.h`：

```cpp
/**
 * @file pass_log.h
 * @brief Pass 模块统一日志接口
 */

#pragma once

#include "interface/utils/log_facade.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"

namespace npu::tile_fwk {

// 自动获取 Pass 模块名的宏
#define PASS_MODULE_NAME() GetPassModuleName(__FILE__)

inline std::string GetPassModuleName(const char* file) {
    std::string path(file);
    
    // 从文件路径提取模块名
    // 例如: "passes/tile_graph_pass/graph_partition.cpp" -> "tile_graph_pass"
    auto pos = path.rfind("passes/");
    if (pos != std::string::npos) {
        auto start = pos + 7;  // strlen("passes/")
        auto end = path.find('/', start);
        if (end != std::string::npos) {
            return path.substr(start, end - start);
        }
    }
    
    // 回退：使用文件名
    pos = path.rfind('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

// 新的 Pass 日志宏（自动获取模块名）
#define PLOG_DEBUG(fmt, ...) \
    ULOG_DEBUG_M(PASS_MODULE_NAME().c_str(), fmt, ##__VA_ARGS__)
#define PLOG_INFO(fmt, ...) \
    ULOG_INFO_M(PASS_MODULE_NAME().c_str(), fmt, ##__VA_ARGS__)
#define PLOG_WARN(fmt, ...) \
    ULOG_WARN_M(PASS_MODULE_NAME().c_str(), fmt, ##__VA_ARGS__)
#define PLOG_ERROR(fmt, ...) \
    ULOG_ERROR_M(PASS_MODULE_NAME().c_str(), fmt, ##__VA_ARGS__)

// 带操作信息的日志
#define PLOG_OP_DEBUG(op, fmt, ...) \
    PLOG_DEBUG("[%s] " fmt, GetOpName(op).c_str(), ##__VA_ARGS__)
#define PLOG_OP_INFO(op, fmt, ...) \
    PLOG_INFO("[%s] " fmt, GetOpName(op).c_str(), ##__VA_ARGS__)
#define PLOG_OP_ERROR(op, fmt, ...) \
    PLOG_ERROR("[%s] " fmt, GetOpName(op).c_str(), ##__VA_ARGS__)

// 兼容旧接口（逐步废弃）
#define APASS_LOG_DEBUG_F PLOG_DEBUG
#define APASS_LOG_INFO_F PLOG_INFO
#define APASS_LOG_WARN_F PLOG_WARN
#define APASS_LOG_ERROR_F PLOG_ERROR

} // namespace npu::tile_fwk
```

---

## 7. 长期改进方案

### 7.1 引入 spdlog 统一 C++ 日志

**目标**：使用成熟的 spdlog 库替代自定义日志系统。

**优势**：
- 高性能（异步日志、零拷贝格式化）
- 功能完善（轮转、压缩、多后端）
- 社区活跃，文档完善
- 格式化语法现代化（fmt 库）

**实施方案**：

**步骤 1：添加 spdlog 依赖**

修改 `framework/CMakeLists.txt`：

```cmake
# 添加 spdlog 依赖
include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

target_link_libraries(tile_fwk_compiler PRIVATE spdlog::spdlog)
```

**步骤 2：创建 spdlog 包装层**

创建 `framework/src/interface/utils/spdlog_backend.h`：

```cpp
/**
 * @file spdlog_backend.h
 * @brief spdlog 日志后端实现
 */

#pragma once

#include "log_facade.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/async.h>

namespace npu::tile_fwk {

/**
 * @brief spdlog 日志后端
 */
class SpdlogBackend : public ILogBackend {
public:
    /**
     * @brief 创建 spdlog 后端
     * 
     * @param name 日志器名称
     * @param config 配置选项
     */
    struct Config {
        bool enableConsole = true;
        bool enableFile = false;
        std::string logFile;
        size_t maxFileSize = 10 * 1024 * 1024;  // 10MB
        size_t maxFiles = 5;
        bool async = true;
        size_t asyncQueueSize = 8192;
        std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v";
    };
    
    explicit SpdlogBackend(const std::string& name, const Config& config = {});
    ~SpdlogBackend() override;
    
    void Log(UnifiedLogLevel level, const LogContext& ctx, const std::string& message) override;
    void SetLevel(UnifiedLogLevel level) override;
    UnifiedLogLevel GetLevel() const override;
    void Flush() override;
    
    // 获取底层 spdlog logger（用于高级操作）
    std::shared_ptr<spdlog::logger> GetLogger() const { return logger_; }
    
private:
    static spdlog::level::level_enum ToSpdlogLevel(UnifiedLogLevel level);
    static UnifiedLogLevel FromSpdlogLevel(spdlog::level::level_enum level);
    
    std::shared_ptr<spdlog::logger> logger_;
    Config config_;
};

/**
 * @brief 全局 spdlog 初始化
 */
class SpdlogInitializer {
public:
    static void Initialize(const SpdlogBackend::Config& config = {});
    static void Shutdown();
    static std::shared_ptr<spdlog::logger> GetLogger(const std::string& name = "pypto");
    
private:
    static std::once_flag initFlag_;
    static bool initialized_;
};

// 便捷宏（使用 spdlog 原生格式化）
#define SLOG_TRACE(...) SPDLOG_LOGGER_TRACE(SpdlogInitializer::GetLogger(), __VA_ARGS__)
#define SLOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(SpdlogInitializer::GetLogger(), __VA_ARGS__)
#define SLOG_INFO(...)  SPDLOG_LOGGER_INFO(SpdlogInitializer::GetLogger(), __VA_ARGS__)
#define SLOG_WARN(...)  SPDLOG_LOGGER_WARN(SpdlogInitializer::GetLogger(), __VA_ARGS__)
#define SLOG_ERROR(...) SPDLOG_LOGGER_ERROR(SpdlogInitializer::GetLogger(), __VA_ARGS__)
#define SLOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(SpdlogInitializer::GetLogger(), __VA_ARGS__)

} // namespace npu::tile_fwk
```

**步骤 3：迁移计划**

```cpp
// 阶段 1：新代码使用 spdlog
// 新增的 Pass 和模块使用 SLOG_* 宏

// 阶段 2：逐步替换旧代码
// 使用脚本批量替换 ALOG_* -> SLOG_*
// 替换规则：
//   ALOG_DEBUG << msg -> SLOG_DEBUG("{}", msg)
//   ALOG_INFO_F(fmt, args) -> SLOG_INFO(fmt, args)

// 阶段 3：移除旧日志系统
// 删除 log.h 中的 Logger 类
// 保留兼容宏指向 spdlog
```

### 7.2 实现分布式日志追踪

**目标**：支持跨进程、跨设备的日志关联和追踪。

**架构设计**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        日志追踪系统                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Host 进程   │    │ Device 进程 │    │ AICPU 进程  │         │
│  │ trace_id: A │    │ trace_id: A │    │ trace_id: A │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                    ┌───────▼───────┐                           │
│                    │  日志收集器   │                           │
│                    │  (Collector)  │                           │
│                    └───────┬───────┘                           │
│                            │                                    │
│                    ┌───────▼───────┐                           │
│                    │  日志存储     │                           │
│                    │  (Storage)    │                           │
│                    └───────┬───────┘                           │
│                            │                                    │
│                    ┌───────▼───────┐                           │
│                    │  日志分析     │                           │
│                    │  (Analysis)   │                           │
│                    └───────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**实现方案**：

创建 `framework/src/interface/utils/trace_context.h`：

```cpp
/**
 * @file trace_context.h
 * @brief 分布式追踪上下文
 */

#pragma once

#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>

namespace npu::tile_fwk {

/**
 * @brief 追踪上下文
 * 
 * 用于关联跨进程、跨设备的日志
 */
class TraceContext {
public:
    /**
     * @brief 追踪 ID 结构
     * 
     * 格式: {timestamp_ms}-{random}-{sequence}
     * 示例: 1704456000123-a1b2c3d4-0001
     */
    struct TraceId {
        uint64_t timestamp;
        uint32_t random;
        uint32_t sequence;
        
        std::string ToString() const;
        static TraceId FromString(const std::string& str);
        static TraceId Generate();
    };
    
    /**
     * @brief Span 表示一个操作的执行范围
     */
    struct Span {
        TraceId traceId;
        std::string spanId;
        std::string parentSpanId;
        std::string name;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point endTime;
        
        void End();
        uint64_t DurationUs() const;
    };
    
    // 获取当前线程的追踪上下文
    static TraceContext& Current();
    
    // 开始新的追踪
    void StartTrace(const std::string& name);
    void StartTrace(const TraceId& traceId, const std::string& name);
    
    // 创建子 Span
    Span StartSpan(const std::string& name);
    void EndSpan();
    
    // 获取当前追踪信息
    TraceId GetTraceId() const { return traceId_; }
    std::string GetCurrentSpanId() const;
    
    // 传播追踪上下文（用于跨进程传递）
    std::string Serialize() const;
    void Deserialize(const std::string& data);
    
    // 清除当前追踪
    void Clear();
    
private:
    TraceId traceId_;
    std::vector<Span> spanStack_;
    static thread_local TraceContext instance_;
};

/**
 * @brief RAII 风格的 Span 管理
 */
class ScopedSpan {
public:
    explicit ScopedSpan(const std::string& name)
        : span_(TraceContext::Current().StartSpan(name)) {}
    
    ~ScopedSpan() {
        TraceContext::Current().EndSpan();
    }
    
    TraceContext::Span& Get() { return span_; }
    
private:
    TraceContext::Span span_;
};

// 便捷宏
#define TRACE_SPAN(name) npu::tile_fwk::ScopedSpan _trace_span_##__LINE__(name)
#define TRACE_FUNCTION() TRACE_SPAN(__FUNCTION__)

// 带追踪信息的日志宏
#define TLOG_INFO(fmt, ...) \
    ULOG_INFO("[trace:%s] " fmt, TraceContext::Current().GetTraceId().ToString().c_str(), ##__VA_ARGS__)
#define TLOG_ERROR(fmt, ...) \
    ULOG_ERROR("[trace:%s] " fmt, TraceContext::Current().GetTraceId().ToString().c_str(), ##__VA_ARGS__)

} // namespace npu::tile_fwk
```

### 7.3 添加性能指标日志

**目标**：记录编译和执行过程中的性能指标。

创建 `framework/src/interface/utils/metrics_logger.h`：

```cpp
/**
 * @file metrics_logger.h
 * @brief 性能指标日志系统
 */

#pragma once

#include <string>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace npu::tile_fwk {

/**
 * @brief 指标类型
 */
enum class MetricType {
    COUNTER,    // 累计计数器
    GAUGE,      // 瞬时值
    HISTOGRAM,  // 直方图
    TIMER       // 计时器
};

/**
 * @brief 指标值
 */
struct MetricValue {
    MetricType type;
    double value;
    std::chrono::steady_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> labels;
};

/**
 * @brief 性能指标收集器
 */
class MetricsCollector {
public:
    static MetricsCollector& Instance();
    
    // 计数器操作
    void IncrementCounter(const std::string& name, double value = 1.0,
                         const std::unordered_map<std::string, std::string>& labels = {});
    
    // 瞬时值操作
    void SetGauge(const std::string& name, double value,
                  const std::unordered_map<std::string, std::string>& labels = {});
    
    // 直方图操作
    void RecordHistogram(const std::string& name, double value,
                         const std::unordered_map<std::string, std::string>& labels = {});
    
    // 计时器操作
    void RecordTimer(const std::string& name, std::chrono::microseconds duration,
                     const std::unordered_map<std::string, std::string>& labels = {});
    
    // 导出指标（JSON 格式）
    std::string ExportJson() const;
    
    // 导出指标（Prometheus 格式）
    std::string ExportPrometheus() const;
    
    // 重置所有指标
    void Reset();
    
private:
    MetricsCollector() = default;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<MetricValue>> metrics_;
};

/**
 * @brief RAII 风格的计时器
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name,
                const std::unordered_map<std::string, std::string>& labels = {})
        : name_(name)
        , labels_(labels)
        , start_(std::chrono::steady_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        MetricsCollector::Instance().RecordTimer(name_, duration, labels_);
    }
    
private:
    std::string name_;
    std::unordered_map<std::string, std::string> labels_;
    std::chrono::steady_clock::time_point start_;
};

// 便捷宏
#define METRIC_TIMER(name) npu::tile_fwk::ScopedTimer _metric_timer_##__LINE__(name)
#define METRIC_TIMER_LABELED(name, ...) \
    npu::tile_fwk::ScopedTimer _metric_timer_##__LINE__(name, {__VA_ARGS__})

#define METRIC_COUNTER(name, value) \
    npu::tile_fwk::MetricsCollector::Instance().IncrementCounter(name, value)
#define METRIC_GAUGE(name, value) \
    npu::tile_fwk::MetricsCollector::Instance().SetGauge(name, value)

/**
 * @brief 预定义的编译器指标名称
 */
namespace Metrics {
    // 编译阶段指标
    constexpr const char* COMPILE_TIME_TOTAL = "pypto_compile_time_total_us";
    constexpr const char* COMPILE_TIME_PASS = "pypto_compile_time_pass_us";
    constexpr const char* COMPILE_PASS_COUNT = "pypto_compile_pass_count";
    
    // 内存指标
    constexpr const char* MEMORY_ALLOCATED = "pypto_memory_allocated_bytes";
    constexpr const char* MEMORY_PEAK = "pypto_memory_peak_bytes";
    
    // 操作指标
    constexpr const char* OPERATION_COUNT = "pypto_operation_count";
    constexpr const char* FUNCTION_COUNT = "pypto_function_count";
    
    // 缓存指标
    constexpr const char* CACHE_HIT = "pypto_cache_hit_count";
    constexpr const char* CACHE_MISS = "pypto_cache_miss_count";
}

} // namespace npu::tile_fwk
```

### 7.4 Python/C++ 日志统一

**目标**：实现 Python 和 C++ 层日志的统一管理。

创建 `python/pypto/native_logging.py`：

```python
#!/usr/bin/env python3
# coding: utf-8
"""Python/C++ 日志统一接口。

通过 pybind11 绑定，实现 Python 和 C++ 层日志的统一控制。
"""

from typing import Optional, Dict, Any
import logging
import os

from . import pypto_impl


class NativeLogBridge:
    """Python 和 C++ 日志桥接器。"""
    
    # 日志级别映射
    LEVEL_MAP = {
        "TRACE": 0,
        "DEBUG": 1,
        "INFO": 2,
        "WARN": 3,
        "WARNING": 3,
        "ERROR": 4,
        "FATAL": 5,
        "CRITICAL": 5,
        "OFF": 6,
    }
    
    _initialized = False
    
    @classmethod
    def initialize(cls, 
                   level: Optional[str] = None,
                   log_file: Optional[str] = None,
                   enable_json: bool = False) -> None:
        """初始化统一日志系统。
        
        Parameters
        ----------
        level : str, optional
            日志级别
        log_file : str, optional
            日志文件路径
        enable_json : bool
            是否启用 JSON 格式
        """
        if cls._initialized:
            return
        
        # 从环境变量获取配置
        level = level or os.environ.get("PYPTO_LOG_LEVEL", "INFO")
        log_file = log_file or os.environ.get("PYPTO_LOG_FILE")
        enable_json = enable_json or os.environ.get("PYPTO_LOG_JSON", "0") == "1"
        
        # 转换为数字级别
        numeric_level = cls.LEVEL_MAP.get(level.upper(), 2)
        
        # 初始化 C++ 日志系统
        pypto_impl.InitializeLogging(numeric_level, log_file or "", enable_json)
        
        # 配置 Python logging 以转发到 C++
        cls._setup_python_bridge(level)
        
        cls._initialized = True
    
    @classmethod
    def _setup_python_bridge(cls, level: str) -> None:
        """设置 Python 日志桥接到 C++。"""
        
        class NativeHandler(logging.Handler):
            """转发 Python 日志到 C++ 的处理器。"""
            
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    msg = self.format(record)
                    native_level = {
                        logging.DEBUG: 1,
                        logging.INFO: 2,
                        logging.WARNING: 3,
                        logging.ERROR: 4,
                        logging.CRITICAL: 5,
                    }.get(record.levelno, 2)
                    
                    pypto_impl.LogMessage(
                        native_level,
                        record.pathname,
                        record.lineno,
                        record.funcName or "",
                        msg
                    )
                except Exception:
                    self.handleError(record)
        
        # 配置 pypto logger
        logger = logging.getLogger("pypto")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        
        handler = NativeHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """动态设置日志级别。"""
        numeric_level = cls.LEVEL_MAP.get(level.upper(), 2)
        pypto_impl.SetLogLevel(numeric_level)
        logging.getLogger("pypto").setLevel(getattr(logging, level.upper(), logging.INFO))
    
    @classmethod
    def flush(cls) -> None:
        """刷新所有日志缓冲区。"""
        pypto_impl.FlushLogs()


# 便捷函数
def init_logging(**kwargs) -> None:
    """初始化统一日志系统。"""
    NativeLogBridge.initialize(**kwargs)


def set_log_level(level: str) -> None:
    """设置日志级别。"""
    NativeLogBridge.set_level(level)


def flush_logs() -> None:
    """刷新日志缓冲区。"""
    NativeLogBridge.flush()
```

---

## 8. 实施路线图

### 8.1 短期（1-2 周）

| 任务 | 优先级 | 工作量 | 负责人 |
|-----|-------|-------|-------|
| 创建 Python 统一日志配置模块 | P0 | 2天 | - |
| 创建环境变量文档 | P0 | 1天 | - |
| 添加日志级别字符串映射 | P1 | 1天 | - |
| 修改 `_controller.py` 使用新日志 | P1 | 1天 | - |

### 8.2 中期（1-2 月）

| 任务 | 优先级 | 工作量 | 负责人 |
|-----|-------|-------|-------|
| 实现统一日志门面 | P0 | 1周 | - |
| 实现日志轮转功能 | P1 | 3天 | - |
| 添加 JSON 格式输出 | P1 | 3天 | - |
| 统一 Pass 日志 | P2 | 3天 | - |
| 更新所有 Pass 使用新日志 | P2 | 1周 | - |

### 8.3 长期（3-6 月）

| 任务 | 优先级 | 工作量 | 负责人 |
|-----|-------|-------|-------|
| 引入 spdlog 库 | P1 | 2周 | - |
| 迁移现有代码到 spdlog | P1 | 4周 | - |
| 实现分布式追踪 | P2 | 2周 | - |
| 实现性能指标日志 | P2 | 2周 | - |
| Python/C++ 日志统一 | P2 | 2周 | - |
| 移除旧日志系统 | P3 | 2周 | - |

### 8.4 里程碑

```
Week 1-2:   [短期] Python 日志统一 + 文档
Week 3-6:   [中期] 日志门面 + 轮转 + JSON
Week 7-10:  [中期] Pass 日志统一
Week 11-18: [长期] spdlog 集成 + 迁移
Week 19-24: [长期] 追踪 + 指标 + 统一
```

---

## 附录

### A. 环境变量参考

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `PYPTO_LOG_LEVEL` | `INFO` | 全局日志级别 (TRACE/DEBUG/INFO/WARN/ERROR/FATAL/OFF) |
| `PYPTO_LOG_FILE` | - | 日志文件路径 |
| `PYPTO_LOG_FORMAT` | `detailed` | 日志格式 (simple/detailed/json) |
| `PYPTO_LOG_JSON` | `0` | 启用 JSON 格式 (0/1) |
| `GLOBAL_LOG_LEVEL` | `3` | C++ 日志级别（数字，已废弃，建议使用 PYPTO_LOG_LEVEL） |
| `PTO_BACKTRACE` | `0` | 显示 Python 异常回溯 (0/1) |
| `TILEFWK_CONFIG_PATH` | - | 配置文件路径 |

### B. 日志级别对照表

| 级别 | 数值 | Python | C++ log.h | C++ device_log.h | spdlog |
|-----|------|--------|-----------|-----------------|--------|
| TRACE | 0 | - | - | - | trace |
| DEBUG | 1 | DEBUG(10) | DEBUG(0) | 0 | debug |
| INFO | 2 | INFO(20) | INFO(1) | 1 | info |
| WARN | 3 | WARNING(30) | WARN(2) | 2 | warn |
| ERROR | 4 | ERROR(40) | ERROR(3) | 3 | error |
| FATAL | 5 | CRITICAL(50) | FATAL(4) | - | critical |
| OFF | 6 | - | NONE(6) | - | off |

---

## 总结

PyPTO 当前的日志系统存在碎片化、不一致等问题，通过本文提出的短期、中期、长期改进方案，可以逐步实现：

1. **统一的日志接口**：通过日志门面屏蔽底层实现差异
2. **灵活的日志控制**：支持环境变量、API 动态调整日志级别
3. **丰富的输出格式**：支持文本、JSON、结构化日志
4. **完善的日志管理**：支持轮转、压缩、异步写入
5. **跨语言统一**：Python 和 C++ 日志统一管理
6. **可观测性支持**：分布式追踪、性能指标

建议按照路线图分阶段实施，优先解决 P0 级问题，逐步完善日志系统。

