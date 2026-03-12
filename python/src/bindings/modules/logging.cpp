/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file logging.cpp
 * \brief Python bindings for logging utilities
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "../bindings.h"
#include "core/logging.h"

namespace py = pybind11;

namespace pypto {

static void logDebug(const std::string &message) {
    LOG_DEBUG << message;
}
static void logInfo(const std::string &message) {
    LOG_INFO << message;
}
static void logWarn(const std::string &message) {
    LOG_WARN << message;
}
static void logError(const std::string &message) {
    LOG_ERROR << message;
}
static void logFatal(const std::string &message) {
    LOG_FATAL << message;
}
static void logEvent(const std::string &message) {
    LOG_EVENT << message;
}
static void check(bool condition, const std::string &message) {
    if (!condition)
        throw pypto::ir::ValueError(message);
}
static void internalCheck(bool condition, const std::string &message) {
    INTERNAL_CHECK(condition) << message;
}

void BindLogging(py::module_ &m) {
    py::enum_<LogLevel>(m, "LogLevel", py::arithmetic())
        .value("DEBUG", LogLevel::DEBUG)
        .value("INFO", LogLevel::INFO)
        .value("WARN", LogLevel::WARN)
        .value("ERROR", LogLevel::ERROR)
        .value("FATAL", LogLevel::FATAL)
        .value("EVENT", LogLevel::EVENT)
        .value("NONE", LogLevel::NONE)
        .export_values();

    m.def("set_log_level", &LoggerManager::ResetLevel, py::arg("level"));
    m.def("log_debug", &logDebug, py::arg("message"));
    m.def("log_info", &logInfo, py::arg("message"));
    m.def("log_warn", &logWarn, py::arg("message"));
    m.def("log_error", &logError, py::arg("message"));
    m.def("log_fatal", &logFatal, py::arg("message"));
    m.def("log_event", &logEvent, py::arg("message"));
    m.def("check", &check, py::arg("condition"), py::arg("message"));
    m.def("internal_check", &internalCheck, py::arg("condition"), py::arg("message"));
}

} // namespace pypto
