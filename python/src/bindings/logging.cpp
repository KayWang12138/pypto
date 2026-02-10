/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "bindings.h"
#include "core/logging.h"

namespace py = pybind11;

namespace pypto {

static void log_debug(const std::string &message) { LOG_DEBUG << message; }
static void log_info(const std::string &message) { LOG_INFO << message; }
static void log_warn(const std::string &message) { LOG_WARN << message; }
static void log_error(const std::string &message) { LOG_ERROR << message; }
static void log_fatal(const std::string &message) { LOG_FATAL << message; }
static void log_event(const std::string &message) { LOG_EVENT << message; }
static void check(bool condition, const std::string &message) {
  if (!condition) throw pypto::ir::ValueError(message);
}
static void internal_check(bool condition, const std::string &message) { INTERNAL_CHECK(condition) << message; }

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
  m.def("log_debug", &log_debug, py::arg("message"));
  m.def("log_info", &log_info, py::arg("message"));
  m.def("log_warn", &log_warn, py::arg("message"));
  m.def("log_error", &log_error, py::arg("message"));
  m.def("log_fatal", &log_fatal, py::arg("message"));
  m.def("log_event", &log_event, py::arg("message"));
  m.def("check", &check, py::arg("condition"), py::arg("message"));
  m.def("internal_check", &internal_check, py::arg("condition"), py::arg("message"));
}

} // namespace pypto
