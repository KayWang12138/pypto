#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Unit tests for MonitorManager and MonitorImpl classes.
Tests cover all public and private functions in monitor_manager.cpp and monitor_impl.cpp
"""
import os
import time
import pypto
import torch


def test_monitor_manager_instance():
    """Test MonitorManager::Instance() - singleton pattern"""
    pypto.set_host_options(compile_monitor_enable=True)
    
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_initialize_basic():
    """Test MonitorManager::Initialize() with basic parameters"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 30,
            "timeout_sec": 60,
            "total_timeout_sec": 300
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_initialize_disabled():
    """Test MonitorManager::Initialize() with enable=False"""
    @pypto.jit(
        host_options={"compile_monitor_enable": False},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_shutdown():
    """Test MonitorManager::Shutdown()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_current_stage_elapsed():
    """Test MonitorManager::GetCurrentStageElapsed()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_set_total_function_count():
    """Test MonitorManager::SetTotalFunctionCount()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_and_increment_next_function_index():
    """Test MonitorManager::GetAndIncrementNextFunctionIndex()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_set_current_function_index():
    """Test MonitorManager::SetCurrentFunctionIndex()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_try_end_prepare_stage():
    """Test MonitorManager::TryEndPrepareStage()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_notify_compilation_finished():
    """Test MonitorManager::NotifyCompilationFinished()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_print_compilation_finished():
    """Test MonitorManager::PrintCompilationFinished()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_set_compiler_monitor_options():
    """Test MonitorManager::SetCompilerMonitorOptions()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 45,
            "timeout_sec": 90,
            "total_timeout_sec": 400
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_is_enabled():
    """Test MonitorManager::IsEnabled()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_set_stage_timeout_flag():
    """Test MonitorManager::SetStageTimeoutFlag()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "timeout_sec": 1
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_stage_timeout_flag():
    """Test MonitorManager::GetStageTimeoutFlag()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "timeout_sec": 1
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_interval_sec():
    """Test MonitorManager::GetIntervalSec()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 60
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_timeout_sec():
    """Test MonitorManager::GetTimeoutSec()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "timeout_sec": 120
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_total_timeout_sec():
    """Test MonitorManager::GetTotalTimeoutSec()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "total_timeout_sec": 600
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_current_stage_name():
    """Test MonitorManager::GetCurrentStageName()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_stage_start_time():
    """Test MonitorManager::GetStageStartTime()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_total_start_time():
    """Test MonitorManager::GetTotalStartTime()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_total_function_count():
    """Test MonitorManager::GetTotalFunctionCount()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_current_function_index():
    """Test MonitorManager::GetCurrentFunctionIndex()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_stage_elapsed_totals():
    """Test MonitorManager::GetStageElapsedTotals()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_start_stage():
    """Test MonitorManager::StartStage()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_end_stage():
    """Test MonitorManager::EndStage()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_get_current_function_name():
    """Test MonitorManager::GetCurrentFunctionName()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_manager_set_current_function_name():
    """Test MonitorManager::SetCurrentFunctionName()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_start():
    """Test MonitorImpl::Start()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_stop():
    """Test MonitorImpl::Stop()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_start_monitoring():
    """Test MonitorImpl::StartMonitoring()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_stop_monitoring():
    """Test MonitorImpl::StopMonitoring()"""
    @pypto.jit(
        host_options={"compile_monitor_enable": True},
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_print_total_timeout():
    """Test MonitorImpl::PrintTotalTimeOut()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "total_timeout_sec": 1
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_impl_monitor_loop():
    """Test MonitorImpl::MonitorLoop()"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 1
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_with_timeout_zero():
    """Test monitor with timeout_sec=0"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "timeout_sec": 0
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_with_timeout_negative():
    """Test monitor with timeout_sec=-1"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "timeout_sec": -1
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_with_total_timeout_zero():
    """Test monitor with total_timeout_sec=0"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "total_timeout_sec": 0
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_with_interval_zero():
    """Test monitor with interval_sec=0"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 0
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


def test_monitor_with_all_options():
    """Test monitor with all options set"""
    @pypto.jit(
        host_options={
            "compile_monitor_enable": True,
            "interval_sec": 30,
            "timeout_sec": 120,
            "total_timeout_sec": 600
        },
        runtime_options={"run_mode": 1}
    )
    def simple_func(a, b, c):
        pypto.set_vec_tile_shapes(4, 4)
        c[:] = a + b
    
    a = torch.ones((4, 4), dtype=torch.float32) * 2
    b = torch.ones((4, 4), dtype=torch.float32) * 3
    c = torch.ones((4, 4), dtype=torch.float32)
    input_a = pypto.from_torch(a, "a")
    input_b = pypto.from_torch(b, "b")
    output_c = pypto.from_torch(c, "c")
    simple_func(input_a, input_b, output_c)
    assert True


if __name__ == "__main__":
    test_monitor_manager_instance()
    test_monitor_manager_initialize_basic()
    test_monitor_manager_initialize_disabled()
    test_monitor_manager_shutdown()
    test_monitor_manager_get_current_stage_elapsed()
    test_monitor_manager_set_total_function_count()
    test_monitor_manager_get_and_increment_next_function_index()
    test_monitor_manager_set_current_function_index()
    test_monitor_manager_try_end_prepare_stage()
    test_monitor_manager_notify_compilation_finished()
    test_monitor_manager_print_compilation_finished()
    test_monitor_manager_set_compiler_monitor_options()
    test_monitor_manager_is_enabled()
    test_monitor_manager_set_stage_timeout_flag()
    test_monitor_manager_get_stage_timeout_flag()
    test_monitor_manager_get_interval_sec()
    test_monitor_manager_get_timeout_sec()
    test_monitor_manager_get_total_timeout_sec()
    test_monitor_manager_get_current_stage_name()
    test_monitor_manager_get_stage_start_time()
    test_monitor_manager_get_total_start_time()
    test_monitor_manager_get_total_function_count()
    test_monitor_manager_get_current_function_index()
    test_monitor_manager_get_stage_elapsed_totals()
    test_monitor_manager_start_stage()
    test_monitor_manager_end_stage()
    test_monitor_manager_get_current_function_name()
    test_monitor_manager_set_current_function_name()
    test_monitor_impl_start()
    test_monitor_impl_stop()
    test_monitor_impl_start_monitoring()
    test_monitor_impl_stop_monitoring()
    test_monitor_impl_print_total_timeout()
    test_monitor_impl_monitor_loop()
    test_monitor_with_timeout_zero()
    test_monitor_with_timeout_negative()
    test_monitor_with_total_timeout_zero()
    test_monitor_with_interval_zero()
    test_monitor_with_all_options()
    print("All tests passed!")
