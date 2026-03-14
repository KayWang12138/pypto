#!/usr/bin/env python3
# coding: utf-8

from pypto import pypto_impl
from pypto_gpu_vk.runtime import VkRuntime


def test_vk_runtime_scaffold_is_exposed():
    runtime = VkRuntime()
    assert runtime.status in {
        pypto_impl.GpuVkStatus.SUCCESS,
        pypto_impl.GpuVkStatus.UNAVAILABLE,
        pypto_impl.GpuVkStatus.NOT_SUPPORTED,
        pypto_impl.GpuVkStatus.DEVICE_NOT_FOUND,
    }
    assert runtime.available() is (runtime.status == pypto_impl.GpuVkStatus.SUCCESS)
    assert runtime.pipeline_cache_entry_count() == 0
