#!/usr/bin/env python3
# coding: utf-8

from pypto_gpu_vk.runtime import VkRuntime


def test_vk_runtime_scaffold_is_exposed():
    runtime = VkRuntime()
    assert runtime.available() is False
    assert runtime.pipeline_cache_entry_count() == 0
