#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
NPU Communication Library Python Bindings
Based on NVSHMEM design principles for fine-grain NPU communication
"""

import ctypes
import numpy as np
from typing import Optional, Union, Tuple
import threading

# Import pypto_impl for low-level operations
try:
    from . import pypto_impl
except ImportError:
    pypto_impl = None


class NpuCommError(Exception):
    """NPU Communication Library Exception"""
    pass


class NpuCommConfig:
    """NPU Communication Configuration"""

    def __init__(self,
                 num_pes: int = 8,
                 my_pe: int = 0,
                 symmetric_heap_size: int = 1024 * 1024 * 1024,  # 1GB
                 enable_npu_optimization: bool = True,
                 enable_fine_grain_comm: bool = True,
                 enable_pgas_mode: bool = True,
                 max_concurrent_operations: int = 64,
                 max_message_size: int = 64 * 1024):  # 64KB

        self.num_pes = num_pes
        self.my_pe = my_pe
        self.symmetric_heap_size = symmetric_heap_size
        self.enable_npu_optimization = enable_npu_optimization
        self.enable_fine_grain_comm = enable_fine_grain_comm
        self.enable_pgas_mode = enable_pgas_mode
        self.max_concurrent_operations = max_concurrent_operations
        self.max_message_size = max_message_size


class NpuComm:
    """NPU Communication Library Main Interface"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._lock = threading.Lock()
            self._symmetric_heap = {}
            self._config = None
            self._initialized = True

    def init(self, config: NpuCommConfig) -> None:
        """Initialize NPU communication library"""
        with self._lock:
            if self._config is not None:
                raise NpuCommError("NPU communication library already initialized")

            self._config = config
            print(f"[NPUCOMM] Initializing Python bindings for PE {config.my_pe} of {config.num_pes} PEs")

            # Initialize underlying C++ library if available
            if pypto_impl is not None:
                try:
                    # Call C++ initialization through pypto_impl
                    self._cpp_comm = pypto_impl.NpuComm()
                    self._cpp_comm.init(config.num_pes, config.my_pe)
                    print("[NPUCOMM] C++ backend initialized")
                except Exception as e:
                    print(f"[NPUCOMM] Warning: C++ backend not available: {e}")
                    self._cpp_comm = None
            else:
                print("[NPUCOMM] Warning: pypto_impl not available, using pure Python implementation")
                self._cpp_comm = None

    def finalize(self) -> None:
        """Finalize NPU communication library"""
        with self._lock:
            if self._config is None:
                return

            print(f"[NPUCOMM] Finalizing Python bindings for PE {self._config.my_pe}")

            if self._cpp_comm is not None:
                self._cpp_comm.finalize()

            self._symmetric_heap.clear()
            self._config = None

    def shmalloc(self, size: int) -> np.ndarray:
        """Allocate memory in symmetric heap"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if size > self._config.symmetric_heap_size:
            raise NpuCommError(f"Requested size {size} exceeds symmetric heap size {self._config.symmetric_heap_size}")

        # Allocate numpy array as symmetric heap memory
        array = np.zeros(size, dtype=np.uint8)
        addr = array.__array_interface__['data'][0]

        self._symmetric_heap[addr] = array
        print(f"[NPUCOMM] Allocated {size} bytes in symmetric heap")

        return array

    def shfree(self, array: np.ndarray) -> None:
        """Free memory from symmetric heap"""
        addr = array.__array_interface__['data'][0]

        if addr in self._symmetric_heap:
            del self._symmetric_heap[addr]
            print("[NPUCOMM] Freed symmetric heap allocation")
        else:
            raise NpuCommError("Memory not found in symmetric heap")

    def put(self, dest: Union[np.ndarray, int], src: Union[np.ndarray, int],
            size: Optional[int] = None, pe: int = 0) -> None:
        """Single-sided put operation"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if isinstance(dest, np.ndarray) and isinstance(src, np.ndarray):
            if size is None:
                size = min(len(dest), len(src))
            dest_addr = dest.__array_interface__['data'][0]
            src_addr = src.__array_interface__['data'][0]
        else:
            # Handle raw addresses
            dest_addr = dest if isinstance(dest, int) else dest.__array_interface__['data'][0]
            src_addr = src if isinstance(src, int) else src.__array_interface__['data'][0]

        if pe == self._config.my_pe:
            # Local operation
            if self._cpp_comm is not None:
                self._cpp_comm.put(dest_addr, src_addr, size, pe)
            else:
                # Pure Python implementation
                ctypes.memmove(dest_addr, src_addr, size)
        else:
            # Remote operation
            if self._cpp_comm is not None:
                self._cpp_comm.put(dest_addr, src_addr, size, pe)
            else:
                # Simulate remote operation (in real implementation, this would use network)
                print(f"[NPUCOMM] Simulating remote put to PE {pe}: {size} bytes")
                ctypes.memmove(dest_addr, src_addr, size)

    def get(self, dest: Union[np.ndarray, int], src: Union[np.ndarray, int],
            size: Optional[int] = None, pe: int = 0) -> None:
        """Single-sided get operation"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if isinstance(dest, np.ndarray) and isinstance(src, np.ndarray):
            if size is None:
                size = min(len(dest), len(src))
            dest_addr = dest.__array_interface__['data'][0]
            src_addr = src.__array_interface__['data'][0]
        else:
            dest_addr = dest if isinstance(dest, int) else dest.__array_interface__['data'][0]
            src_addr = src if isinstance(src, int) else src.__array_interface__['data'][0]

        if pe == self._config.my_pe:
            # Local operation
            if self._cpp_comm is not None:
                self._cpp_comm.get(dest_addr, src_addr, size, pe)
            else:
                ctypes.memmove(dest_addr, src_addr, size)
        else:
            # Remote operation
            if self._cpp_comm is not None:
                self._cpp_comm.get(dest_addr, src_addr, size, pe)
            else:
                print(f"[NPUCOMM] Simulating remote get from PE {pe}: {size} bytes")
                ctypes.memmove(dest_addr, src_addr, size)

    def atomic_fetch_add(self, dest: Union[np.ndarray, int], value: int, pe: int) -> int:
        """Atomic fetch-and-add operation"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        dest_addr = dest if isinstance(dest, int) else dest.__array_interface__['data'][0]

        if self._cpp_comm is not None:
            return self._cpp_comm.atomic_fetch_add(dest_addr, value, pe)
        else:
            # Pure Python atomic operation (simplified)
            if pe == self._config.my_pe:
                # Access the memory location
                old_value = ctypes.c_int64.from_address(dest_addr).value
                ctypes.c_int64.from_address(dest_addr).value = old_value + value
                return old_value
            else:
                print(f"[NPUCOMM] Simulating remote atomic fetch-add on PE {pe}")
                return 0

    def barrier_all(self) -> None:
        """Global barrier synchronization"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if self._cpp_comm is not None:
            self._cpp_comm.barrier_all()
        else:
            print("[NPUCOMM] Simulating global barrier")
            # In real implementation, this would synchronize across all PEs

    def fence(self) -> None:
        """Memory fence operation"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if self._cpp_comm is not None:
            self._cpp_comm.fence()
        else:
            # Memory fence using threading
            self._lock.acquire()
            self._lock.release()

    def quiet(self) -> None:
        """Wait for all outstanding communications to complete"""
        if self._config is None:
            raise NpuCommError("NPU communication library not initialized")

        if self._cpp_comm is not None:
            self._cpp_comm.quiet()
        else:
            print("[NPUCOMM] Simulating quiet operation")
            # In real implementation, this would wait for all pending operations

    @property
    def my_pe(self) -> int:
        """Get current PE ID"""
        return self._config.my_pe if self._config else -1

    @property
    def num_pes(self) -> int:
        """Get total number of PEs"""
        return self._config.num_pes if self._config else 0


# Global instance
_npucomm_instance = NpuComm()


def init(config: Optional[NpuCommConfig] = None) -> None:
    """Initialize NPU communication library"""
    if config is None:
        config = NpuCommConfig()
    _npucomm_instance.init(config)


def finalize() -> None:
    """Finalize NPU communication library"""
    _npucomm_instance.finalize()


def shmalloc(size: int) -> np.ndarray:
    """Allocate memory in symmetric heap"""
    return _npucomm_instance.shmalloc(size)


def shfree(array: np.ndarray) -> None:
    """Free memory from symmetric heap"""
    _npucomm_instance.shfree(array)


def put(dest: Union[np.ndarray, int], src: Union[np.ndarray, int],
        size: Optional[int] = None, pe: int = 0) -> None:
    """Single-sided put operation"""
    _npucomm_instance.put(dest, src, size, pe)


def get(dest: Union[np.ndarray, int], src: Union[np.ndarray, int],
        size: Optional[int] = None, pe: int = 0) -> None:
    """Single-sided get operation"""
    _npucomm_instance.get(dest, src, size, pe)


def atomic_fetch_add(dest: Union[np.ndarray, int], value: int, pe: int) -> int:
    """Atomic fetch-and-add operation"""
    return _npucomm_instance.atomic_fetch_add(dest, value, pe)


def barrier_all() -> None:
    """Global barrier synchronization"""
    _npucomm_instance.barrier_all()


def fence() -> None:
    """Memory fence operation"""
    _npucomm_instance.fence()


def quiet() -> None:
    """Wait for all outstanding communications to complete"""
    _npucomm_instance.quiet()


def my_pe() -> int:
    """Get current PE ID"""
    return _npucomm_instance.my_pe


def num_pes() -> int:
    """Get total number of PEs"""
    return _npucomm_instance.num_pes

