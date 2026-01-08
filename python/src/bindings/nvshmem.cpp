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
 * \file nvshmem.cpp
 * \brief NVSHMEM Python Bindings
 */

#include "pybind_common.h"
#include "bindings.h"
#include "models/npucomm/nvshmem.h"

namespace pypto {

// NVSHMEM Library Management
void BindNVSHMEM_Library(py::module& m) {
    m.def("nvshmem_init", &nvshmem_init, "Initialize NVSHMEM library");
    m.def("nvshmem_init_thread", [](int requested) {
        int provided;
        int status = nvshmem_init_thread(requested, &provided);
        return std::make_tuple(status, provided);
    }, "Initialize NVSHMEM with thread support");

    m.def("nvshmem_my_pe", &nvshmem_my_pe, "Get current PE ID");
    m.def("nvshmem_n_pes", &nvshmem_n_pes, "Get number of PEs");

    m.def("nvshmem_finalize", &nvshmem_finalize, "Finalize NVSHMEM library");
    m.def("nvshmem_global_exit", &nvshmem_global_exit, "Global exit");

    m.def("nvshmem_info_get_version", &nvshmem_info_get_version, "Get NVSHMEM version");
    m.def("nvshmem_info_get_name", &nvshmem_info_get_name, "Get NVSHMEM name");
}

// NVSHMEM Memory Management
void BindNVSHMEM_Memory(py::module& m) {
    m.def("nvshmem_malloc", &nvshmem_malloc, "Allocate memory");
    m.def("nvshmem_calloc", &nvshmem_calloc, "Allocate zero-initialized memory");
    m.def("nvshmem_align", &nvshmem_align, "Allocate aligned memory");
    m.def("nvshmem_free", &nvshmem_free, "Free memory");

    m.def("nvshmem_shmalloc", &nvshmem_shmalloc, "Allocate symmetric heap memory");
    m.def("nvshmem_shfree", &nvshmem_shfree, "Free symmetric heap memory");
    m.def("nvshmem_shmemalign", &nvshmem_shmemalign, "Allocate aligned symmetric heap memory");
    m.def("nvshmem_shrealloc", &nvshmem_shrealloc, "Reallocate symmetric heap memory");
}

// NVSHMEM Remote Memory Access
void BindNVSHMEM_RMA(py::module& m) {
    // Basic put/get operations
    m.def("nvshmem_put", [](py::buffer dest, py::buffer src, int pe) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t size = src_info.size * src_info.itemsize;
        nvshmem_put(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr), size, pe);
    }, "Put operation");

    m.def("nvshmem_get", [](py::buffer dest, py::buffer src, int pe) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t size = src_info.size * src_info.itemsize;
        nvshmem_get(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr), size, pe);
    }, "Get operation");

    // Non-blocking operations
    m.def("nvshmem_put_nbi", [](py::buffer dest, py::buffer src, int pe) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t size = src_info.size * src_info.itemsize;
        nvshmem_put_nbi(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr), size, pe);
    }, "Non-blocking put operation");

    m.def("nvshmem_get_nbi", [](py::buffer dest, py::buffer src, int pe) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t size = src_info.size * src_info.itemsize;
        nvshmem_get_nbi(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr), size, pe);
    }, "Non-blocking get operation");

    // Typed operations
    m.def("nvshmem_put8", &nvshmem_put8, "8-bit put operation");
    m.def("nvshmem_put32", &nvshmem_put32, "32-bit put operation");
    m.def("nvshmem_put64", &nvshmem_put64, "64-bit put operation");

    m.def("nvshmem_get8", &nvshmem_get8, "8-bit get operation");
    m.def("nvshmem_get32", &nvshmem_get32, "32-bit get operation");
    m.def("nvshmem_get64", &nvshmem_get64, "64-bit get operation");

    // Single element operations
    m.def("nvshmem_p", py::overload_cast<int8_t*, int8_t, int>(&nvshmem_p), "8-bit put single element");
    m.def("nvshmem_p", py::overload_cast<int32_t*, int32_t, int>(&nvshmem_p), "32-bit put single element");
    m.def("nvshmem_p", py::overload_cast<int64_t*, int64_t, int>(&nvshmem_p), "64-bit put single element");

    m.def("nvshmem_g", py::overload_cast<const int8_t*, int>(&nvshmem_g), "8-bit get single element");
    m.def("nvshmem_g", py::overload_cast<const int32_t*, int>(&nvshmem_g), "32-bit get single element");
    m.def("nvshmem_g", py::overload_cast<const int64_t*, int>(&nvshmem_g), "64-bit get single element");
}

// NVSHMEM Atomic Operations
void BindNVSHMEM_Atomic(py::module& m) {
    // Fetch operations
    m.def("nvshmem_atomic_fetch_add", &nvshmem_atomic_fetch_add, "Atomic fetch and add");
    m.def("nvshmem_atomic_fetch_and", &nvshmem_atomic_fetch_and, "Atomic fetch and AND");
    m.def("nvshmem_atomic_fetch_or", &nvshmem_atomic_fetch_or, "Atomic fetch and OR");
    m.def("nvshmem_atomic_fetch_xor", &nvshmem_atomic_fetch_xor, "Atomic fetch and XOR");

    // Non-fetch operations
    m.def("nvshmem_atomic_add", &nvshmem_atomic_add, "Atomic add");
    m.def("nvshmem_atomic_and", &nvshmem_atomic_and, "Atomic AND");
    m.def("nvshmem_atomic_or", &nvshmem_atomic_or, "Atomic OR");
    m.def("nvshmem_atomic_xor", &nvshmem_atomic_xor, "Atomic XOR");

    // Compare and swap
    m.def("nvshmem_atomic_compare_swap", &nvshmem_atomic_compare_swap, "Atomic compare and swap");
    m.def("nvshmem_atomic_swap", &nvshmem_atomic_swap, "Atomic swap");

    // Increment/decrement
    m.def("nvshmem_atomic_fetch_inc", &nvshmem_atomic_fetch_inc, "Atomic fetch and increment");
    m.def("nvshmem_atomic_inc", &nvshmem_atomic_inc, "Atomic increment");

    // Set operations
    m.def("nvshmem_atomic_fetch_set", &nvshmem_atomic_fetch_set, "Atomic fetch and set");
    m.def("nvshmem_atomic_set", &nvshmem_atomic_set, "Atomic set");
}

// NVSHMEM Memory Ordering
void BindNVSHMEM_Ordering(py::module& m) {
    m.def("nvshmem_fence", &nvshmem_fence, "Memory fence");
    m.def("nvshmem_quiet", &nvshmem_quiet, "Quiet operation");
}

// NVSHMEM Point-to-Point Synchronization
void BindNVSHMEM_Sync(py::module& m) {
    // Constants for comparison operations
    m.attr("NVSHMEM_CMP_EQ") = NVSHMEM_CMP_EQ;
    m.attr("NVSHMEM_CMP_NE") = NVSHMEM_CMP_NE;
    m.attr("NVSHMEM_CMP_GT") = NVSHMEM_CMP_GT;
    m.attr("NVSHMEM_CMP_GE") = NVSHMEM_CMP_GE;
    m.attr("NVSHMEM_CMP_LT") = NVSHMEM_CMP_LT;
    m.attr("NVSHMEM_CMP_LE") = NVSHMEM_CMP_LE;

    m.def("nvshmem_wait_until", [](void* addr, int cmp, int64_t value) {
        nvshmem_wait_until(addr, cmp, value);
    }, "Wait until condition is met");

    m.def("nvshmem_test", [](void* addr, int cmp, int64_t value) {
        return nvshmem_test(addr, cmp, value);
    }, "Test condition");
}

// NVSHMEM Collective Operations
void BindNVSHMEM_Collective(py::module& m) {
    m.def("nvshmem_barrier", &nvshmem_barrier, "Barrier synchronization");
    m.def("nvshmem_barrier_all", &nvshmem_barrier_all, "Global barrier");

    m.def("nvshmem_sync", &nvshmem_sync, "Synchronization");
    m.def("nvshmem_sync_all", &nvshmem_sync_all, "Global synchronization");

    // Broadcast operations
    m.def("nvshmem_broadcast", [](py::buffer dest, py::buffer src, int root) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_broadcast(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                         nelems, root, 0, 0, nvshmem_n_pes());
    }, "Broadcast operation");

    // Collect operations
    m.def("nvshmem_fcollect", [](py::buffer dest, py::buffer src, int contrib) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_fcollect(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                        nelems, contrib, 0, 0, nvshmem_n_pes());
    }, "Flat collect operation");

    // All-to-all operations
    m.def("nvshmem_alltoall", [](py::buffer dest, py::buffer src) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_alltoall(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                        nelems, 0, 0, nvshmem_n_pes());
    }, "All-to-all operation");

    // Reduction operations
    m.def("nvshmem_sum_reduce", [](py::buffer dest, py::buffer src) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_sum_reduce(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                          nelems, 0, 0, nvshmem_n_pes());
    }, "Sum reduction");

    m.def("nvshmem_max_reduce", [](py::buffer dest, py::buffer src) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_max_reduce(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                          nelems, 0, 0, nvshmem_n_pes());
    }, "Max reduction");

    m.def("nvshmem_min_reduce", [](py::buffer dest, py::buffer src) {
        py::buffer_info dest_info = dest.request();
        py::buffer_info src_info = src.request();
        size_t nelems = src_info.size;
        nvshmem_min_reduce(static_cast<void*>(dest_info.ptr), static_cast<const void*>(src_info.ptr),
                          nelems, 0, 0, nvshmem_n_pes());
    }, "Min reduction");
}

// Main NVSHMEM binding function
void BindNVSHMEM(py::module& m) {
    auto nvshmem_module = m.def_submodule("nvshmem", "NVSHMEM-compatible Communication Library");

    BindNVSHMEM_Library(nvshmem_module);
    BindNVSHMEM_Memory(nvshmem_module);
    BindNVSHMEM_RMA(nvshmem_module);
    BindNVSHMEM_Atomic(nvshmem_module);
    BindNVSHMEM_Ordering(nvshmem_module);
    BindNVSHMEM_Sync(nvshmem_module);
    BindNVSHMEM_Collective(nvshmem_module);

    nvshmem_module.doc() = "NVSHMEM-compatible communication library for NPU clusters";
}

} // namespace pypto

