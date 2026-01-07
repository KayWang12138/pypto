/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NVSHMEM-compatible API header for NPU Communication Library
 * This header provides NVSHMEM-compatible function declarations
 */

#pragma once
#ifndef NVSHMEM_H
#define NVSHMEM_H

#include <cstdint>
#include <cstddef>

// Include our internal implementation
#include "npucomm.h"

// NVSHMEM API Declarations (compatible with NVSHMEM 3.4.5)

// Library Setup, Exit, and Query
int nvshmem_init();
int nvshmem_init_thread(int requested, int *provided);
int nvshmem_init_attr(const npu::comm::InitAttr *attr);
int nvshmem_my_pe();
int nvshmem_n_pes();
int nvshmem_finalize();
int nvshmem_global_exit(int status);

void *nvshmem_ptr(const void *ptr, int pe);
const char *nvshmem_info_get_version();
const char *nvshmem_info_get_name();

// Memory Management
void *nvshmem_malloc(size_t size);
void *nvshmem_calloc(size_t nmemb, size_t size);
void *nvshmem_align(size_t alignment, size_t size);
void nvshmem_free(void *ptr);

void *nvshmem_shmalloc(size_t size);
int nvshmem_shfree(void *ptr);
void *nvshmem_shmemalign(size_t alignment, size_t size);
void *nvshmem_shrealloc(void *ptr, size_t size);

// Remote Memory Access
void nvshmem_put(void *dest, const void *src, size_t size, int pe);
void nvshmem_get(void *dest, const void *src, size_t size, int pe);
void nvshmem_put_nbi(void *dest, const void *src, size_t size, int pe);
void nvshmem_get_nbi(void *dest, const void *src, size_t size, int pe);

// Typed put/get operations
void nvshmem_put8(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_put16(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_put32(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_put64(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_put128(void *dest, const void *src, size_t nelems, int pe);

void nvshmem_get8(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_get16(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_get32(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_get64(void *dest, const void *src, size_t nelems, int pe);
void nvshmem_get128(void *dest, const void *src, size_t nelems, int pe);

// Single element operations
void nvshmem_p(int8_t *dest, int8_t value, int pe);
void nvshmem_p(int16_t *dest, int16_t value, int pe);
void nvshmem_p(int32_t *dest, int32_t value, int pe);
void nvshmem_p(int64_t *dest, int64_t value, int pe);

int8_t nvshmem_g(const int8_t *src, int pe);
int16_t nvshmem_g(const int16_t *src, int pe);
int32_t nvshmem_g(const int32_t *src, int pe);
int64_t nvshmem_g(const int64_t *src, int pe);

// Strided operations
void nvshmem_iput32(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);
void nvshmem_iget32(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe);

// Atomic Memory Operations
int64_t nvshmem_atomic_fetch_add(int64_t *dest, int64_t value, int pe);
void nvshmem_atomic_add(int64_t *dest, int64_t value, int pe);
int64_t nvshmem_atomic_fetch_and(int64_t *dest, int64_t value, int pe);
void nvshmem_atomic_and(int64_t *dest, int64_t value, int pe);
int64_t nvshmem_atomic_fetch_or(int64_t *dest, int64_t value, int pe);
void nvshmem_atomic_or(int64_t *dest, int64_t value, int pe);
int64_t nvshmem_atomic_fetch_xor(int64_t *dest, int64_t value, int pe);
void nvshmem_atomic_xor(int64_t *dest, int64_t value, int pe);
int64_t nvshmem_atomic_compare_swap(int64_t *dest, int64_t compare, int64_t swap, int pe);
int64_t nvshmem_atomic_swap(int64_t *dest, int64_t value, int pe);
int64_t nvshmem_atomic_fetch_inc(int64_t *dest, int pe);
void nvshmem_atomic_inc(int64_t *dest, int pe);
int64_t nvshmem_atomic_fetch_set(int64_t *dest, int64_t value, int pe);
void nvshmem_atomic_set(int64_t *dest, int64_t value, int pe);

// Memory Ordering
void nvshmem_fence();
void nvshmem_quiet();

// Point-to-Point Synchronization
void nvshmem_wait_until(void *addr, int cmp, int64_t value);
int nvshmem_test(void *addr, int cmp, int64_t value);

// Collective Communication
void nvshmem_barrier(int PE_start, int logPE_stride, int PE_size);
void nvshmem_barrier_all();
void nvshmem_sync(int PE_start, int logPE_stride, int PE_size);
void nvshmem_sync_all();

void nvshmem_broadcast(void *dest, const void *src, size_t nelems, int root, int PE_start, int logPE_stride, int PE_size);
void nvshmem_fcollect(void *dest, const void *src, size_t nelems, int contrib, int PE_start, int logPE_stride, int PE_size);
void nvshmem_alltoall(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);

// Reduction Operations
void nvshmem_and_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_or_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_xor_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_max_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_min_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_sum_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);
void nvshmem_prod_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size);

// Team Management (simplified)
typedef int nvshmem_team_t;

int nvshmem_team_my_pe(nvshmem_team_t team);
int nvshmem_team_n_pes(nvshmem_team_t team);
int nvshmem_team_split_strided(nvshmem_team_t parent_team, int start, int stride, int size, nvshmem_team_t *new_team);
int nvshmem_team_destroy(nvshmem_team_t team);

// Constants (NVSHMEM compatibility)
#define NVSHMEM_MAJOR_VERSION 1
#define NVSHMEM_MINOR_VERSION 0
#define NVSHMEM_PATCH_VERSION 0

#define NVSHMEM_VENDOR_STRING "Huawei NPU Communication Library"

// Comparison operations for wait/test
#define NVSHMEM_CMP_EQ 0
#define NVSHMEM_CMP_NE 1
#define NVSHMEM_CMP_GT 2
#define NVSHMEM_CMP_GE 3
#define NVSHMEM_CMP_LT 4
#define NVSHMEM_CMP_LE 5

// Thread levels
#define NVSHMEM_THREAD_SINGLE 0
#define NVSHMEM_THREAD_FUNNELED 1
#define NVSHMEM_THREAD_SERIALIZED 2
#define NVSHMEM_THREAD_MULTIPLE 3

// Team handles
#define NVSHMEM_TEAM_WORLD 0
#define NVSHMEM_TEAM_SHARED 1
#define NVSHMEM_TEAM_NODE 2

#endif // NVSHMEM_H
