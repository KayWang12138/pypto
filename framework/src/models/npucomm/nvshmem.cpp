/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * NVSHMEM-compatible API implementation for NPU Communication Library
 */

#include "nvshmem.h"
#include "npucomm.h"

// Library Setup, Exit, and Query
int nvshmem_init() {
    return static_cast<int>(npu::comm::NpuComm::init());
}

int nvshmem_init_thread(int requested, int *provided) {
    npu::comm::ThreadLevel req_level = static_cast<npu::comm::ThreadLevel>(requested);
    npu::comm::ThreadLevel prov_level;
    npu::comm::CommStatus status = npu::comm::NpuComm::init_thread(req_level, &prov_level);
    *provided = static_cast<int>(prov_level);
    return static_cast<int>(status);
}

int nvshmem_init_attr(const npu::comm::InitAttr *attr) {
    return static_cast<int>(npu::comm::NpuComm::init_attr(*attr));
}

int nvshmem_my_pe() {
    return npu::comm::NpuComm::my_pe();
}

int nvshmem_n_pes() {
    return npu::comm::NpuComm::num_pes();
}

int nvshmem_finalize() {
    return static_cast<int>(npu::comm::NpuComm::finalize());
}

int nvshmem_global_exit(int status) {
    return static_cast<int>(npu::comm::NpuComm::global_exit(status));
}

void *nvshmem_ptr(const void *ptr, int pe) {
    return npu::comm::NpuComm::ptr(ptr, pe);
}

const char *nvshmem_info_get_version() {
    return npu::comm::NpuComm::info_get_version();
}

const char *nvshmem_info_get_name() {
    return npu::comm::NpuComm::info_get_name();
}

// Memory Management
void *nvshmem_malloc(size_t size) {
    return npu::comm::NpuComm::malloc(size);
}

void *nvshmem_calloc(size_t nmemb, size_t size) {
    return npu::comm::NpuComm::calloc(nmemb, size);
}

void *nvshmem_align(size_t alignment, size_t size) {
    return npu::comm::NpuComm::align(alignment, size);
}

void nvshmem_free(void *ptr) {
    npu::comm::NpuComm::free(ptr);
}

void *nvshmem_shmalloc(size_t size) {
    return npu::comm::NpuComm::shmalloc(size);
}

int nvshmem_shfree(void *ptr) {
    return static_cast<int>(npu::comm::NpuComm::shfree(ptr));
}

void *nvshmem_shmemalign(size_t alignment, size_t size) {
    return npu::comm::NpuComm::shmemalign(alignment, size);
}

void *nvshmem_shrealloc(void *ptr, size_t size) {
    return npu::comm::NpuComm::shrealloc(ptr, size);
}

// Remote Memory Access
void nvshmem_put(void *dest, const void *src, size_t size, int pe) {
    npu::comm::NpuComm::put(dest, src, size, pe);
}

void nvshmem_get(void *dest, const void *src, size_t size, int pe) {
    npu::comm::NpuComm::get(dest, src, size, pe);
}

void nvshmem_put_nbi(void *dest, const void *src, size_t size, int pe) {
    npu::comm::NpuComm::put_nbi(dest, src, size, pe);
}

void nvshmem_get_nbi(void *dest, const void *src, size_t size, int pe) {
    npu::comm::NpuComm::get_nbi(dest, src, size, pe);
}

// Typed put/get operations
void nvshmem_put8(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::put8(dest, src, nelems, pe);
}

void nvshmem_put16(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::put16(dest, src, nelems, pe);
}

void nvshmem_put32(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::put32(dest, src, nelems, pe);
}

void nvshmem_put64(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::put64(dest, src, nelems, pe);
}

void nvshmem_put128(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::put128(dest, src, nelems, pe);
}

void nvshmem_get8(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::get8(dest, src, nelems, pe);
}

void nvshmem_get16(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::get16(dest, src, nelems, pe);
}

void nvshmem_get32(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::get32(dest, src, nelems, pe);
}

void nvshmem_get64(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::get64(dest, src, nelems, pe);
}

void nvshmem_get128(void *dest, const void *src, size_t nelems, int pe) {
    npu::comm::NpuComm::get128(dest, src, nelems, pe);
}

// Single element operations
void nvshmem_p(int8_t *dest, int8_t value, int pe) {
    npu::comm::NpuComm::p8(dest, value, pe);
}

void nvshmem_p(int16_t *dest, int16_t value, int pe) {
    npu::comm::NpuComm::p16(dest, value, pe);
}

void nvshmem_p(int32_t *dest, int32_t value, int pe) {
    npu::comm::NpuComm::p32(dest, value, pe);
}

void nvshmem_p(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::p64(dest, value, pe);
}

int8_t nvshmem_g(const int8_t *src, int pe) {
    return npu::comm::NpuComm::g8(src, pe);
}

int16_t nvshmem_g(const int16_t *src, int pe) {
    return npu::comm::NpuComm::g16(src, pe);
}

int32_t nvshmem_g(const int32_t *src, int pe) {
    return npu::comm::NpuComm::g32(src, pe);
}

int64_t nvshmem_g(const int64_t *src, int pe) {
    return npu::comm::NpuComm::g64(src, pe);
}

// Strided operations
void nvshmem_iput32(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    npu::comm::NpuComm::iput32(dest, src, dst, sst, nelems, pe);
}

void nvshmem_iget32(void *dest, const void *src, ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe) {
    npu::comm::NpuComm::iget32(dest, src, dst, sst, nelems, pe);
}

// Atomic Memory Operations
int64_t nvshmem_atomic_fetch_add(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_fetch_add(dest, value, pe);
}

void nvshmem_atomic_add(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::atomic_add(dest, value, pe);
}

int64_t nvshmem_atomic_fetch_and(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_fetch_and(dest, value, pe);
}

void nvshmem_atomic_and(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::atomic_and(dest, value, pe);
}

int64_t nvshmem_atomic_fetch_or(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_fetch_or(dest, value, pe);
}

void nvshmem_atomic_or(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::atomic_or(dest, value, pe);
}

int64_t nvshmem_atomic_fetch_xor(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_fetch_xor(dest, value, pe);
}

void nvshmem_atomic_xor(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::atomic_xor(dest, value, pe);
}

int64_t nvshmem_atomic_compare_swap(int64_t *dest, int64_t compare, int64_t swap, int pe) {
    return npu::comm::NpuComm::atomic_compare_swap(dest, compare, swap, pe);
}

int64_t nvshmem_atomic_swap(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_swap(dest, value, pe);
}

int64_t nvshmem_atomic_fetch_inc(int64_t *dest, int pe) {
    return npu::comm::NpuComm::atomic_fetch_inc(dest, pe);
}

void nvshmem_atomic_inc(int64_t *dest, int pe) {
    npu::comm::NpuComm::atomic_inc(dest, pe);
}

int64_t nvshmem_atomic_fetch_set(int64_t *dest, int64_t value, int pe) {
    return npu::comm::NpuComm::atomic_fetch_set(dest, value, pe);
}

void nvshmem_atomic_set(int64_t *dest, int64_t value, int pe) {
    npu::comm::NpuComm::atomic_set(dest, value, pe);
}

// Memory Ordering
void nvshmem_fence() {
    npu::comm::NpuComm::fence();
}

void nvshmem_quiet() {
    npu::comm::NpuComm::quiet();
}

// Point-to-Point Synchronization
void nvshmem_wait_until(void *addr, int cmp, int64_t value) {
    npu::comm::CmpOp cmp_op = static_cast<npu::comm::CmpOp>(cmp);
    npu::comm::NpuComm::wait_until(addr, cmp_op, value);
}

int nvshmem_test(void *addr, int cmp, int64_t value) {
    npu::comm::CmpOp cmp_op = static_cast<npu::comm::CmpOp>(cmp);
    return npu::comm::NpuComm::test(addr, cmp_op, value);
}

// Collective Communication
void nvshmem_barrier(int PE_start, int logPE_stride, int PE_size) {
    // For simplicity, ignore team parameters and use barrier_all
    npu::comm::NpuComm::barrier_all();
}

void nvshmem_barrier_all() {
    npu::comm::NpuComm::barrier_all();
}

void nvshmem_sync(int PE_start, int logPE_stride, int PE_size) {
    // For simplicity, ignore team parameters and use sync_all
    npu::comm::NpuComm::sync_all();
}

void nvshmem_sync_all() {
    npu::comm::NpuComm::sync_all();
}

void nvshmem_broadcast(void *dest, const void *src, size_t nelems, int root, int PE_start, int logPE_stride, int PE_size) {
    // For simplicity, ignore team parameters
    npu::comm::NpuComm::broadcast(dest, src, nelems, root, 0);
}

void nvshmem_fcollect(void *dest, const void *src, size_t nelems, int contrib, int PE_start, int logPE_stride, int PE_size) {
    // For simplicity, ignore team parameters
    npu::comm::NpuComm::fcollect(dest, src, nelems, contrib, 0);
}

void nvshmem_alltoall(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    // For simplicity, ignore team parameters
    npu::comm::NpuComm::alltoall(dest, src, nelems, 0);
}

// Reduction Operations
void nvshmem_and_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::and_reduce(dest, src, nelems, 0);
}

void nvshmem_or_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::or_reduce(dest, src, nelems, 0);
}

void nvshmem_xor_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::xor_reduce(dest, src, nelems, 0);
}

void nvshmem_max_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::max_reduce(dest, src, nelems, 0);
}

void nvshmem_min_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::min_reduce(dest, src, nelems, 0);
}

void nvshmem_sum_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::sum_reduce(dest, src, nelems, 0);
}

void nvshmem_prod_reduce(void *dest, const void *src, size_t nelems, int PE_start, int logPE_stride, int PE_size) {
    npu::comm::NpuComm::prod_reduce(dest, src, nelems, 0);
}

// Team Management (simplified)
int nvshmem_team_my_pe(nvshmem_team_t team) {
    // Simplified: return my_pe for all teams
    return nvshmem_my_pe();
}

int nvshmem_team_n_pes(nvshmem_team_t team) {
    // Simplified: return n_pes for all teams
    return nvshmem_n_pes();
}

int nvshmem_team_split_strided(nvshmem_team_t parent_team, int start, int stride, int size, nvshmem_team_t *new_team) {
    // Simplified implementation
    *new_team = parent_team + 1; // Just increment team ID
    return 0; // Success
}

int nvshmem_team_destroy(nvshmem_team_t team) {
    // Simplified: do nothing
    return 0; // Success
}
