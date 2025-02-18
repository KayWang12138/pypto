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
"""
TENSORUTILS_TEMPLATE = '''

#pragma once
#include "kernel_operator.h"



#ifdef JUSTFORHINTER
// just for type hinting.
#undef __aicore__
#define __aicore__
#include "stub_fun.h"
#endif 


using namespace AscendC;

#define PIPE_FIX (pipe_t)10

#define ALLCUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLCUBE_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x0, append_to_pipe>(iiii)
#define ALLVEC_WAIT(iiii, append_to_pipe) CrossCoreWaitFlag<0x0, append_to_pipe>(iiii)
#define CUBE_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_CUBE(iiii, append_to_pipe) CrossCoreWaitFlag<0x2, append_to_pipe>(iiii)
#define VEC_READY(iiii, append_to_pipe) CrossCoreSetFlag<0x2, append_to_pipe>(iiii)
#define WAIT_VEC(iiii, append_to_pipe) CrossCoreWaitFlag<0x2, append_to_pipe>(iiii)

typedef enum{
    PGM=0,
    PL1=1,
    PL0A=2,
    PL0B=3,
    PL0C=4,
    PUB=5,
} pos_t;


__aicore__ constexpr int Align16B(int x){
    return (x + 15) / 16 * 16;
}

__aicore__ constexpr int Align32B(int x){
    return (x + 31) / 32 * 32;
}

__aicore__ constexpr int Align64B(int x){
    return (x + 63) / 64 * 64;
}

__aicore__ constexpr int Align128B(int x){
    return (x + 127) / 128 * 128;
}

__aicore__ constexpr int Align256B(int x){
    return (x + 255) / 256 * 256;
}

__aicore__ constexpr int Align512B(int x){
    return (x + 511) / 512 * 512;
}

__aicore__ inline int CeilDiv(int a, int b){
    return (a + b - 1) / b;
}

template <typename T, typename T1, typename T2>
__aicore__ inline T1 shiftAddr(T1 base, uint64_t size, T2 &offset){
    auto res = base + offset;
    offset += size*sizeof(T);
    return res;
}


/* ------------- Tensor ------------- */ 

template <typename T, pos_t pos>
class Tensor{};


template <typename T>
class Tensor<T, PGM>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__gm__ T*) offset;
    }
    __aicore__ inline Tensor(__gm__ uint8_t* ptr){
        m_ptr = (__gm__ T*) ptr;
    }
    __aicore__ inline Tensor(__gm__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__gm__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    __aicore__ inline __gm__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __gm__ void* vptr(){
        return (__gm__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PGM> operator[](int off){
        return Tensor<T, PGM>((__gm__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PGM>(){
        return Tensor<U, PGM>((__gm__ uint8_t*) m_ptr);
    }
private:
    __gm__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL1>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__cbuf__ T*) offset;
    }
    __aicore__ inline Tensor(__cbuf__ uint8_t* ptr){
        m_ptr = (__cbuf__ T*) ptr;
    }
    __aicore__ inline Tensor(__cbuf__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__cbuf__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    __aicore__ inline __cbuf__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __cbuf__ void* vptr(){
        return (__cbuf__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PL1> operator[](int off){
        return Tensor<T, PL1>((__cbuf__ uint8_t*)(m_ptr + off));
    }
    __aicore__ inline Tensor<T, PL1> rowcol(int r, int c, int C){
        return Tensor<T, PL1>(((__cbuf__ uint8_t*)m_ptr) + (r*C + c) * 16*32);
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PL1>(){
        return Tensor<U, PL1>((__cbuf__ uint8_t*) m_ptr);
    }
private:
    __cbuf__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0A>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__ca__ T*) offset;
    }
    __aicore__ inline Tensor(__ca__ uint8_t* ptr){
        m_ptr = (__ca__ T*) ptr;
    }
    __aicore__ inline __ca__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __ca__ void* vptr(){
        return (__ca__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PL0A> operator[](int off){
        return Tensor<T, PL0A>((__ca__ uint8_t*)(m_ptr + off));
    }
    __aicore__ inline Tensor<T, PL0A> rowcol(int r, int c, int C){
        return Tensor<T, PL0A>((__ca__ uint8_t*)m_ptr + (r*C + c) * 16*32);
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PL0A>(){
        return Tensor<U, PL0A>((__ca__ uint8_t*) m_ptr);
    }
private:
    __ca__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0B>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__cb__ T*) offset;
    }
    __aicore__ inline Tensor(__cb__ uint8_t* ptr){
        m_ptr = (__cb__ T*) ptr;
    }
    __aicore__ inline __cb__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __cb__ void* vptr(){
        return (__cb__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PL0B> operator[](int off){
        return Tensor<T, PL0B>((__cb__ uint8_t*)(m_ptr + off));
    }
    __aicore__ inline Tensor<T, PL0B> rowcol(int r, int c, int C){
        return Tensor<T, PL0B>((__cb__ uint8_t*)m_ptr + (r*C + c) * 16*32);
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PL0B>(){
        return Tensor<U, PL0B>((__cb__ uint8_t*) m_ptr);
    }
private:
    __cb__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0C>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__cc__ T*) offset;
    }
    __aicore__ inline Tensor(__cc__ uint8_t* ptr){
        m_ptr = (__cc__ T*) ptr;
    }
    __aicore__ inline __cc__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __cc__ void* vptr(){
        return (__cc__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PL0C> operator[](int off){
        return Tensor<T, PL0C>((__cc__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PL0C>(){
        return Tensor<U, PL0C>((__cc__ uint8_t*) m_ptr);
    }
private:
    __cc__ T* m_ptr;
};


template <typename T>
class Tensor<T, PUB>{
public:
    __aicore__ inline Tensor(){}
    __aicore__ inline Tensor(uint64_t offset){
        m_ptr = (__ubuf__ T*) offset;
    }
    __aicore__ inline Tensor(__ubuf__ uint8_t* ptr){
        m_ptr = (__ubuf__ T*) ptr;
    }
    __aicore__ inline Tensor(__ubuf__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__ubuf__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    __aicore__ inline __ubuf__ T* ptr(){
        return m_ptr;
    }
    __aicore__ inline __ubuf__ void* vptr(){
        return (__ubuf__ void*) m_ptr;
    }
    __aicore__ inline Tensor<T, PUB> operator[](int off){
        return Tensor<T, PUB>((__ubuf__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    __aicore__ inline operator Tensor<U, PUB>(){
        return Tensor<U, PUB>((__ubuf__ uint8_t*) m_ptr);
    }
private:
    __ubuf__ T* m_ptr;
};

/* ------------- Tensor ------------- */ 


/* ------------- Double Buffer ------------- */ 

template <typename T, pos_t pos>
class DBuff{
};


template <typename T>
class DBuff<T, PGM>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PGM>(base + offset);
        tsr2 = Tensor<T, PGM>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__gm__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PGM>(ptr + offset);
        tsr2 = Tensor<T, PGM>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline Tensor<T, PGM> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PGM> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL1>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL1>(base + offset);
        tsr2 = Tensor<T, PL1>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__cbuf__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL1>(ptr + offset);
        tsr2 = Tensor<T, PL1>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(Tensor<T, PL1> t1, Tensor<T, PL1> t2){
        tsr1 = t1;
        tsr2 = t2;
    }
    __aicore__ inline Tensor<T, PL1> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL1> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0A>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size){
        tsr1 = Tensor<T, PL0A>(base);
        tsr2 = Tensor<T, PL0A>(base + size*sizeof(T));
    }
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0A>(base + offset);
        tsr2 = Tensor<T, PL0A>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__ca__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0A>(ptr + offset);
        tsr2 = Tensor<T, PL0A>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline Tensor<T, PL0A> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0A> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0B>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size){
        tsr1 = Tensor<T, PL0B>(base);
        tsr2 = Tensor<T, PL0B>(base + size*sizeof(T));
    }
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0B>(base + offset);
        tsr2 = Tensor<T, PL0B>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__cb__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0B>(ptr + offset);
        tsr2 = Tensor<T, PL0B>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline Tensor<T, PL0B> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0B> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0C>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size){
        tsr1 = Tensor<T, PL0C>(base);
        tsr2 = Tensor<T, PL0C>(base + size*sizeof(T));
    }
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0C>(base + offset);
        tsr2 = Tensor<T, PL0C>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__cc__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0C>(ptr + offset);
        tsr2 = Tensor<T, PL0C>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline Tensor<T, PL0C> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0C> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PUB>{
public:
    __aicore__ inline DBuff(){}
    __aicore__ inline DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PUB>(base + offset);
        tsr2 = Tensor<T, PUB>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline DBuff(__ubuf__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PUB>(ptr + offset);
        tsr2 = Tensor<T, PUB>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    __aicore__ inline Tensor<T, PUB> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PUB> tsr1, tsr2;
};

/* ------------- Double Buffer ------------- */ 


/* ------------- Events ------------- */ 

template <pipe_t p1, pipe_t p2>
class SEvent{
public:
    __aicore__ inline SEvent(){}
    __aicore__ inline SEvent(int e_id1, int e_id2){
        id1 = (event_t)e_id1; 
        id2 = (event_t)e_id2;
    }
    __aicore__ inline SEvent(event_t e_id1, event_t e_id2){
        id1 = e_id1; 
        id2 = e_id2;
    }
    __aicore__ inline void wait(){
        wait_flag(p1, p2, id1);
    }
    __aicore__ inline void set(){
        set_flag(p1, p2, id1);
    }
    __aicore__ inline void setall(){
        set();
    }
    __aicore__ inline void release(){
        wait();
    }

private:
    event_t id1=(event_t)0, id2=(event_t)1;
};



template <pipe_t p1, pipe_t p2>
class DEvent{
public:
    __aicore__ inline DEvent(){}
    __aicore__ inline DEvent(int e_id1, int e_id2){
        id1 = (event_t)e_id1; 
        id2 = (event_t)e_id2;
    }
    __aicore__ inline DEvent(event_t e_id1, event_t e_id2){
        id1 = e_id1; 
        id2 = e_id2;
    }
    __aicore__ inline void wait(){
        if (wait_cnt%2==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        wait_cnt ++;
    }
    __aicore__ inline void set(){
        if (set_cnt%2==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        set_cnt ++;
    }
    __aicore__ inline void setall(){
        set();
        set();
    }
    __aicore__ inline void release(){
        for (int i=wait_cnt; i<set_cnt; ++i){
            wait();
        }
    }

private:
    event_t id1=(event_t)0, id2=(event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};


template <pipe_t p1, pipe_t p2>
class DEventP{
public:
    __aicore__ inline DEventP(){}
    __aicore__ inline DEventP(const int e_id1, const int e_id2): id1((event_t)e_id1), id2((event_t)e_id2){
        // id1 = (event_t)e_id1; 
        // id2 = (event_t)e_id2;
    }
    __aicore__ inline DEventP(const event_t e_id1, const event_t e_id2): id1(e_id1), id2(e_id2){
        // id1 = e_id1; 
        // id2 = e_id2;
    }
    __aicore__ inline void wait(int &wait_cnt){
        if (wait_cnt==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        // wait_cnt ++;
    }
    __aicore__ inline void set(int &set_cnt){
        if (set_cnt==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        // set_cnt ++;
    }
    __aicore__ inline void wait(int &&wait_cnt){
        if (wait_cnt==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        // wait_cnt ++;
    }
    __aicore__ inline void set(int &&set_cnt){
        if (set_cnt==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        // set_cnt ++;
    }
    __aicore__ inline void setall(){
        set_flag(p1, p2, id1);
        set_flag(p1, p2, id2);
    }
    __aicore__ inline void release(){
        // for (int i=wait_cnt; i<set_cnt; ++i){
            // wait();
        // }
        wait_flag(p1, p2, id1);
        wait_flag(p1, p2, id2);
    }

private:
    const event_t id1, id2;
};


/* ------------- Events ------------- */ 

/* ------------- Funcs -------------- */
template<typename T>
__aicore__ inline void L1ND2NZ(Tensor<T, PL1> dst, Tensor<T, PGM> src, int h, int w, int W, int H){
    copy_gm_to_cbuf_multi_nd2nz_b16(dst.ptr(), src.ptr(), 0, 1, h, w, 0, W, (H+15)/16*16, 1, 0);
}

template <typename T>
__aicore__ inline void L0NZ2ZZ(Tensor<T, PL0A> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(mdst+15)/16; ++i){
        load_cbuf_to_ca(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, false, (addr_cal_mode_t)0);
    }
}

template <typename T>
__aicore__ inline void L0NZ2ZZ(Tensor<T, PL0B> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(mdst+15)/16; ++i){
        load_cbuf_to_cb(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, false, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2ZN(Tensor<T, PL0A> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(mdst+15)/16; ++i){
        load_cbuf_to_ca(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, true, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2ZN(Tensor<T, PL0B> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(mdst+15)/16; ++i){
        load_cbuf_to_cb(dst[16*i*((ndst+15)/16*16)].ptr(), src[i*16*16].ptr(), 0, (ndst+15)/16, (msrc+15)/16, 0, 0, true, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2NZ(Tensor<T, PL0A> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(ndst+15)/16; ++i){
        load_cbuf_to_ca(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, false, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2NZ(Tensor<T, PL0B> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(ndst+15)/16; ++i){
        load_cbuf_to_cb(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, false, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2NN(Tensor<T, PL0A> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(ndst+15)/16; ++i){
        load_cbuf_to_ca(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, true, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void L0NZ2NN(Tensor<T, PL0B> dst, Tensor<T, PL1> src, int mdst, int ndst, int msrc, int nsrc){
    for (int i=0; i<(ndst+15)/16; ++i){
        load_cbuf_to_cb(dst[16*i*((mdst+15)/16*16)].ptr(), src[16*i*((msrc+15)/16*16)].ptr(), 0, (mdst+15)/16, 1, 0, 0, true, (addr_cal_mode_t)0);
    }
}

template<typename T> 
__aicore__ inline void LOADL0(Tensor<T, PL0A> dst, Tensor<T, PL1> src, int m, int n){
    load_cbuf_to_ca(dst.ptr(), src.ptr(), 0, m*n/16/16, 1, 0, false);
}

template<typename T> 
__aicore__ inline void LOADL0(Tensor<T, PL0B> dst, Tensor<T, PL1> src, int m, int n){
    load_cbuf_to_cb(dst.ptr(), src.ptr(), 0, m*n/16/16, 1, 0, false);
}

template <typename T> 
__aicore__ inline void VECNZ2ND(Tensor<T, PUB> dst, Tensor<T, PGM> src, int mdst, int ndst, int msrc, int nsrc){
    int n_small = (ndst>nsrc) ? nsrc : ndst;
    int m_small = (mdst>msrc) ? msrc : nsrc;
    for (int i=0; i<(n_small+15)/16; ++i){
        copy_gm_to_ubuf(dst[i*16].vptr(), src[i*16*((msrc+15)/16*16)].vptr(), 0, m_small, sizeof(T)/2, 0, ((ndst+15)/16-1)*sizeof(T)/2 );
    }
}

template <typename T> 
__aicore__ inline void VECNZ2ND(Tensor<T, PUB> dst, Tensor<T, PUB> src, int mdst, int ndst, int msrc, int nsrc){
    int n_small = (ndst>nsrc) ? nsrc : ndst;
    int m_small = (mdst>msrc) ? msrc : nsrc;
    for (int i=0; i<(n_small+15)/16; ++i){
        copy_ubuf_to_ubuf(dst[i*16].vptr(), src[i*16*((msrc+15)/16*16)].vptr(), 0, m_small, sizeof(T)/2, 0, ((ndst+15)/16-1)*sizeof(T)/2 );
    }
}

template <typename T> 
__aicore__ inline void VECNZ2ND(Tensor<T, PGM> dst, Tensor<T, PUB> src, int mdst, int ndst, int msrc, int nsrc){
    int m_small = (mdst>msrc) ? msrc : mdst;
    int n_burst = (ndst>nsrc) ? nsrc : ndst;
    for (int i=0; i<m_small; ++i){
        copy_ubuf_to_gm(dst[i*ndst].vptr(), src[i*16].vptr(), 0, (n_burst+15)/16, sizeof(T)/2, ((msrc+15)/16*16-1)*sizeof(T)/2, 0);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2ND(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int N){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (m+15)/16*16, 0x0, NoQuant, 0, false, true);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (m+15)/16*16, 0x0, F322F16, 0, false, true);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (m+15)/16*16, 0x0, F322BF16, 0, false, true);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (m+15)/16*16, 0x0, NoQuant, 0, false, true);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2ND(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int N, int nz_M){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, 0x0, NoQuant, 0, false, true);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, 0x0, F322F16, 0, false, true);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, 0x0, F322BF16, 0, false, true);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, 0x0, NoQuant, 0, false, true);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2ND(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int N, int nz_M, uint8_t uflag){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322F16, 0, false, true);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, F322BF16, 0, false, true);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, N, (nz_M+15)/16*16, uflag, NoQuant, 0, false, true);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (m+15)/16*16, 0x0, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (m+15)/16*16, 0x0, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (m+15)/16*16, 0x0, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T)/2, (m+15)/16*16, 0x0, NoQuant, 0, false, false);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M, int src_M){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (src_M+15)/16*16, 0x0, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, 0x0, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, 0x0, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T2)/2, (src_M+15)/16*16, 0x0, NoQuant, 0, false, false);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ(Tensor<T, PGM> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M, int src_M, uint8_t uflag){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (src_M+15)/16*16, uflag, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, uflag, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, uflag, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T2)/2, (src_M+15)/16*16, uflag, NoQuant, 0, false, false);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2L1_NZ(Tensor<T, PL1> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (m+15)/16*16, 0x0, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (m+15)/16*16, 0x0, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (m+15)/16*16, 0x0, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T)/2, (m+15)/16*16, 0x0, NoQuant, 0, false, false);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2L1_NZ(Tensor<T, PL1> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M, int src_M){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (src_M+15)/16*16, 0x0, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, 0x0, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, 0x0, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T2)/2, (src_M+15)/16*16, 0x0, NoQuant, 0, false, false);
    }
}

template <typename T, typename T2>
__aicore__ inline void L0C2L1_NZ(Tensor<T, PL1> dst, Tensor<T2, PL0C> src, int m, int n, int dst_M, int src_M, uint8_t uflag){
    if constexpr(std::is_same<T, float>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*2, (src_M+15)/16*16, uflag, NoQuant, 0, false, false);
    }else if constexpr(std::is_same<T, half>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, uflag, F322F16, 0, false, false);
    }else if constexpr(std::is_same<T, bfloat16_t>::value && std::is_same<T2, float>::value){
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M, (src_M+15)/16*16, uflag, F322BF16, 0, false, false);
    }else{
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, n, m, dst_M*sizeof(T2)/2, (src_M+15)/16*16, uflag, NoQuant, 0, false, false);
    }
}


template <typename T1, typename T2, typename T3>
__aicore__ inline void MMAD(Tensor<T1, PL0C> dst, Tensor<T2, PL0A> src0, Tensor<T3, PL0B> src1, uint16_t m, uint16_t k, uint16_t n, bool cmatrixInitVal, uint8_t unitFlag){
    mad(dst.ptr(), src0.ptr(), src1.ptr(), m, k, n, 0x0, false, false, cmatrixInitVal);
} 


'''
