/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hypercube_overlap_checker.h
 * \brief
 */

#ifndef HYPERCUBE_OVERLAP_CHECKER_H
#define HYPERCUBE_OVERLAP_CHECKER_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
namespace npu::tile_fwk {

template <typename T>
class HypercubeOverlapChecker{
public:
    void Insert(const std::vector<int>& hypercube, T value);
    std::vector<T> Find(const std::vector<int>& hypercube); // [x_min, x_max, y_min, y_max, ...]
    void Erase(const std::vector<int>& hypercube, T value);
    void Shape2Keys(const std::vector<int> &hypercube, std::vector<uint64_t>& result, int dimIdx=0, uint64_t currValue=0);
    bool NoOverlap(const std::vector<int> &hypercube1, const std::vector<int> &hypercube2);
    bool SamePairVal(const std::pair<std::vector<int>, T>& pairVal1, const std::pair<std::vector<int>, T>& pairVal2);
    void Clear();

    int wide_{128};
    std::unordered_map<uint64_t, std::vector<std::pair<std::vector<int>, T>>> hashBucket_;

    std::unordered_set<T> container;
};

template<typename T>
void HypercubeOverlapChecker<T>::Clear() {
    hashBucket_.clear();
    container.clear();
}

template<typename T>
bool HypercubeOverlapChecker<T>::SamePairVal(const std::pair<std::vector<int>, T>& pairVal1, const std::pair<std::vector<int>, T>& pairVal2) {
    if (pairVal1.second != pairVal2.second){
        return false;
    }
    return true;
}

template<typename T>
bool HypercubeOverlapChecker<T>::NoOverlap(const std::vector<int> &hypercube1, const std::vector<int> &hypercube2) {
    constexpr int elementOfDim = 2;
    if(hypercube1.size()%elementOfDim != 0 || hypercube1.size() != hypercube2.size()){
        return true;
    }
    int dim = hypercube1.size()/elementOfDim;
    for(int i=0;i<dim;i++){
        int pStart = hypercube1[i*elementOfDim];
        int pEnd = hypercube1[i*elementOfDim+1];
        int qStart = hypercube2[i*elementOfDim];
        int qEnd = hypercube2[i*elementOfDim+1];
        if(pEnd <= qStart || pStart >= qEnd){
            return true;
        }
    }
    return false;
}

template<typename T>
void HypercubeOverlapChecker<T>::Shape2Keys(const std::vector<int> &hypercube, std::vector<uint64_t>& result, int dimIdx, uint64_t currValue){
    constexpr int elementOfDim = 2;
    if (dimIdx*elementOfDim >= static_cast<int>(hypercube.size())){
        result.push_back(currValue);
        return;
    }
    int start = hypercube[dimIdx*elementOfDim];
    int end = hypercube[dimIdx*elementOfDim + 1];
    int startGrid = start/wide_;
    int endGrid = end/wide_;

    constexpr int smallPrime = 131071;
    int newvalue = currValue * smallPrime;

    for(int i=startGrid; i<=endGrid; i++){
        Shape2Keys(hypercube, result, dimIdx+1, newvalue+i);
    }
}

template<typename T>
void HypercubeOverlapChecker<T>::Erase(const std::vector<int> &hypercube, T value) {
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    std::pair<std::vector<int>, T> pairVal{hypercube, value};
    for(auto key: keys){
        auto newEnd = std::remove_if(hashBucket_[key].begin(), hashBucket_[key].end(),
                                     [this, &pairVal](auto& pairVal2) { return SamePairVal(pairVal, pairVal2); });
        hashBucket_[key].erase(newEnd, hashBucket_[key].end());
    }
    container.erase(value);
}

template<typename T>
std::vector<T> HypercubeOverlapChecker<T>::Find(const std::vector<int> &hypercube) {
    std::vector<T> result;
    std::unordered_set<T> alreadyChecked;
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    for(auto key: keys){
        for(auto& pairVal: hashBucket_[key]){
            if( alreadyChecked.count(pairVal.second)==0 && !NoOverlap(hypercube, pairVal.first)){
                result.push_back(pairVal.second);
            }
            alreadyChecked.insert(pairVal.second);
        }
    }
    return result;
}

template<typename T>
void HypercubeOverlapChecker<T>::Insert(const std::vector<int> &hypercube, T value) {
    std::vector<uint64_t> keys;
    Shape2Keys(hypercube, keys);
    std::pair<std::vector<int>, T> pairVal{hypercube, value};
    for(auto key: keys){
        hashBucket_[key].push_back(pairVal);
    }
    container.insert(value);
}

}
#endif 
