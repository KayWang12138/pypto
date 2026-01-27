/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file union_find.hpp
 * \brief
 */

#ifndef PASS_OSP_UNION_FIND_H
#define PASS_OSP_UNION_FIND_H

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "passes/algorithms/osp/concepts/graph_traits.hpp"

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Structure to represent an object in the union-find universe.
 *
 * @tparam T Type of the unique identifier (name).
 * @tparam IndexT Type of the index used for internal references.
 * @tparam WorkwT Type of the weight associated with the object.
 */
template <typename T, typename IndexT, typename WorkwT>
struct UnionFindObject {
    const T name_;       /** Unique identifier of the object. */
    IndexT parentIndex_; /** Index of the parent object in the union-find tree. */
    unsigned rank_;      /** Rank of the object, used for union operation optimization. */
    WorkwT weight_;      /** Weight associated with the object. */

    /**
     * @brief Constructs a new UnionFindObject.
     *
     * @param name Unique identifier.
     * @param parentIndex Index of the parent object.
     * @param weight Weight of the object. Default is 0.
     */
    explicit UnionFindObject(const T &name, IndexT parentIndex, WorkwT weight = 0)
        : name_(name), parentIndex_(parentIndex), rank_(1), weight_(weight) {}

    UnionFindObject(const UnionFindObject &other) = default;
    UnionFindObject &operator=(const UnionFindObject &other) = default;
};

/**
 * @brief Class to execute a union-find algorithm with path compression and union by rank.
 *
 * This class manages a set of elements partitioned into disjoint sets. It supports adding elements
 * and merging sets (finding connected components).
 *
 * @tparam T Type of the unique identifier (name).
 * @tparam IndexT Type of the index used for internal references.
 * @tparam WorkwT Type of the weight associated with the object.
 */
template <typename T, typename IndexT, typename WorkwT>
class UnionFindUniverse {
  private:
    std::vector<UnionFindObject<T, IndexT, WorkwT>> universe_;
    std::unordered_map<T, IndexT> namesToIndices_;
    std::set<IndexT> componentIndices_;

    IndexT FindOrigin(IndexT index) {
        while (index != universe_[index].parentIndex_) {
            universe_[index].parentIndex_ = universe_[universe_[index].parentIndex_].parentIndex_;
            index = universe_[index].parentIndex_;
        }
        return index;
    }

    int Join(IndexT index, IndexT otherIndex) {
        index = FindOrigin(index);
        otherIndex = FindOrigin(otherIndex);

        if (index == otherIndex) {
            return 0;
        }

        if (universe_[index].rank_ >= universe_[otherIndex].rank_) {
            universe_[otherIndex].parentIndex_ = index;
            universe_[index].weight_ += universe_[otherIndex].weight_;
            componentIndices_.erase(otherIndex);

            if (universe_[index].rank_ == universe_[otherIndex].rank_) {
                universe_[index].rank_++;
            }
        } else {
            universe_[index].parentIndex_ = otherIndex;
            universe_[otherIndex].weight_ += universe_[index].weight_;
            componentIndices_.erase(index);
        }
        return -1;
    }

    IndexT GetIndexFromName(const T &name) const { return namesToIndices_.at(name); }

    void AddObjectInternal(const T &name, WorkwT weight) {
        if (namesToIndices_.find(name) != namesToIndices_.end()) {
            throw std::runtime_error("This name already exists in the universe.");
        }
        IndexT newIndex = static_cast<IndexT>(universe_.size());
        universe_.emplace_back(name, newIndex, weight);
        namesToIndices_[name] = newIndex;
        componentIndices_.emplace(newIndex);
    }

  public:
    explicit UnionFindUniverse() = default;

    UnionFindUniverse(const UnionFindUniverse &other) = default;
    UnionFindUniverse &operator=(const UnionFindUniverse &other) = default;
    UnionFindUniverse(UnionFindUniverse &&other) noexcept = default;
    UnionFindUniverse &operator=(UnionFindUniverse &&other) noexcept = default;
    ~UnionFindUniverse() = default;

    /**
     * @brief Resets the universe, clearing all objects and components.
     */
    void Reset() {
        universe_.clear();
        namesToIndices_.clear();
        componentIndices_.clear();
    }

    /**
     * @brief Checks if an object exists in the universe.
     * @param name The name of the object.
     * @return True if the object exists, false otherwise.
     */
    [[nodiscard]] bool IsInUniverse(const T &name) const noexcept { return namesToIndices_.find(name) != namesToIndices_.end(); }

    /**
     * @brief Finds the representative name of the component containing the object.
     * @param name The name of the object.
     * @return The name of the component's representative.
     */
    [[nodiscard]] T FindOriginByName(const T &name) { return universe_[FindOrigin(namesToIndices_.at(name))].name_; }

    /**
     * @brief Joins the components containing the two objects.
     * @param name Name of the first object.
     * @param otherName Name of the second object.
     */
    void JoinByName(const T &name, const T &otherName) { Join(namesToIndices_.at(name), namesToIndices_.at(otherName)); }

    /**
     * @brief Retrieves the current number of connected components.
     * @return Number of disjoint sets.
     */
    [[nodiscard]] std::size_t GetNumberOfConnectedComponents() const noexcept { return componentIndices_.size(); }

    /**
     * @brief Retrieves the names of all component representatives.
     * @return Vector of names.
     */
    [[nodiscard]] std::vector<T> GetComponentNames() const {
        std::vector<T> componentNames;
        componentNames.reserve(componentIndices_.size());
        for (auto &indx : componentIndices_) {
            componentNames.emplace_back(universe_[indx].name_);
        }
        return componentNames;
    }

    /**
     * @brief Retrieves the weight of the component containing the given object.
     * @param name Name of the object.
     * @return Total weight of the component.
     */
    [[nodiscard]] WorkwT GetWeightOfComponentByName(const T &name) {
        IndexT index = GetIndexFromName(name);
        index = FindOrigin(index);
        return universe_[index].weight_;
    }

    /**
     * @brief Retrieves all connected components grouping member names.
     * @return Vector of components, where each component is a vector of names.
     */
    [[nodiscard]] std::vector<std::vector<T>> GetConnectedComponents() {
        std::vector<std::vector<IndexT>> connectedComponentsByIndex;
        connectedComponentsByIndex.resize(universe_.size());
        for (IndexT i = 0; i < static_cast<IndexT>(universe_.size()); i++) {
            connectedComponentsByIndex[FindOrigin(i)].emplace_back(i);
        }

        std::vector<std::vector<T>> connectedComponentsByName;
        connectedComponentsByName.reserve(componentIndices_.size());

        for (auto &comp : connectedComponentsByIndex) {
            if (comp.empty()) {
                continue;
            }
            std::vector<T> namesInComp;
            namesInComp.reserve(comp.size());
            for (const auto &indx : comp) {
                namesInComp.emplace_back(universe_[indx].name_);
            }
            connectedComponentsByName.push_back(std::move(namesInComp));
        }

        return connectedComponentsByName;
    }

    /**
     * @brief Adds a single object to the universe.
     * @param name Name of the object.
     */
    void AddObject(const T &name) { AddObjectInternal(name, 0); }

    /**
     * @brief Adds a single object with weight.
     * @param name Name of the object.
     * @param weight Weight of the object.
     */
    void AddObject(const T &name, const WorkwT weight) { AddObjectInternal(name, weight); }
};

}    // namespace osp
}    // namespace npu::tile_fwk
#endif    // PASS_OSP_UNION_FIND_H