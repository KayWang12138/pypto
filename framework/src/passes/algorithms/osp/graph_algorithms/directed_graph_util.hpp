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
 * \file directed_graph_util.hpp
 * \brief
 */

#ifndef OSP_DIRECTED_GRAPH_UTIL_HPP
#define OSP_DIRECTED_GRAPH_UTIL_HPP

#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

#include "passes/algorithms/osp/concepts/directed_graph_concept.hpp"

/**
 * @file directed_graph_util.hpp
 * @brief Utility functions and classes for working with directed graphs.
 *
 * This file provides a collection of utility functions, iterators, and views
 * for performing operations on directed graphs. These utilities include
 * functions for checking graph properties, retrieving specific vertices,
 * and traversing the graph using BFS and DFS.
 */

namespace npu::tile_fwk {
namespace osp {

/**
 * @brief Checks if there is an edge between two vertices in the graph.
 *
 * @tparam GraphT The type of the graph.
 * @param src The source vertex.
 * @param dest The destination vertex.
 * @param graph The graph to check.
 * @return true if there is an edge from src to dest, false otherwise.
 */
template <typename GraphT>
bool Edge(const VertexIdxT<GraphT> &src, const VertexIdxT<GraphT> &dest, const GraphT &graph) {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    for (const auto &child : graph.Children(src)) {
        if (child == dest) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Helper struct for iterating over vertices with a condition.
 *
 * This struct provides an iterator that filters vertices based on a given condition.
 * It is used to create views for source and sink vertices in a directed graph.
 *
 */
template <typename CondEval, typename GraphT, typename IteratorT>
struct VertexCondIterator {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    // TODO static_assert(is_callabl_v<cond_eval>;

    const GraphT &graph_;
    IteratorT currentVertex_;
    CondEval cond_;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = VertexIdxT<GraphT>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    VertexCondIterator(const GraphT &graph, const IteratorT &start) : graph_(graph), currentVertex_(start) {
        while (currentVertex_ != graph_.Vertices().end()) {
            // if (cond.eval(graph, *current_vertex)) {
            if (cond_(graph_, *currentVertex_)) {
                break;
            }
            currentVertex_++;
        }
    }

    value_type operator*() const { return currentVertex_.operator*(); }

    // Prefix increment
    VertexCondIterator &operator++() {
        currentVertex_++;

        while (currentVertex_ != graph_.Vertices().end()) {
            if (cond_(graph_, *currentVertex_)) {
                break;
            }
            currentVertex_++;
        }

        return *this;
    }

    // Postfix increment
    VertexCondIterator operator++(int) {
        VertexCondIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    inline bool operator==(const VertexCondIterator &other) { return currentVertex_ == other.currentVertex_; };

    inline bool operator!=(const VertexCondIterator &other) { return currentVertex_ != other.currentVertex_; };
};

/**
 * @brief Views for source vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template <typename GraphT>
class SourceVerticesView {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct SourceEval {
        // static bool eval(const GraphT &graph, const VertexIdxT<GraphT> &v) { return graph.InDegree(v) == 0; }
        bool operator()(const GraphT &graph, const VertexIdxT<GraphT> &v) const { return graph.InDegree(v) == 0; }
    };

    using SourceIterator = VertexCondIterator<SourceEval, GraphT, decltype(graph_.Vertices().begin())>;

  public:
    SourceVerticesView(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return SourceIterator(graph_, graph_.Vertices().begin()); }

    auto end() const { return SourceIterator(graph_, graph_.Vertices().end()); }

    auto size() const { return graph_.NumVertices(); }
};

/**
 * @brief Views for sink vertices in a directed graph.
 *
 * These classes provide iterators to traverse the source and sink vertices
 * of a directed graph.
 */
template <typename GraphT>
class SinkVerticesView {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;

    struct SinkEval {
        // static bool eval(const GraphT &graph, const VertexIdxT<GraphT> &v) { return graph.OutDegree(v) == 0; }
        bool operator()(const GraphT &graph, const VertexIdxT<GraphT> &v) { return graph.OutDegree(v) == 0; }
    };

    using SinkIterator = VertexCondIterator<SinkEval, GraphT, decltype(graph_.Vertices().begin())>;

  public:
    SinkVerticesView(const GraphT &graph) : graph_(graph) {}

    auto begin() const { return SinkIterator(graph_, graph_.Vertices().begin()); }

    auto end() const { return SinkIterator(graph_, graph_.Vertices().end()); }

    auto size() const { return graph_.NumVertices(); }
};


/**
 * @brief Traversal iterator for directed graphs.
 *
 * This iterator allows traversing the vertices of a directed graph.
 * It uses a container wrapper to manage the traversal order.
 * The adj_iterator can be used to setup the traversal along children or parents.
 */
template <typename GraphT, typename ContainerWrapper, typename AdjIterator>
struct TraversalIterator {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;

    AdjIterator adjIter_;

    ContainerWrapper vertexContainer_;

    std::unordered_set<VertexIdxT<GraphT>> visited_;
    VertexIdxT<GraphT> currentVertex_;

  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = VertexIdxT<GraphT>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type *;
    using reference = const value_type &;

    TraversalIterator(const GraphT &graph, const VertexIdxT<GraphT> &start)
        : graph_(graph), adjIter_(graph), currentVertex_(start) {
        if (graph_.NumVertices() == start) {
            return;
        }

        visited_.insert(start);

        for (const auto &v : adjIter_.Iterate(currentVertex_)) {
            vertexContainer_.Push(v);
            visited_.insert(v);
        }
    }

    value_type operator*() const { return currentVertex_; }

    // Prefix increment
    TraversalIterator &operator++() {
        if (vertexContainer_.empty()) {
            currentVertex_ = graph_.NumVertices();
            return *this;
        }

        currentVertex_ = vertexContainer_.PopNext();

        for (const auto &v : adjIter_.Iterate(currentVertex_)) {
            if (visited_.find(v) == visited_.end()) {
                vertexContainer_.Push(v);
                visited_.insert(v);
            }
        }

        return *this;
    }

    // Postfix increment !! expensive
    TraversalIterator operator++(int) {
        TraversalIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    inline bool operator==(const TraversalIterator &other) { return currentVertex_ == other.currentVertex_; };

    inline bool operator!=(const TraversalIterator &other) { return currentVertex_ != other.currentVertex_; };
};

template <typename GraphT>
struct ChildIterator {
    const GraphT &graph_;

    ChildIterator(const GraphT &graph) : graph_(graph) {}

    inline auto Iterate(const VertexIdxT<GraphT> &v) const { return graph_.Children(v); }
};

template <typename GraphT>
struct BfsQueueWrapper {
    std::queue<VertexIdxT<GraphT>> queue_;

    void Push(const VertexIdxT<GraphT> &v) { queue_.push(v); }

    VertexIdxT<GraphT> PopNext() {
        auto v = queue_.front();
        queue_.pop();
        return v;
    }

    bool empty() const { return queue_.empty(); }
};

/**
 * @brief Views for traversing a directed graph using BFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using breadth-first search (BFS).
 */
template <typename GraphT>
class BfsView {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> startVertex_;

    using BfsIterator = TraversalIterator<GraphT, BfsQueueWrapper<GraphT>, ChildIterator<GraphT>>;

  public:
    BfsView(const GraphT &graph, const VertexIdxT<GraphT> &start) : graph_(graph), startVertex_(start) {}

    auto begin() const { return BfsIterator(graph_, startVertex_); }

    auto end() const { return BfsIterator(graph_, graph_.NumVertices()); }

    auto size() const { return graph_.NumVertices(); }
};

template <typename GraphT>
struct DfsStackWrapper {
    std::vector<VertexIdxT<GraphT>> stack_;

    void Push(const VertexIdxT<GraphT> &v) { stack_.push_back(v); }

    VertexIdxT<GraphT> PopNext() {
        auto v = stack_.back();
        stack_.pop_back();
        return v;
    }

    bool empty() const { return stack_.empty(); }
};

/**
 * @brief Views for traversing a directed graph using DFS.
 *
 * These classes provide iterators to traverse the vertices of a directed graph strating from a given vertex
 * using depth-first search (DFS).
 */
template <typename GraphT>
class DfsView {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");

    const GraphT &graph_;
    VertexIdxT<GraphT> startVertex_;

    using DfsIterator = TraversalIterator<GraphT, DfsStackWrapper<GraphT>, ChildIterator<GraphT>>;

  public:
    DfsView(const GraphT &graph, const VertexIdxT<GraphT> &start) : graph_(graph), startVertex_(start) {}

    auto begin() const { return DfsIterator(graph_, startVertex_); }

    auto end() const { return DfsIterator(graph_, graph_.NumVertices()); }

    auto size() const { return graph_.NumVertices(); }
};

template <typename GraphT>
struct ParentsIterator {
    const GraphT &graph_;

    ParentsIterator(const GraphT &graph) : graph_(graph) {}

    inline auto Iterate(const VertexIdxT<GraphT> &v) const { return graph_.Parents(v); }
};

/**
 * @brief Computes the weakly connected components of a directed graph.
 *
 * A weakly connected component is a maximal subgraph where for any two vertices
 * u, v in the subgraph, there is a path between u and v in the underlying
 * undirected graph.
 *
 * @tparam GraphT The type of the graph, which must satisfy the `directed_graph` concept.
 * @param graph The input directed graph.
 * @param[out] components A vector where `components[i]` will be the component ID for vertex `i`.
 * @return The total number of weakly connected components.
 */
template <typename GraphT>
std::size_t ComputeWeaklyConnectedComponents(const GraphT &graph, std::vector<VertexIdxT<GraphT>> &components) {
    static_assert(isDirectedGraphV<GraphT>, "GraphT must satisfy the directed_graph concept");
    using VertexType = VertexIdxT<GraphT>;

    if (graph.NumVertices() == 0) {
        components.clear();
        return 0;
    }

    components.assign(graph.NumVertices(), std::numeric_limits<VertexType>::max());
    VertexType componentId = 0;

    for (const auto &v : graph.Vertices()) {
        if (components[v] == std::numeric_limits<VertexType>::max()) {
            std::vector<VertexType> q;
            q.push_back(v);
            components[v] = componentId;
            size_t head = 0;

            while (head < q.size()) {
                VertexType u = q[head++];
                for (const auto &neighbor : graph.Parents(u)) {
                    if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                        components[neighbor] = componentId;
                        q.push_back(neighbor);
                    }
                }
                for (const auto &neighbor : graph.Children(u)) {
                    if (components[neighbor] == std::numeric_limits<VertexType>::max()) {
                        components[neighbor] = componentId;
                        q.push_back(neighbor);
                    }
                }
            }
            componentId++;
        }
    }
    return componentId;
}

/**
 * @brief Counts the number of weakly connected components in a directed graph.
 * @param graph The input directed graph.
 * @return The number of weakly connected components.
 */
template <typename GraphT>
std::size_t CountWeaklyConnectedComponents(const GraphT &graph) {
    std::vector<VertexIdxT<GraphT>> components;
    return ComputeWeaklyConnectedComponents(graph, components);
}

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_DIRECTED_GRAPH_UTIL_HPP
