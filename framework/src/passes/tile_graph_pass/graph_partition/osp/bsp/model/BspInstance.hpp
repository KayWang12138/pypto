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
* \file BspInstance.hpp
* \brief
*/

#pragma once

#include <iostream>

#include "BspArchitecture.hpp"
#include "passes/tile_graph_pass/graph_partition/osp/concepts/computational_dag_concept.hpp"
#include "passes/tile_graph_pass/graph_partition/osp/graph_algorithms/computational_dag_construction_util.hpp"
#include "passes/tile_graph_pass/graph_partition/osp/graph_algorithms/computational_dag_util.hpp"

namespace osp {

/**
 * @class BspInstance
 * @brief Represents a scheduling problem instance for the Bulk Synchronous Parallel (BSP) model.
 *
 * The BspInstance class serves as a container for all the necessary information to define a
 * BSP scheduling problem. It acts as the "ground" object that holds the actual implementation
 * of the graph and architecture.
 *
 * It aggregates three main components:
 *
 * 1. **Computational DAG**: The directed acyclic graph representing the program to be executed.
 *    It defines the tasks (nodes), their dependencies (directed edges), and associated weights (work, memory, communication).
 *
 * 2. **BSP Architecture**: The hardware model description, including the number of processors,
 *    their types, memory bounds, and communication/synchronization costs.
 *    Note that processor indices are represented using `unsigned`.
 *
 * 3. **Node-Processor Compatibility**: A matrix defining which node types can be executed on which
 *    processor types. This enables the modeling of heterogeneous systems (e.g., CPU + GPU) where
 *    certain nodes are restricted to specific hardware accelerators.
 *
 * @warning Be careful when assigning an existing graph to a BspInstance. Depending on the
 * constructor or assignment operator used, this may result in a deep copy of the graph structure,
 * which can be expensive for large graphs.
 *
 * This class provides a unified interface to access and modify these components, facilitating
 * the development of scheduling algorithms that need to query problem constraints and properties.
 *
 * @tparam Graph_t The type of the computational DAG, which must satisfy the `is_computational_dag` concept.
 */
template<typename Graph_t>
class BspInstance {
    static_assert(is_computational_dag_v<Graph_t>, "BspInstance can only be used with computational DAGs.");

  private:
    /**
     * @brief The computational DAG representing the program structure.
     *
     * It contains the graph topology (nodes and directed edges) as well as attributes such as node types,
     * work weights, memory weights, and edge communication weights.
     */
    Graph_t cdag;
    /**
     * @brief The BSP architecture model.
     *
     * It defines the hardware characteristics including processor types, memory limits,
     * communication bandwidth/latency (send costs), and global synchronization costs.
     */
    BspArchitecture<Graph_t> architecture;

    /**
     * @brief Stores the compatibility between node types and processor types.
     *
     * The architecture defines a type for each processor, and the DAG defines a type for each node.
     * This matrix stores for each node type and processor type whether they are compatible, i.e.,
     * if a node of that type can be assigned to a processor of the given type in a schedule.
     * @note The outer vector is indexed by node type, the inner vector is indexed by processor type.
     */
    std::vector<std::vector<bool>> nodeProcessorCompatibility = std::vector<std::vector<bool>>({{true}});

    /**
     * @brief The type of the vectex types in the computational DAG.
     * If the DAG does not support vertex types, this is `unsigned`.
     */
    using vertex_type_t_or_default = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
    using processor_type_t = unsigned;

  public:
    /**
     * @brief Default constructor for the BspInstance class.
     */
    BspInstance() = default;

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are copied!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(const Graph_t &cdag_, const BspArchitecture<Graph_t> &architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(cdag_), architecture(architecture_), nodeProcessorCompatibility(nodeProcessorCompatibility_) {}

    /**
     * @brief Constructs a BspInstance object with the specified computational DAG and BSP architecture.
     * Computational DAG and BSP architecture are moved!
     *
     * @param cdag The computational DAG for the instance.
     * @param architecture The BSP architecture for the instance.
     */
    BspInstance(Graph_t &&cdag_, BspArchitecture<Graph_t> &&architecture_,
                std::vector<std::vector<bool>> nodeProcessorCompatibility_ = std::vector<std::vector<bool>>({{true}}))
        : cdag(std::move(cdag_)), architecture(std::move(architecture_)), nodeProcessorCompatibility(nodeProcessorCompatibility_) {
    }

    template<typename Graph_t_other>
    explicit BspInstance(const BspInstance<Graph_t_other> &other)
        : architecture(other.getArchitecture()),
          nodeProcessorCompatibility(other.getNodeProcessorCompatibilityMatrix()) {
        constructComputationalDag(other.getComputationalDag(), cdag);
    }

    BspInstance(const BspInstance<Graph_t> &other) = default;
    BspInstance(BspInstance<Graph_t> &&other) noexcept = default;

    BspInstance<Graph_t> &operator=(const BspInstance<Graph_t> &other) = default;
    BspInstance<Graph_t> &operator=(BspInstance<Graph_t> &&other) noexcept = default;

    /**
     * @brief Returns a reference to the BSP architecture of the instance.
     * Assigning the BSP architecture via the reference creates a copy of the architecture.
     * The move operator may be used to transfer ownership of the architecture.
     */
    [[nodiscard]] const BspArchitecture<Graph_t> &getArchitecture() const { return architecture; }
    [[nodiscard]] BspArchitecture<Graph_t> &getArchitecture() { return architecture; }

    /**
     * @brief Returns a reference to the computational DAG of the instance.
     * Assigning the computational DAG via the reference creates a copy of the DAG.
     * The move operator may be used to transfer ownership of the DAG.
     */
    [[nodiscard]] const Graph_t &getComputationalDag() const { return cdag; }
    [[nodiscard]] Graph_t &getComputationalDag() { return cdag; }

    /**
     * @brief Returns the number of vertices in the computational DAG.
     */
    [[nodiscard]] vertex_idx_t<Graph_t> numberOfVertices() const { return cdag.num_vertices(); }

    /**
     * @brief Returns a view over the vertex indices of the computational DAG.
     */
    [[nodiscard]] auto vertices() const { return cdag.vertices(); }

    /**
     * @brief Returns a view over the processor indices of the BSP architecture.
     */
    [[nodiscard]] auto processors() const { return architecture.processors(); }

    /**
     * @brief Returns the number of processors in the BSP architecture.
     */
    [[nodiscard]] unsigned numberOfProcessors() const { return architecture.numberOfProcessors(); }

    /**
     * @brief Returns the communication costs between two processors. Does not perform bounds checking.
     * The communication costs are the send costs multiplied by the communication costs.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.communicationCosts(p_send, p_receive);
    }

    /**
     * @brief Returns the send costs between two processors. Does not perform bounds checking.
     * Does not take communication costs into account.
     *
     * @param p_send The index of the sending processor.
     * @param p_receive The index of the receiving processor.
     */
    [[nodiscard]] v_commw_t<Graph_t> sendCosts(const unsigned p_send, const unsigned p_receive) const {
        return architecture.sendCosts(p_send, p_receive);
    }

    /**
     * @brief Returns a copy of the send costs matrix.
     */
    [[nodiscard]] std::vector<std::vector<v_commw_t<Graph_t>>> sendCosts() const { return architecture.sendCosts(); }

    /**
     * @brief Returns the flattened send costs vector.
     */
    [[nodiscard]] const std::vector<v_commw_t<Graph_t>> &sendCostsVector() const {
        return architecture.sendCostsVector();
    }

    /**
     * @brief Returns the communication costs of the BSP architecture.
     */
    [[nodiscard]] v_commw_t<Graph_t> communicationCosts() const { return architecture.communicationCosts(); }

    /**
     * @brief Returns the synchronization costs of the BSP architecture.
     */
    [[nodiscard]] v_commw_t<Graph_t> synchronisationCosts() const { return architecture.synchronisationCosts(); }

    /**
     * @brief Returns the memory bound for a specific processor.
     * @param proc The processor index.
     */
    [[nodiscard]] v_memw_t<Graph_t> memoryBound(const unsigned proc) const { return architecture.memoryBound(proc); }

    /**
     * @brief Sets the communication costs of the BSP architecture.
     * @param cost The communication costs to set.
     */
    void setCommunicationCosts(const v_commw_t<Graph_t> cost) { architecture.setCommunicationCosts(cost); }

    /**
     * @brief Sets the synchronisation costs of the BSP architecture.
     * @param cost The synchronisation costs to set.
     */
    void setSynchronisationCosts(const v_commw_t<Graph_t> cost) { architecture.setSynchronisationCosts(cost); }

    /**
     * @brief Sets the number of processors. Processor type is set to 0 for all processors.
     * Resets send costs to uniform (1) and diagonal to 0. The memory bound is set to 100 for all processors.
     * @param numberOfProcessors The number of processors. Must be greater than 0.
     * @throws std::invalid_argument if the number of processors is 0.
     */
    void setNumberOfProcessors(const unsigned num) { architecture.setNumberOfProcessors(num); }

    /**
     * @brief Returns the processor type for a given processor index. Does not perform bounds checking.
     * @param proc The processor index.
     */
    [[nodiscard]] vertex_type_t_or_default processorType(const unsigned proc) const { return architecture.processorType(proc); }

    /**
     * @brief Checks if a node is compatible with a processor. Does not perform bounds checking.
     *
     * @param node The node index.
     * @param processor_id The processor index.
     * @return True if the node is compatible with the processor, false otherwise.
     */
    [[nodiscard]] bool isCompatible(const vertex_idx_t<Graph_t> &node, const unsigned processor_id) const {
        return isCompatibleType(cdag.vertex_type(node), architecture.processorType(processor_id));
    }

    /**
     * @brief Checks if a node type is compatible with a processor type. Does not perform bounds checking.
     *
     * @param nodeType The node type.
     * @param processorType The processor type.
     * @return True if the node type is compatible with the processor type, false otherwise.
     */
    [[nodiscard]] bool isCompatibleType(const vertex_type_t_or_default nodeType, const processor_type_t processorType) const {
        return nodeProcessorCompatibility[nodeType][processorType];
    }

    /**
     * @brief Sets the node-processor compatibility matrix. The matrix is copied. Dimensions are not checked.
     * @param compatibility_ The compatibility matrix.
     */
    void setNodeProcessorCompatibility(const std::vector<std::vector<bool>> &compatibility_) {
        nodeProcessorCompatibility = compatibility_;
    }

    /**
     * @brief Returns the node-processor compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getNodeProcessorCompatibilityMatrix() const {
        return nodeProcessorCompatibility;
    }

    /**
     * @brief Returns the node type - processor type compatibility matrix.
     */
    [[nodiscard]] const std::vector<std::vector<bool>> &getProcessorCompatibilityMatrix() const { return nodeProcessorCompatibility; }

    /**
     * @brief Sets the compatibility matrix to be diagonal. This implies that node type `i` is only compatible with processor type `i`.
     * @param number_of_types The number of types.
     */
    void setDiagonalCompatibilityMatrix(const vertex_type_t_or_default number_of_types) {
        nodeProcessorCompatibility.assign(number_of_types, std::vector<bool>(number_of_types, false));
        for (vertex_type_t_or_default i = 0; i < number_of_types; ++i)
            nodeProcessorCompatibility[i][i] = true;
    }
};

} // namespace osp