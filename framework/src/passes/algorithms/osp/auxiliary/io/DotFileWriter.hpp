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
 * \file DotFileWriter.hpp
 * \brief
 */

#ifndef OSP_DOTFILEWRITER_HPP
#define OSP_DOTFILEWRITER_HPP

#include <fstream>
#include <string>

#include "passes/algorithms/osp/bsp/model/BspSchedule.hpp"
#include "passes/algorithms/osp/concepts/computational_dag_concept.hpp"

namespace npu::tile_fwk {
namespace osp {

class DotFileWriter {
  private:
    template <typename GraphT>

    struct EdgeWriterDot {
        const GraphT &graph_;

        EdgeWriterDot(const GraphT &graph) : graph_(graph) {}

        void operator()(std::ostream &out, const EdgeDescT<GraphT> &i) const {
            out << Source(i, graph_) << "->" << Target(i, graph_) << " ["
                << "comm_weight=\"" << graph_.EdgeCommWeight(i) << "\";"
                << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleDot {
        const BspSchedule<GraphT> &schedule_;

        VertexWriterScheduleDot(const BspSchedule<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexMemWeight(i) << "\";";

            if constexpr (hasTypedVerticesV<GraphT>) {
                out << "type=\"" << schedule_.GetInstance().GetComputationalDag().VertexType(i) << "\";";
            }

            out << "proc=\"" << schedule_.AssignedProcessor(i) << "\";" << "superstep=\"" << schedule_.AssignedSuperstep(i)
                << "\";";

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterGraphDot {
        const GraphT &graph_;

        VertexWriterGraphDot(const GraphT &graph) : graph_(graph) {}

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << graph_.VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << graph_.VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << graph_.VertexMemWeight(i) << "\";";

            if constexpr (hasTypedVerticesV<GraphT>) {
                out << "type=\"" << graph_.VertexType(i) << "\";";
            }

            out << "]";
        }
    };

    template <typename GraphT, typename ColorContainerT>
    struct ColoredVertexWriterGraphDot {
        const GraphT &graph_;
        const ColorContainerT &colors_;
        std::vector<std::string> colorStrings_;
        std::vector<std::string> shapeStrings_;

        ColoredVertexWriterGraphDot(const GraphT &graph, const ColorContainerT &colors) : graph_(graph), colors_(colors) {
            colorStrings_ = {"lightcoral",      "palegreen",   "lightblue",     "gold",
                             "orchid",          "sandybrown",  "aquamarine",    "burlywood",
                             "hotpink",         "yellowgreen", "skyblue",       "khaki",
                             "violet",          "salmon",      "turquoise",     "tan",
                             "deeppink",        "chartreuse",  "deepskyblue",   "lemonchiffon",
                             "magenta",         "orangered",   "cyan",          "wheat",
                             "mediumvioletred", "limegreen",   "dodgerblue",    "lightyellow",
                             "darkviolet",      "tomato",      "paleturquoise", "bisque",
                             "crimson",         "lime",        "steelblue",     "papayawhip",
                             "purple",          "darkorange",  "cadetblue",     "peachpuff",
                             "indianred",       "springgreen", "powderblue",    "cornsilk",
                             "mediumorchid",    "chocolate",   "darkturquoise", "navajowhite",
                             "firebrick",       "seagreen",    "royalblue",     "lightgoldenrodyellow",
                             "darkmagenta",     "coral",       "teal",          "moccasin",
                             "maroon",          "forestgreen", "blue",          "yellow",
                             "darkorchid",      "red",         "green",         "navy",
                             "darkred",         "darkgreen",   "mediumblue",    "ivory",
                             "indigo",          "orange",      "darkcyan",      "antiquewhite"};

            shapeStrings_ = {"oval", "rect", "hexagon", "parallelogram"};
        }

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            if (i >= static_cast<VertexIdxT<GraphT>>(colors_.size())) {
                // Fallback for safety: print without color if colors vector is mismatched or palette is empty.
                out << i << " [";
            } else {
                // Use modulo operator to cycle through the fixed palette if there are more color
                // groups than available colors.
                const std::string &color = colorStrings_[colors_[i] % colorStrings_.size()];
                out << i << " [style=filled;fillcolor=" << color << ";";
            }

            out << "work_weight=\"" << graph_.VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << graph_.VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << graph_.VertexMemWeight(i) << "\";";

            if constexpr (hasTypedVerticesV<GraphT>) {
                out << "type=\"" << graph_.VertexType(i) << "\";shape=\""
                    << shapeStrings_[graph_.VertexType(i) % shapeStrings_.size()] << "\";";
            }

            out << "]";
        }
    };

    template <typename GraphT, typename VertexWriterT>
    void WriteGraphStructure(std::ostream &os, const GraphT &graph, const VertexWriterT &vertexWriter) const {
        os << "digraph G {\n";
        for (const auto &v : graph.Vertices()) {
            vertexWriter(os, v);
            os << "\n";
        }

        if constexpr (hasEdgeWeightsV<GraphT>) {
            EdgeWriterDot<GraphT> edgeWriter(graph);

            for (const auto &e : Edges(graph)) {
                edgeWriter(os, e);
                os << "\n";
            }

        } else {
            for (const auto &v : graph.Vertices()) {
                for (const auto &child : graph.Children(v)) {
                    os << v << "->" << child << "\n";
                }
            }
        }
        os << "}\n";
    }

  public:
    /**
     * Constructs a DotFileWriter object with the given BspSchdule.
     *
     * @param schdule_
     */
    DotFileWriter() {}

    /**
     * Writes the BspSchedule to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template <typename GraphT>
    void WriteSchedule(std::ostream &os, const BspSchedule<GraphT> &schedule) const {
        WriteGraphStructure(os, schedule.GetInstance().GetComputationalDag(), VertexWriterScheduleDot<GraphT>(schedule));
    }

    /**
     * Writes the BspSchedule to the specified file in DOT format.
     *
     * @note The file will be overwritten if it already exists.
     * @note The file will be created if it does not exist.
     * @note This function is an alias for `write_dot(std::ostream &os)`
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param filename The name of the file to write the DOT representation of the computational DAG.
     */
    template <typename GraphT>
    void WriteSchedule(const std::string &filename, const BspSchedule<GraphT> &schedule) const {
        std::ofstream os(filename);
        WriteSchedule(os, schedule);
    }
 
    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(std::ostream &os, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(isComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        WriteGraphStructure(os, graph, ColoredVertexWriterGraphDot<GraphT, ColorContainerT>(graph, colors));
    }

    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(const std::string &filename, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(isComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        std::ofstream os(filename);
        WriteColoredGraph(os, graph, colors);
    }

    template <typename GraphT>
    void WriteGraph(std::ostream &os, const GraphT &graph) const {
        static_assert(isComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        WriteGraphStructure(os, graph, VertexWriterGraphDot<GraphT>(graph));
    }

    template <typename GraphT>
    void WriteGraph(const std::string &filename, const GraphT &graph) const {
        static_assert(isComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        std::ofstream os(filename);
        WriteGraph(os, graph);
    }
};

}    // namespace osp
} // namespace npu::tile_fwk
#endif // OSP_DOTFILEWRITER_HPP
