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
 * \file test_osp_algorithms.cpp
 * \brief Unit test for OSP Algorithm files.
 */

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>
#include <string>
#include "gtest/gtest.h"

#include "passes/algorithms/osp/auxiliary/datastructures/union_find.hpp"
#include "passes/algorithms/osp/auxiliary/math/divisors.hpp"
#include "passes/algorithms/osp/auxiliary/balanced_coin_flips.hpp"
#include "passes/algorithms/osp/auxiliary/hash_util.hpp"
#include "passes/algorithms/osp/auxiliary/permute.hpp"
#include "passes/algorithms/osp/bsp/model/BspArchitecture.hpp"

#include "passes/algorithms/osp/coarser/coarser_util.hpp"
#include "passes/algorithms/osp/coarser/sarkar/sarkar.hpp"
#include "passes/algorithms/osp/coarser/sarkar/sarkar_mul.hpp"

#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.hpp"

namespace npu::tile_fwk {
namespace osp {

using GraphType = CompactSparseGraph<>;

class OspAlgorithmTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(OspAlgorithmTest, UnionFind1) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;
    for (const auto &name : names) {
        testUniverse.AddObject(name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);

    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("d", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("d"), testUniverse.FindOriginByName("b"));

    testUniverse.JoinByName("a", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("a", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);
    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));
    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));
}

TEST_F(OspAlgorithmTest, UnionFind2) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;

    for (auto &name : names) {
        testUniverse.AddObject(name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    for (auto &name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 9);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 8);
    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 7);
    testUniverse.JoinByName("c", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);
    testUniverse.JoinByName("d", "e");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);
    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("c", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("g", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    testUniverse.JoinByName("h", "i");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);

    testUniverse.JoinByName("b", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 1);

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("h"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("i"));
    EXPECT_EQ(testUniverse.FindOriginByName("f"), testUniverse.FindOriginByName("g"));

    testUniverse.Reset();
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 0);
}

TEST_F(OspAlgorithmTest, UnionFind3) {
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    UnionFindUniverse<std::string, unsigned, unsigned> testUniverse;
    for (std::size_t i = 0; i < names.size(); ++i) {
        testUniverse.AddObject(names[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        EXPECT_EQ(testUniverse.FindOriginByName(names[i]), names[i]);
        EXPECT_EQ(testUniverse.GetWeightOfComponentByName(names[i]), weights[i]);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    testUniverse.JoinByName("b", "c");
    testUniverse.JoinByName("d", "b");
    testUniverse.JoinByName("a", "c");
    testUniverse.JoinByName("a", "d");

    testUniverse.JoinByName("e", "f");

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("a"), 7);
    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("b"), 7);
    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("e"), 2);

    std::vector<std::vector<std::string>> components = testUniverse.GetConnectedComponents();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto &comp : components) {
        totalCompWeights += testUniverse.GetWeightOfComponentByName(comp.at(0));
        totalElements += static_cast<unsigned>(comp.size());
        for (auto &name : comp) {
            EXPECT_TRUE(std::any_of(
                names.cbegin(), names.cend(), [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto &wt : weights) {
        totalWeight += wt;
    }

    EXPECT_EQ(totalElements, names.size());
    EXPECT_EQ(totalWeight, totalCompWeights);

    for (auto &name : names) {
        EXPECT_TRUE(std::any_of(components.cbegin(), components.cend(), [name](std::vector<std::string> comp) {
            return std::any_of(
                comp.cbegin(), comp.cend(), [name](std::string other_name) { return name == other_name; });
        }));
    }
}

TEST_F(OspAlgorithmTest, IntSqrt) {
    for (std::size_t root = 1U; root < 200U; ++root) {
        for (std::size_t num = root * root; num < (root + 1U) * (root + 1U); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }

    for (int root = 1; root < 300; ++root) {
        for (int num = root * root; num < (root + 1) * (root + 1); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }
}

TEST_F(OspAlgorithmTest, Divisors) {
    for (std::size_t num = 1U; num < 1000U; ++num) {
        const std::vector<std::size_t> divs = DivisorsList(num);
        for (const std::size_t &div : divs) {
            EXPECT_EQ(num % div, 0U);
        }

        auto it = divs.begin();
        for (std::size_t i = 1U; i <= num; ++i) {
            if (num % i == 0) {
                EXPECT_TRUE(it != divs.end());
                EXPECT_EQ(i, *it);
                ++it;
            }
        }
        EXPECT_TRUE(it == divs.end());
    }
}

bool thueMorseGen(long unsigned int n) {
    unsigned long int bin_sum = 0;
    while (n != 0) {
        bin_sum += n % 2;
        n /= 2;
    }
    return bool(bin_sum % 2);
}

TEST_F(OspAlgorithmTest, RandomBiasedCoin) {
    BiasedRandom Coin;
    bool valAnd = true;
    bool valOr = false;
    for (int i = 0; i < 1000; i++) {
        const bool flip = Coin.GetFlip();
        valAnd &= flip;
        valOr |= flip;
    }

    // Can both technically fail but insanely unlikely
    EXPECT_FALSE(valAnd);
    EXPECT_TRUE(valOr);
}

TEST_F(OspAlgorithmTest, ThueMorse) {
    ThueMorseSequence Coin(0);

    std::vector<bool> beginning(
        {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1});
    std::vector<bool> generated;
    for (long unsigned i = 0; i < beginning.size(); i++) {
        const bool next = Coin.GetFlip();
        generated.emplace_back(next);
    }

    EXPECT_TRUE(beginning == generated);

    ThueMorseSequence Test_Coin_in_seq(0);
    for (unsigned i = 0; i < 200u; i++) {
        EXPECT_EQ(Test_Coin_in_seq.GetFlip(), thueMorseGen(i));
    }
}

TEST_F(OspAlgorithmTest, CombineHashes) {
    std::size_t hash1 = 1729U;
    std::size_t hash2 = 1729U;
    HashCombine(hash1, 1U);
    HashCombine(hash2, 2U);
    // Can technically fail but is highly unlikely
    EXPECT_NE(hash1, hash2);
}

TEST_F(OspAlgorithmTest, InPlaceInversePermutationRandom) {
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned i = 0; i < 5U; ++i) {
        std::shuffle(vec.begin(), vec.end(), gen);

        std::vector<unsigned> invPerm(vec.size());
        for (unsigned j = 0; j < vec.size(); ++j) {
            invPerm[vec[j]] = j;
        }

        InversePermuteInplace(vec, invPerm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            EXPECT_EQ(vec[j], sol[j]);
        }
    }
}

TEST_F(OspAlgorithmTest, InPlaceInversePermutationChar) {
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4, 0, 1, 2, 3, 6, 5});
    std::vector<char> sol({'e', 'a', 'b', 'c', 'd', 'g', 'f'});

    InversePermuteInplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        EXPECT_EQ(vec[j], sol[j]);
    }
}

TEST_F(OspAlgorithmTest, Architecture) {
    std::vector<std::vector<unsigned>> uniformSentCosts = {
        {0, 1, 1, 1},
        {1, 0, 1, 1},
        {1, 1, 0, 1},
        {1, 1, 1, 0}
    };

    BspArchitecture<GraphType> architecture;
    architecture.SetNumberOfProcessors(4);
    architecture.SetCommunicationCosts(2);
    architecture.SetSynchronisationCosts(3);

    EXPECT_EQ(architecture.NumberOfProcessors(), 4);
    EXPECT_EQ(architecture.CommunicationCosts(), 2);
    EXPECT_EQ(architecture.SynchronisationCosts(), 3);
    EXPECT_EQ(architecture.GetMemoryConstraintType(), MemoryConstraintType::NONE);
    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 1);

    EXPECT_EQ(architecture.MemoryBound(0), 100);
    EXPECT_EQ(architecture.MemoryBound(1), 100);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 100);
    architecture.SetMemoryBound(200, 3);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 200);

    EXPECT_EQ(architecture.ProcessorTypes()[0], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[1], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[2], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[3], 0);

    EXPECT_EQ(architecture.ProcessorType(0), 0);
    EXPECT_EQ(architecture.ProcessorType(1), 0);
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 0);
    architecture.SetProcessorsWithTypes({0, 0, 0, 1});
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 1);

    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);

    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 2);

    EXPECT_TRUE(architecture.SendCost() == uniformSentCosts);

    std::vector<std::vector<unsigned>> expectedSendCosts = {
        {0, 2, 2, 2},
        {2, 0, 2, 2},
        {2, 2, 0, 2},
        {2, 2, 2, 0}
    };

    architecture.SetSendCosts(expectedSendCosts);
    EXPECT_TRUE(architecture.SendCost() == expectedSendCosts);

    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 4);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);

    architecture.SetSendCosts(uniformSentCosts);
    EXPECT_TRUE(architecture.SendCost() == uniformSentCosts);

    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);
}

TEST_F(OspAlgorithmTest, EmptyGraph) {
    GraphType graph;

    EXPECT_EQ(graph.NumVertices(), 0);
    EXPECT_EQ(graph.NumEdges(), 0);
}

TEST_F(OspAlgorithmTest, NoEdgesGraph) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({});

    GraphType graph(10, edges);

    EXPECT_EQ(graph.NumVertices(), 10);
    EXPECT_EQ(graph.NumEdges(), 0);
}

TEST_F(OspAlgorithmTest, LineGraph) {
    const std::set<std::pair<std::size_t, std::size_t>> edges({
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6},
        {6, 7}
    });

    GraphType graph(8, edges);

    EXPECT_EQ(graph.NumVertices(), 8);
    EXPECT_EQ(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr);
        ++cntr;
    }
    EXPECT_EQ(graph.NumVertices(), cntr);

    for (const auto &vert : graph.Vertices()) {
        if (vert != 7) {
            EXPECT_EQ(graph.OutDegree(vert), 1);
            for (const std::size_t &chld : graph.Children(vert)) {
                EXPECT_EQ(chld, vert + 1);
            }
            auto chldren = graph.Children(vert);
            EXPECT_EQ(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                EXPECT_EQ(*it, vert + 1);
            }

        } else {
            EXPECT_EQ(graph.OutDegree(vert), 0);
            for (const std::size_t &chld : graph.Children(vert)) {
                EXPECT_TRUE(false);
                EXPECT_EQ(chld, 100);
            }
            auto chldren = graph.Children(vert);
            EXPECT_EQ(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
            for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
                EXPECT_TRUE(false);
                EXPECT_EQ(*it, 100);
            }
        }
    }
    for (const auto &vert : graph.Vertices()) {
        if (vert != 0) {
            EXPECT_EQ(graph.InDegree(vert), 1);
            for (const std::size_t &par : graph.Parents(vert)) {
                EXPECT_EQ(par, vert - 1);
            }
            auto prnts = graph.Parents(vert);
            EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                EXPECT_EQ(*it, vert - 1);
            }
        } else {
            EXPECT_EQ(graph.InDegree(vert), 0);
            for (const std::size_t &par : graph.Parents(vert)) {
                EXPECT_TRUE(false);
                EXPECT_EQ(par, 100);
            }
            auto prnts = graph.Parents(vert);
            EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
            for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
                EXPECT_TRUE(false);
                EXPECT_EQ(*it, 100);
            }
        }
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

TEST_F(OspAlgorithmTest, Graph1) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    GraphType graph(11, edges);

    EXPECT_EQ(graph.NumVertices(), 11);
    EXPECT_EQ(graph.NumEdges(), 11);

    std::size_t cntr0 = 0;
    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr0);
        ++cntr0;
    }
    EXPECT_EQ(graph.NumVertices(), cntr0);

    const std::vector<std::vector<std::size_t>> outEdges({
        {1, 2},
        {2, 6},
        {3},
        {7},
        {6},
        {6},
        {7, 10},
        {9},
        {},
        {},
        {}
    });

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &chld : graph.Children(vert)) {
            EXPECT_EQ(chld, outEdges[vert][cntr]);
            ++cntr;
        }
        auto chldrn = graph.Children(vert);
        EXPECT_EQ(chldrn.crend() - chldrn.crbegin(), graph.OutDegree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            --cntr;
            EXPECT_EQ(*it, outEdges[vert][cntr]);
        }
    }

    const std::vector<std::vector<std::size_t>> inEdges({
        {},
        {0},
        {0, 1},
        {2},
        {},
        {},
        {1, 4, 5},
        {3, 6},
        {},
        {7},
        {6}
    });

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto &par : graph.Parents(vert)) {
            EXPECT_EQ(par, inEdges[vert][cntr]);
            ++cntr;
        }
        auto prnts = graph.Parents(vert);
        EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            --cntr;
            EXPECT_EQ(*it, inEdges[vert][cntr]);
        }
    }

    for (const auto &vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

TEST_F(OspAlgorithmTest, GraphWorkWeights) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    GraphType graph(11, edges);
    for (auto vert : graph.Vertices()) {
        graph.SetVertexWorkWeight(vert, ww[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexWorkWeight(vert), ww[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexWorkWeight(vert, wt);
        EXPECT_EQ(graph.VertexWorkWeight(vert), wt);
    }
}

TEST_F(OspAlgorithmTest, GraphCommWeights) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    GraphType graph(11, edges);
    for (auto vert : graph.Vertices()) {
        graph.SetVertexCommWeight(vert, cw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexCommWeight(vert), cw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexCommWeight(vert, wt);
        EXPECT_EQ(graph.VertexCommWeight(vert), wt);
    }
}

TEST_F(OspAlgorithmTest, GraphMemWeights) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    GraphType graph(11, edges);

    for (auto vert : graph.Vertices()) {
        graph.SetVertexMemWeight(vert, mw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexMemWeight(vert), mw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexMemWeight(vert, wt);
        EXPECT_EQ(graph.VertexMemWeight(vert), wt);
    }
}

TEST_F(OspAlgorithmTest, GraphVtype) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    std::vector<unsigned> vt(11);
    std::iota(vt.begin(), vt.end(), 33);

    GraphType graph(11, edges);

    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), vt[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexType(vert, wt);
        EXPECT_EQ(graph.VertexType(vert), wt);
    }
}

TEST_F(OspAlgorithmTest, ExpansionMapValidity) {
    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap1 = {{0}, {1}, {2}, {3}};
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap1));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap2 = {{0}, {2}, {3}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap2));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap3 = {
        {0, 3}
    };
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap3));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap4 = {
        {0, 3},
        {2, 1, 4},
        {5}
    };
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap4));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap5 = {{0}, {}, {2}, {3}, {1}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap5));
}

TEST_F(OspAlgorithmTest, ContractionMapValidity) {
    const std::vector<VertexIdxT<GraphType>> contractionMap1 = {0, 1, 2, 3};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap1));

    const std::vector<VertexIdxT<GraphType>> contractionMap2 = {0, 1, 1, 1};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap2));

    const std::vector<VertexIdxT<GraphType>> contractionMap3 = {0, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap3));

    const std::vector<VertexIdxT<GraphType>> contractionMap4 = {2, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap4));
}

TEST_F(OspAlgorithmTest, ContractionMapCoarsening) {
    std::set<std::pair<VertexIdxT<GraphType>, VertexIdxT<GraphType>>> edges({
        {0, 1},
        {1, 2}
    });
    GraphType graph(6, edges);

    GraphType coarseGraph1;

    std::vector<VertexIdxT<GraphType>> contractionMap({0, 0, 1, 1, 2, 3});
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));
    EXPECT_TRUE(coarser_util::ConstructCoarseDag(graph, coarseGraph1, contractionMap));
    EXPECT_TRUE(contractionMap == std::vector<VertexIdxT<GraphType>>({0, 0, 1, 1, 2, 3}));

    EXPECT_EQ(coarseGraph1.NumVertices(), 4);
    EXPECT_EQ(coarseGraph1.NumEdges(), 1);

    EXPECT_EQ(coarseGraph1.OutDegree(0), 1);
    EXPECT_EQ(coarseGraph1.OutDegree(1), 0);
    EXPECT_EQ(coarseGraph1.OutDegree(2), 0);

    EXPECT_EQ(coarseGraph1.InDegree(0), 0);
    EXPECT_EQ(coarseGraph1.InDegree(1), 1);
    EXPECT_EQ(coarseGraph1.InDegree(2), 0);

    for (const auto &vert : coarseGraph1.Children(0)) {
        EXPECT_EQ(vert, 1);
    }

    for (const auto &vert : coarseGraph1.Parents(1)) {
        EXPECT_EQ(vert, 0);
    }
}

TEST_F(OspAlgorithmTest, TestTopSort) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });

    const GraphType graph(11, edges);

    std::vector<VertexIdxT<GraphType>> verts(graph.NumVertices());
    std::iota(verts.begin(), verts.end(), 0);
    const auto topOrderVec = GetTopOrder<GraphType>(graph);
    EXPECT_TRUE(std::is_permutation(topOrderVec.cbegin(), topOrderVec.cend(), verts.cbegin(), verts.cend()));
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(topOrderVec.cbegin(), topOrderVec.cend(), vert),
                                    std::find(topOrderVec.cbegin(), topOrderVec.cend(), chld)),
                      0);
        }
    }
}

void testCoarseningAlgorithm(Coarser<GraphType, GraphType> &coarser) {
    const std::vector<std::pair<std::size_t, std::size_t>> edges({
        {0,  1},
        {2,  3},
        {6, 10},
        {7,  9},
        {0,  2},
        {4,  6},
        {1,  6},
        {6,  7},
        {5,  6},
        {3,  7},
        {1,  2}
    });
    std::vector<unsigned> vt(11, 0);
    vt[0] = 1U;
    vt[1] = 1U;
    vt[2] = 1U;

    GraphType graph(11, edges);
    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    GraphType coarseGraph;
    std::vector<VertexIdxT<GraphType>> contractionMap;

    EXPECT_TRUE(coarser.CoarsenDag(graph, coarseGraph, contractionMap));
    EXPECT_EQ(contractionMap.size(), graph.NumVertices());
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), coarseGraph.VertexType(contractionMap[vert]));
    }

    // Acyclic check
    std::vector<VertexIdxT<GraphType>> coarseVerts(coarseGraph.NumVertices());
    std::iota(coarseVerts.begin(), coarseVerts.end(), 0);
    const auto coarseTopOrder = GetTopOrder<GraphType>(coarseGraph);
    EXPECT_TRUE(std::is_permutation(coarseTopOrder.cbegin(), coarseTopOrder.cend(), coarseVerts.cbegin(), coarseVerts.cend()));
    for (const auto vert : coarseGraph.Vertices()) {
        for (const auto chld : coarseGraph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), vert),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), chld)),
                      0);
        }
    }
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GE(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[vert]),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[chld])),
                      0);
        }
    }

    // Grouping of Vertex Types
    std::vector<unsigned> coarseTypes(coarseGraph.NumVertices(), std::numeric_limits<unsigned>::max());
    for (const auto vert : graph.Vertices()) {
        const auto coarseVert = contractionMap[vert];
        if (coarseTypes[coarseVert] != std::numeric_limits<unsigned>::max()) {
            EXPECT_EQ(coarseTypes[coarseVert], graph.VertexType(vert));
        }
        coarseTypes[coarseVert] = graph.VertexType(vert);
    }
}

TEST_F(OspAlgorithmTest, CoarsenSarkar) {
    sarkar_params::Parameters<VWorkwT<GraphType>> params;
    params.mode_ = sarkar_params::Mode::LINES;
    params.commCost_ = 100;
    params.useTopPoset_ = true;

    Sarkar<GraphType, GraphType> coarser(params);

    testCoarseningAlgorithm(coarser);

    params.useTopPoset_ = false;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_EVEN;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_ODD;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::HOMOGENEOUS_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);
}


TEST_F(OspAlgorithmTest, CoarsenSarkarML) {
    sarkar_params::MulParameters<VWorkwT<GraphType>> params;
    params.commCostVec_ = {100};

    SarkarMul<GraphType, GraphType> coarser;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.commCostVec_ = {1, 2, 10, 50, 100};
    params.bufferMergeMode_ = sarkar_params::BufferMergeMode::FULL;
    coarser.SetParameters(params);

    testCoarseningAlgorithm(coarser);
}

} // namespace osp
} // namespace npu::tile_fwk