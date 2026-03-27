#ifndef CORE_ESTIMATOR_HPP
#define CORE_ESTIMATOR_HPP

#include <vector>
#include <set>
#include <queue>
#include <algorithm>

inline std::pair<int, int> estimateRequiredCores(
    int subgraphNum,
    const std::vector<bool>& isCubeGraph,
    const std::vector<std::set<int>>& subgraphOutGraph,
    const std::vector<int>& subgraphLatency
) {
    if (subgraphNum == 0) {
        return {0, 0};
    }

    std::vector<int> inDegree(subgraphNum, 0);
    for (int i = 0; i < subgraphNum; ++i) {
        for (int consumer : subgraphOutGraph[i]) {
            inDegree[consumer]++;
        }
    }

    std::vector<int> earliestStart(subgraphNum, 0);
    std::queue<int> q;
    
    for (int i = 0; i < subgraphNum; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
            earliestStart[i] = 0;
        }
    }

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        int endTime = earliestStart[node] + subgraphLatency[node];
        
        for (int consumer : subgraphOutGraph[node]) {
            earliestStart[consumer] = std::max(earliestStart[consumer], endTime);
            inDegree[consumer]--;
            if (inDegree[consumer] == 0) {
                q.push(consumer);
            }
        }
    }

    std::vector<std::tuple<int, bool, int>> events;
    for (int i = 0; i < subgraphNum; ++i) {
        int startTime = earliestStart[i];
        int endTime = startTime + subgraphLatency[i];
        events.push_back({startTime, true, i});
        events.push_back({endTime, false, i});
    }

    auto cmp = [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    };
    std::sort(events.begin(), events.end(), cmp);

    int maxCCores = 0;
    int maxVCores = 0;
    int currentCCores = 0;
    int currentVCores = 0;

    for (const auto& event : events) {
        bool isStart = std::get<1>(event);
        int nodeId = std::get<2>(event);
        
        if (isStart) {
            if (isCubeGraph[nodeId]) {
                currentCCores++;
                maxCCores = std::max(maxCCores, currentCCores);
            } else {
                currentVCores++;
                maxVCores = std::max(maxVCores, currentVCores);
            }
        } else {
            if (isCubeGraph[nodeId]) {
                currentCCores--;
            } else {
                currentVCores--;
            }
        }
    }

    return {maxCCores, maxVCores};
}

#endif