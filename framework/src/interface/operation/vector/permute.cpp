#include "interface/utils/operator_tracer.h"
#include "interface/operation/operation_common.h"
#include "tensor_transformation.h"
#include <algorithm>
#include "unary.h"
namespace npu::tile_fwk {

void validateAndNormalizePermutation(std::vector<int> &perm, int shapeSize);
bool isIdentityPermutation(const std::vector<int> &perm);
int findTargetPosition(const std::vector<int> &invPerm, int targetIndex, int startSearch);
Tensor permuteTensor(const Tensor &self, const std::vector<int> &perm, int shapeSize);
Tensor permuteAnyDims(Tensor tensor, std::vector<int> invPerm);
int calculateMinTransposeCount(const std::vector<int> &perm);
Tensor applyOptimalTransposeSequence(const Tensor &self, const std::vector<int> &perm, int shapeSize);

struct MergeAxisOptimization {
    bool canOptimize = false;
    int headMergeEnd = 0;
    int tailMergeStart = 0;
    std::vector<int> mergedPerm;
    std::vector<int64_t> mergedInputShape;
    std::vector<int64_t> mergedOutputShape;
    std::vector<int64_t> resultShape;
};

MergeAxisOptimization analyzeMergeAxisOptimization(
    const std::vector<int64_t> &shape, const std::vector<int> &perm) {
    const int shapeSize = shape.size();
    MergeAxisOptimization result;
    
    std::vector<int> tailDimPositions;
    for (int i = 0; i < shapeSize; ++i) {
        if (perm[i] >= shapeSize - 2) {
            tailDimPositions.push_back(i);
        }
    }
    
    if (tailDimPositions.empty()) {
        return result;
    }
    
    bool isConsecutive = true;
    for (int i = 1; i < static_cast<int>(tailDimPositions.size()); ++i) {
        if (tailDimPositions[i] != tailDimPositions[i-1] + 1) {
            isConsecutive = false;
            break;
        }
    }
    
    if (!isConsecutive) {
        return result;
    }
    
    int tailStart = tailDimPositions[0];
    int tailEnd = tailDimPositions.back();
    
    if (tailStart == 0 && tailEnd == shapeSize - 1) {
        return result;
    }
    
    result.canOptimize = true;
    result.tailMergeStart = tailStart;
    
    int64_t headSize = 1;
    for (int i = 0; i < tailStart; ++i) {
        headSize *= shape[i];
    }
    
    std::vector<int64_t> tailShape;
    for (int i = tailStart; i < shapeSize; ++i) {
        tailShape.push_back(shape[perm[i]]);
    }
    
    if (headSize > 1) {
        result.mergedInputShape.push_back(headSize);
    }
    for (int i = tailStart; i < shapeSize; ++i) {
        result.mergedInputShape.push_back(shape[i]);
    }
    
    int mergedDimCount = result.mergedInputShape.size();
    int tailGroupSize = shapeSize - tailStart;
    
    if (tailStart == 0) {
        result.mergedPerm.resize(mergedDimCount);
        for (int i = 0; i < mergedDimCount; ++i) {
            result.mergedPerm[i] = i;
        }
        if (mergedDimCount == 2) {
            result.mergedPerm = {1, 0};
        }
    } else {
        int tailGroupNewStart = 0;
        for (int i = 0; i < tailStart; ++i) {
            if (perm[i] >= shapeSize - tailGroupSize) {
                tailGroupNewStart++;
            }
        }
        
        result.mergedPerm.clear();
        if (headSize > 1) {
            for (int i = tailGroupNewStart; i < mergedDimCount - tailGroupSize; ++i) {
                result.mergedPerm.push_back(i);
            }
            for (int i = 0; i < tailGroupNewStart; ++i) {
                result.mergedPerm.push_back(mergedDimCount - tailGroupSize + i);
            }
            for (int i = mergedDimCount - tailGroupSize + tailGroupNewStart; i < mergedDimCount; ++i) {
                result.mergedPerm.push_back(i - tailGroupSize + tailGroupNewStart);
            }
        }
    }
    
    for (int i = 0; i < shapeSize; ++i) {
        result.resultShape.push_back(shape[perm[i]]);
    }
    
    return result;
}

Tensor applyMergeAxisTranspose(const Tensor &self, const std::vector<int> &perm, const MergeAxisOptimization &opt) {
    auto mergedTensor = Reshape(self, opt.mergedInputShape);
    
    Tensor transposedTensor;
    if (opt.mergedPerm.size() == 2 && opt.mergedPerm[0] == 1 && opt.mergedPerm[1] == 0) {
        transposedTensor = Transpose(mergedTensor, {0, 1});
    } else if (opt.mergedPerm.size() >= 2) {
        std::vector<int> invPerm(opt.mergedPerm.size());
        for (size_t i = 0; i < opt.mergedPerm.size(); ++i) {
            invPerm[opt.mergedPerm[i]] = static_cast<int>(i);
        }
        transposedTensor = permuteAnyDims(mergedTensor, invPerm);
    } else {
        transposedTensor = mergedTensor;
    }
    
    return Reshape(transposedTensor, opt.resultShape);
}

Tensor Permute(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();
    const int shapeSize = static_cast<int>(self.GetShape().size());
    ASSERT(perm.size() == static_cast<size_t>(shapeSize)) << "Permute dim num should match input dim num.";
    validateAndNormalizePermutation(perm, shapeSize);
    if (isIdentityPermutation(perm)) {
        return self;
    }
    if (shapeSize <= 5) {
        return permuteTensor(self, perm, shapeSize);
    }
    ASSERT(false) << "Permute only supports up to 5 dimensions.";
    return self;
}

void validateAndNormalizePermutation(std::vector<int> &perm, int shapeSize) {
    std::vector<bool> used(shapeSize, false);
    for (int &p : perm) {
        if (p < 0) {
            p += shapeSize;
        }
        ASSERT(p >= 0 && p < shapeSize) << "Permute dim is invalid.";
        ASSERT(!used[p]) << "Permute dims contain duplicate values.";
        used[p] = true;
    }
}

bool isIdentityPermutation(const std::vector<int> &perm) {
    if (perm.size() == 1) {
        return true;
    }
    return std::is_sorted(perm.begin(), perm.end());
}

int findTargetPosition(const std::vector<int> &invPerm, int targetIndex, int startSearch) {
    for (int j = startSearch; j < static_cast<int>(invPerm.size()); ++j) {
        if (invPerm[j] == targetIndex) {
            return j;
        }
    }
    return -1;
}

Tensor permuteTensor(const Tensor &self, const std::vector<int> &perm, int shapeSize) {
    if (shapeSize == 2) {
        return Transpose(self, {perm[0], perm[1]});
    }

    auto optimization = analyzeMergeAxisOptimization(self.GetShape(), perm);
    if (optimization.canOptimize) {
        return applyMergeAxisTranspose(self, perm, optimization);
    }

    std::vector<int> invPerm(shapeSize);
    for (int i = 0; i < shapeSize; ++i) {
        invPerm[perm[i]] = i;
    }

    int greedyTotalCount = 0;
    int greedyTailCount = 0;
    {
        std::vector<int> tempPerm = invPerm;
        const int tailDim1 = shapeSize - 2;
        const int tailDim2 = shapeSize - 1;
        for (int i = 0; i < shapeSize; ++i) {
            int targetPos = findTargetPosition(tempPerm, i, i);
            if (targetPos != i && targetPos != -1) {
                greedyTotalCount++;
                bool isTailSwap = (i == tailDim1 && targetPos == tailDim2) ||
                                  (i == tailDim2 && targetPos == tailDim1);
                if (isTailSwap) {
                    greedyTailCount++;
                }
                std::swap(tempPerm[i], tempPerm[targetPos]);
            }
        }
    }

    auto optimalSeq = getOptimalSwapSequence(invPerm, shapeSize);
    int optimalTotalCount = optimalSeq.totalSwapCount;
    int optimalTailCount = optimalSeq.tailSwapCount;

    int greedyNonTailCount = greedyTotalCount - greedyTailCount;
    int optimalNonTailCount = optimalTotalCount - optimalTailCount;

    const int MOVE_OUT_COST = 2;
    int greedyCost = greedyTotalCount + greedyNonTailCount * MOVE_OUT_COST;
    int optimalCost = optimalTotalCount + optimalNonTailCount * MOVE_OUT_COST;

    if (optimalCost < greedyCost) {
        return applyOptimalTransposeSequence(self, invPerm, shapeSize);
    }

    Tensor result = self;
    result = permuteAnyDims(result, invPerm);
    return result;
}

Tensor permuteAnyDims(Tensor inputTensor, std::vector<int> invPerm) {
    const int shapeSize = invPerm.size();
    for (int i = 0; i < shapeSize; ++i) {
        const int targetPos = findTargetPosition(invPerm, i, i);
        if (targetPos != i && targetPos != -1) {
            inputTensor = Transpose(inputTensor, {i, targetPos});
            std::swap(invPerm[i], invPerm[targetPos]);
        }
    }
    return inputTensor;
}

int calculateMinTransposeCount(const std::vector<int> &perm) {
    const int n = perm.size();
    std::vector<bool> visited(n, false);
    int cycles = 0;
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = perm[j];
            }
            cycles++;
        }
    }
    return n - cycles;
}

struct SwapSequence {
    std::vector<std::pair<int, int>> swaps;
    int tailSwapCount = 0;
    int totalSwapCount = 0;
};

static SwapSequence getOptimalSwapSequence4D(const std::vector<int> &perm) {
    SwapSequence result;
    std::vector<int> p = perm;
    std::vector<bool> fixed(p.size(), false);
    const int n = 4;
    const int tailDim1 = n - 2;
    const int tailDim2 = n - 1;
    for (int i = 0; i < n; ++i) {
        if (fixed[i] || p[i] == i) {
            fixed[i] = true;
            continue;
        }
        int j = i;
        std::vector<int> cycle;
        while (!fixed[j]) {
            cycle.push_back(j);
            j = p[j];
        }
        for (size_t k = 0; k + 1 < cycle.size(); ++k) {
            int dim1 = cycle[k];
            int dim2 = cycle[k + 1];
            bool isTailSwap = (dim1 == tailDim1 && dim2 == tailDim2) ||
                              (dim1 == tailDim2 && dim2 == tailDim1);
            result.swaps.push_back({dim1, dim2});
            if (isTailSwap) {
                result.tailSwapCount++;
            }
            fixed[dim1] = true;
        }
        fixed[cycle.back()] = true;
    }
    result.totalSwapCount = result.swaps.size();
    return result;
}

static SwapSequence getOptimalSwapSequence5D(const std::vector<int> &perm) {
    SwapSequence result;
    std::vector<int> p = perm;
    std::vector<bool> fixed(p.size(), false);
    const int n = 5;
    const int tailDim1 = n - 2;
    const int tailDim2 = n - 1;
    for (int i = 0; i < n; ++i) {
        if (fixed[i] || p[i] == i) {
            fixed[i] = true;
            continue;
        }
        int j = i;
        std::vector<int> cycle;
        while (!fixed[j]) {
            cycle.push_back(j);
            j = p[j];
        }
        for (size_t k = 0; k + 1 < cycle.size(); ++k) {
            int dim1 = cycle[k];
            int dim2 = cycle[k + 1];
            bool isTailSwap = (dim1 == tailDim1 && dim2 == tailDim2) ||
                              (dim1 == tailDim2 && dim2 == tailDim1);
            result.swaps.push_back({dim1, dim2});
            if (isTailSwap) {
                result.tailSwapCount++;
            }
            fixed[dim1] = true;
        }
        fixed[cycle.back()] = true;
    }
    result.totalSwapCount = result.swaps.size();
    return result;
}

static SwapSequence getOptimalSwapSequence(const std::vector<int> &perm, int shapeSize) {
    if (shapeSize == 4) {
        return getOptimalSwapSequence4D(perm);
    }
    return getOptimalSwapSequence5D(perm);
}

Tensor applyOptimalTransposeSequence(const Tensor &self, const std::vector<int> &invPerm, int shapeSize) {
    auto seq = getOptimalSwapSequence(invPerm, shapeSize);
    Tensor result = self;
    for (const auto &swap : seq.swaps) {
        result = Transpose(result, {swap.first, swap.second});
    }
    return result;
}

} // namespace npu::tile_fwk