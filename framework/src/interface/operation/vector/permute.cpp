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
    
    Tensor result = self;
    std::vector<int> invPerm(shapeSize);
    for (int i = 0; i < shapeSize; ++i) {
        invPerm[perm[i]] = i;
    }
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

} // namespace npu::tile_fwk