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


Tensor Permute(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();
    const int shapeSize = static_cast<int>(self.GetShape().size());
    ASSERT(perm.size() == static_cast<size_t>(shapeSize)) << "Permute dim num should match input dim num.";
    // Validate permutation values and convert negative indices
    validateAndNormalizePermutation(perm, shapeSize);
    // Check if permutation is identity [0, 1, 2, ..., dim-1]
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
        // Find the position of the element that should be at position i
        const int targetPos = findTargetPosition(invPerm, i, i);
        if (targetPos != i && targetPos != -1) {
            // Transpose directly to target position
            inputTensor = Transpose(inputTensor, {i, targetPos});
            // Update inverse permutation
            std::swap(invPerm[i], invPerm[targetPos]);
        }
    }
    return inputTensor;
}
} // namespace npu::tile_fwk