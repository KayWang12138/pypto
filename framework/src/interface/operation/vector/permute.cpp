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

/**
 * @brief Permute operation implementation for tensors
 * @param self Input tensor
 * @param perm Permutation of dimensions
 * @return Permuted tensor
 */
Tensor Permute(const Tensor &self, std::vector<int> perm) {
    DECLARE_TRACER();
    const int shapeSize = static_cast<int>(self.GetShape().size());

    // Validate permutation size
    ASSERT(perm.size() == static_cast<size_t>(shapeSize)) << "Permute dim num should match input dim num.";

    // Validate permutation values and convert negative indices
    validateAndNormalizePermutation(perm, shapeSize);

    // Check if permutation is identity [0, 1, 2, ..., dim-1]
    if (isIdentityPermutation(perm)) {
        return self;
    }

    // Handle different dimensions
    if (shapeSize <= 5) {
        return permuteTensor(self, perm, shapeSize);
    }

    // For dimensions > 5, not supported yet
    ASSERT(false) << "Permute only supports up to 5 dimensions.";
    return self;
}

/**
 * @brief Validate permutation and normalize negative indices
 * @param perm Permutation vector
 * @param shapeSize Number of dimensions
 */
void validateAndNormalizePermutation(std::vector<int> &perm, int shapeSize) {
    std::vector<bool> used(shapeSize, false);

    for (int &p : perm) {
        // Convert negative indices to positive
        if (p < 0) {
            p += shapeSize;
        }

        // Validate index range
        ASSERT(p >= 0 && p < shapeSize) << "Permute dim is invalid.";

        // Validate no duplicate indices
        ASSERT(!used[p]) << "Permute dims contain duplicate values.";
        used[p] = true;
    }
}

/**
 * @brief Check if permutation is identity [0, 1, 2, ..., dim-1]
 * @param perm Permutation vector
 * @return true if permutation is identity, false otherwise
 */
bool isIdentityPermutation(const std::vector<int> &perm) {
    // For 1D, always identity
    if (perm.size() == 1) {
        return true;
    }

    // For permutations with size > 1, check if sorted
    return std::is_sorted(perm.begin(), perm.end());
}

/**
 * @brief Find the position of the element that should be at targetIndex
 * @param invPerm Inverse permutation
 * @param targetIndex Target index
 * @param startSearch Start search from this index
 * @return Position of the element, or -1 if not found
 */
int findTargetPosition(const std::vector<int> &invPerm, int targetIndex, int startSearch) {
    for (int j = startSearch; j < static_cast<int>(invPerm.size()); ++j) {
        if (invPerm[j] == targetIndex) {
            return j;
        }
    }
    return -1;
}

/**
 * @brief Permute tensor based on dimension size
 * @param self Input tensor
 * @param perm Permutation vector
 * @param shapeSize Number of dimensions
 * @return Permuted tensor
 */
Tensor permuteTensor(const Tensor &self, const std::vector<int> &perm, int shapeSize) {
    // Handle 2D case
    if (shapeSize == 2) {
        return Transpose(self, {perm[0], perm[1]});
    }

    // Handle 3D, 4D, 5D cases
    Tensor result = self;

    // Build inverse permutation: invPerm[i] = position where element at i should go
    std::vector<int> invPerm(shapeSize);
    for (int i = 0; i < shapeSize; ++i) {
        invPerm[perm[i]] = i;
    }

    // 3D, 4D, 5D: support any axis transpose, use selection sort
    result = permuteAnyDims(result, invPerm);

    return result;
}

/**
 * @brief Permute tensor with any number of dimensions using any axis transposes
 * @param tensor Input tensor
 * @param invPerm Inverse permutation
 * @return Permuted tensor
 */
Tensor permuteAnyDims(Tensor tensor, std::vector<int> invPerm) {
    const int shapeSize = invPerm.size();

    // Use selection sort for any dims (any axis transposes)
    for (int i = 0; i < shapeSize; ++i) {
        // Find the position of the element that should be at position i
        const int targetPos = findTargetPosition(invPerm, i, i);
        if (targetPos != i && targetPos != -1) {
            // Transpose directly to target position
            tensor = Transpose(tensor, {i, targetPos});
            // Update inverse permutation
            std::swap(invPerm[i], invPerm[targetPos]);
        }
    }

    return tensor;
}
} // namespace npu::tile_fwk