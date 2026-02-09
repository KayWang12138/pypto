#ifndef PYPTO_IR_TRANSFORMS_STRUCTURAL_COMPARISON_H_
#define PYPTO_IR_TRANSFORMS_STRUCTURAL_COMPARISON_H_

#include <cstdint>

#include "ir/core.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Compute structural hash of an IR node
 *
 * Computes hash based on IR node tree structure, ignoring Span (source location).
 * Two IR nodes with identical structure will hash to the same value.
 *
 * This can be used with structural_equal as custom hasher/comparator for unordered containers:
 * @code
 * std::unordered_map<IRNodePtr, int,
 *                    decltype(&structural_hash),
 *                    decltype(&structural_equal)> my_map(
 *   16, structural_hash, structural_equal
 * );
 * @endcode
 *
 * @param node IR node to hash
 * @param enable_auto_mapping If true, ignore variable names (e.g., x+1 and y+1 hash the same).
 *                            If false, variable names matter (default).
 * @return Structural hash value
 */
uint64_t structural_hash(const IRNodePtr& node, bool enable_auto_mapping = false);

/**
 * @brief Compute structural hash of a type
 *
 * @param type Type to hash
 * @param enable_auto_mapping If true, ignore variable names (e.g., x+1 and y+1 hash the same).
 *                            If false, variable names matter (default).
 * @return Structural hash value
 */
uint64_t structural_hash(const TypePtr& type, bool enable_auto_mapping = false);

/**
 * @brief Check if two IR nodes are structurally equal
 *
 * Compares IR node tree structure, ignoring Span (source location).
 * Two IR nodes with identical structure are considered equal.
 *
 * @param lhs First IR node
 * @param rhs Second IR node
 * @param enable_auto_mapping If true, automatically map variables (e.g., x+1 equals y+1).
 *                            If false, variable names must match exactly (default).
 * @return true if structurally equal, false otherwise
 */
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping = false);

/**
 * @brief Check if two types are structurally equal
 *
 * @param lhs First type
 * @param rhs Second type
 * @param enable_auto_mapping If true, automatically map variables (e.g., x+1 equals y+1).
 *                            If false, variable names must match exactly (default).
 * @return true if structurally equal, false otherwise
 */
bool structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping = false);

/**
 * @brief Assert two IR nodes are structurally equal
 *
 * Like structural_equal but throws ValueError with detailed error message
 * showing the first mismatch location and Python-printed IR context.
 * Useful for debugging and testing.
 *
 * @param lhs First IR node
 * @param rhs Second IR node
 * @param enable_auto_mapping If true, automatically map variables (e.g., x+1 equals y+1).
 *                            If false, variable names must match exactly (default).
 * @throws ValueError if nodes are not structurally equal, with detailed diagnostic message
 */
void assert_structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping = false);

/**
 * @brief Assert two types are structurally equal
 *
 * Like structural_equal but throws ValueError with detailed error message.
 *
 * @param lhs First type
 * @param rhs Second type
 * @param enable_auto_mapping If true, automatically map variables (e.g., x+1 equals y+1).
 *                            If false, variable names must match exactly (default).
 * @throws ValueError if types are not structurally equal, with detailed diagnostic message
 */
void assert_structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping = false);
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_STRUCTURAL_COMPARISON_H_
