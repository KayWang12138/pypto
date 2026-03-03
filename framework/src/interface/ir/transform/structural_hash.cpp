/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/logging.h"
#include "ir/core.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

/**
 * \brief Hash combine using Boost-inspired algorithm
 */
inline uint64_t hash_combine(uint64_t seed, uint64_t value) {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * \brief Structural hasher for IR nodes
 *
 * Computes hash based on IR node tree structure, ignoring Span (source location).
 * Also serves as a FieldVisitor for the reflection-based field iteration.
 */
class StructuralHasher {
public:
    using resultType = uint64_t;

    explicit StructuralHasher(bool enableAutoMapping) : enableAutoMapping_(enableAutoMapping) {}

    resultType operator()(const IRNodePtr &node) { return HashNode(node); }

    resultType operator()(const TypePtr &type) { return HashType(type); }

    // FieldVisitor interface methods
    [[nodiscard]] resultType InitResult() const { return 0; }

    template <typename IRNodePtrType>
    resultType VisitIRNodeField(const IRNodePtrType &field) {
        INTERNAL_CHECK(field) << "structural_hash encountered null IR node field";
        return HashNode(field);
    }

    // Specialization for std::optional<IRNodePtr>
    template <typename IRNodePtrType>
    resultType VisitIRNodeField(const std::optional<IRNodePtrType> &field) {
        if (field.has_value() && *field) {
            return HashNode(*field);
        } else {
            // Hash empty optional as 0
            return 0;
        }
    }

    template <typename IRNodePtrType>
    resultType VisitIRNodeVectorField(const std::vector<IRNodePtrType> &fields) {
        resultType h = 0;
        for (size_t i = 0; i < fields.size(); ++i) {
            INTERNAL_CHECK(fields[i]) << "structural_hash encountered null IR node in vector at index " << i;
            h = hash_combine(h, HashNode(fields[i]));
        }
        return h;
    }

    template <typename KeyType, typename ValueType, typename Compare>
    resultType VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare> &field) {
        resultType h = 0;
        for (const auto &[key, value] : field) {
            INTERNAL_CHECK(key) << "structural_hash encountered null key in map";
            INTERNAL_CHECK(value) << "structural_hash encountered null value in map";
            // Hash key by name (keys are Op types, not IRNode)
            h = hash_combine(h, static_cast<resultType>(std::hash<std::string>{}(key->name_)));
            // Hash value (values are IRNode types)
            h = hash_combine(h, HashNode(value));
        }
        return h;
    }

    template <typename FVisitOp>
    void VisitIgnoreField([[maybe_unused]] FVisitOp &&visitOp) {
        // Ignore field, do nothing
    }
    template <typename FVisitOp>
    void VisitDefField(FVisitOp &&visitOp) {
        bool enableAutoMapping = true;
        std::swap(enableAutoMapping, enableAutoMapping_);
        visitOp();
        std::swap(enableAutoMapping, enableAutoMapping_);
    }
    template <typename FVisitOp>
    void VisitUsualField(FVisitOp &&visitOp) {
        visitOp();
    }

    resultType VisitLeafField(const int &field) { return static_cast<resultType>(std::hash<int>{}(field)); }

    resultType VisitLeafField(const int64_t &field) { return static_cast<resultType>(std::hash<int64_t>{}(field)); }

    resultType VisitLeafField(const uint64_t &field) { return static_cast<resultType>(std::hash<uint64_t>{}(field)); }

    resultType VisitLeafField(const double &field) { return static_cast<resultType>(std::hash<double>{}(field)); }

    resultType VisitLeafField(const std::string &field) {
        return static_cast<resultType>(std::hash<std::string>{}(field));
    }

    resultType VisitLeafField(const OpPtr &field) {
        return static_cast<resultType>(std::hash<std::string>{}(field->name_));
    }

    resultType VisitLeafField(const DataType &field) {
        return static_cast<resultType>(std::hash<uint8_t>{}(field.Code()));
    }

    resultType VisitLeafField(const FunctionType &field) {
        return static_cast<resultType>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
    }

    resultType VisitLeafField(const MemorySpace &field) {
        return static_cast<resultType>(std::hash<int>{}(static_cast<int>(field)));
    }

    resultType VisitLeafField(const TypePtr &field) {
        INTERNAL_CHECK(field) << "structural_hash encountered null TypePtr field";
        return HashType(field);
    }

    resultType VisitLeafField(const std::vector<TypePtr> &fields) {
        resultType h = 0;
        for (size_t i = 0; i < fields.size(); ++i) {
            INTERNAL_CHECK(fields[i]) << "structural_hash encountered null TypePtr in vector at index " << i;
            h = hash_combine(h, HashType(fields[i]));
        }
        return h;
    }

    // Hash kwargs (vector of pairs - order is preserved and matters)
    resultType VisitLeafField(const std::vector<std::pair<std::string, std::any>> &kwargs) {
        resultType h = 0;
        // Hash keys and values in order (no need to sort since order is preserved)
        for (const auto &[key, value] : kwargs) {
            h = hash_combine(h, std::hash<std::string>{}(key));

            // Hash value based on type
            if (value.type() == typeid(int)) {
                h = hash_combine(h, std::hash<int>{}(AnyCast<int>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(bool)) {
                h = hash_combine(h, std::hash<bool>{}(AnyCast<bool>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(std::string)) {
                h = hash_combine(h, std::hash<std::string>{}(AnyCast<std::string>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(double)) {
                h = hash_combine(h, std::hash<double>{}(AnyCast<double>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(float)) {
                h = hash_combine(h, std::hash<float>{}(AnyCast<float>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(DataType)) {
                h = hash_combine(h, std::hash<uint8_t>{}(AnyCast<DataType>(value, "hashing kwarg: " + key).Code()));
            } else {
                throw TypeError("Invalid kwarg type for key: " + key +
                                ", expected int, bool, std::string, double, float, or DataType, but got " +
                                DemangleTypeName(value.type().name()));
            }
        }
        return h;
    }

    resultType VisitLeafField(const Span & /*field*/) {
        INTERNAL_UNREACHABLE << "structural_hash should not visit Span field";
        return 0; // Never reached, but suppresses -Werror=return-type
    }

    template <typename Desc>
    void CombineResult(resultType &accumulator, resultType fieldHash, const Desc & /*descriptor*/) {
        accumulator = hash_combine(accumulator, fieldHash);
    }

private:
    resultType HashNode(const IRNodePtr &node);
    resultType HashVar(const VarPtr &op);
    resultType HashType(const TypePtr &type);

    template <typename NodePtr>
    resultType HashNodeImpl(const NodePtr &node);

    bool enableAutoMapping_;
    std::unordered_map<IRNodePtr, resultType> hashValueMap_;
    int64_t freeVarCounter_ = 0;
};

template <typename NodePtr>
StructuralHasher::resultType StructuralHasher::HashNodeImpl(const NodePtr &node) {
    using NodeType = typename NodePtr::element_type;

    // Start with type discriminator
    resultType h = static_cast<resultType>(std::hash<std::string>{}(node->TypeName()));

    // Visit all fields using reflection
    auto descriptors = NodeType::GetFieldDescriptors();

    resultType fieldsHash = std::apply(
        [&](auto &&...descs) {
            return reflection::FieldIterator<NodeType, StructuralHasher, decltype(descs)...>::Visit(
                *node, *this, descs...);
        },
        descriptors);

    return hash_combine(h, fieldsHash);
}

StructuralHasher::resultType StructuralHasher::HashVar(const VarPtr &op) {
    resultType h = HashNodeImpl(op);
    if (enableAutoMapping_) {
        // Auto-mapping: map Var pointers to sequential IDs for structural comparison
        h = hash_combine(h, freeVarCounter_++);
    } else {
        // Without auto-mapping: hash the VarPtr itself (pointer-based)
        h = hash_combine(h, static_cast<resultType>(std::hash<VarPtr>{}(op)));
    }
    return h;
}

StructuralHasher::resultType StructuralHasher::HashType(const TypePtr &type) {
    INTERNAL_CHECK(type) << "structural_hash encountered null TypePtr";
    resultType h = static_cast<resultType>(std::hash<std::string>{}(type->TypeName()));
    if (auto scalarType = As<ScalarType>(type)) {
        h = hash_combine(h, static_cast<resultType>(std::hash<uint8_t>{}(scalarType->dtype_.Code())));
    } else if (auto tensorType = As<TensorType>(type)) {
        h = hash_combine(h, static_cast<resultType>(std::hash<uint8_t>{}(tensorType->dtype_.Code())));
        h = hash_combine(h, static_cast<resultType>(tensorType->shape_.size()));
        for (const auto &dim : tensorType->shape_) {
            INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TypePtr";
            h = hash_combine(h, HashNode(dim));
        }
    } else if (auto tileType = As<TileType>(type)) {
        // Hash dtype
        h = hash_combine(h, static_cast<resultType>(std::hash<uint8_t>{}(tileType->dtype_.Code())));
        // Hash shape size and dimensions
        h = hash_combine(h, static_cast<resultType>(tileType->shape_.size()));
        for (const auto &dim : tileType->shape_) {
            INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TileType";
            h = hash_combine(h, HashNode(dim));
        }
        // Hash tile_view if present
        if (tileType->tileView_.has_value()) {
            const auto &tv = tileType->tileView_.value();
            h = hash_combine(h, static_cast<resultType>(1)); // indicate presence
            // Hash valid_shape
            h = hash_combine(h, static_cast<resultType>(tv.validShape.size()));
            for (const auto &dim : tv.validShape) {
                INTERNAL_CHECK(dim) << "structural_hash encountered null valid_shape dimension in TileView";
                h = hash_combine(h, HashNode(dim));
            }
            // Hash stride
            h = hash_combine(h, static_cast<resultType>(tv.stride.size()));
            for (const auto &dim : tv.stride) {
                INTERNAL_CHECK(dim) << "structural_hash encountered null stride dimension in TileView";
                h = hash_combine(h, HashNode(dim));
            }
            // Hash start_offset
            INTERNAL_CHECK(tv.startOffset) << "structural_hash encountered null start_offset in TileView";
            h = hash_combine(h, HashNode(tv.startOffset));
        } else {
            h = hash_combine(h, static_cast<resultType>(0)); // indicate absence
        }
    } else if (auto tupleType = As<TupleType>(type)) {
        h = hash_combine(h, static_cast<resultType>(tupleType->types_.size()));
        for (const auto &t : tupleType->types_) {
            INTERNAL_CHECK(t) << "structural_hash encountered null type in TupleType";
            h = hash_combine(h, HashType(t));
        }
    } else if (IsA<UnknownType>(type)) {
        // UnknownType has no fields, so only hash the type name (already done above)
    } else {
        INTERNAL_CHECK(false) << "HashType encountered unhandled Type: " << type->TypeName();
    }
    return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                                                          \
    if (auto p = As<Type>(node)) {                                                                   \
        INTERNAL_CHECK(dispatched == false) << "HashNodeImpl already dispatched for type " << #Type; \
        hashValue = HashNodeImpl(p);                                                                 \
        dispatched = true;                                                                           \
    }

// Dispatch macro for abstract base classes
#define HASH_DISPATCH_BASE(Type)                                                                     \
    if (auto p = As<Type>(node)) {                                                                   \
        INTERNAL_CHECK(dispatched == false) << "HashNodeImpl already dispatched for type " << #Type; \
        hashValue = HashNodeImpl(p);                                                                 \
        dispatched = true;                                                                           \
    }

StructuralHasher::resultType StructuralHasher::HashNode(const IRNodePtr &node) {
    INTERNAL_CHECK(node) << "structural_hash received null IR node";

    auto it = hashValueMap_.find(node);
    if (it != hashValueMap_.end()) {
        return it->second;
    }

    resultType hashValue = 0;
    bool dispatched = false;

    // IterArg needs special handling: dispatch for fields, then add Var mapping
    HASH_DISPATCH(IterArg)
    HASH_DISPATCH(Var)
    HASH_DISPATCH(ConstInt)
    HASH_DISPATCH(ConstFloat)
    HASH_DISPATCH(ConstBool)
    HASH_DISPATCH(Call)
    HASH_DISPATCH(MakeTuple)
    HASH_DISPATCH(TupleGetItemExpr)

    // BinaryExpr and UnaryExpr are abstract base classes, use dynamic_pointer_cast
    HASH_DISPATCH_BASE(BinaryExpr)
    HASH_DISPATCH_BASE(UnaryExpr)

    HASH_DISPATCH(AssignStmt)
    HASH_DISPATCH(IfStmt)
    HASH_DISPATCH(YieldStmt)
    HASH_DISPATCH(ReturnStmt)
    HASH_DISPATCH(ForStmt)
    HASH_DISPATCH(SeqStmts)
    HASH_DISPATCH(OpStmts)
    HASH_DISPATCH(EvalStmt)
    HASH_DISPATCH(Function)
    HASH_DISPATCH(Program)

    // Free Var types (including IterArg) that may be mapped to other free vars
    // Note: IterArg has already been dispatched above for field hashing,
    // here we add the variable-specific hash
    if (auto iterArg = As<IterArg>(node)) {
        if (enableAutoMapping_) {
            hashValue = hash_combine(hashValue, freeVarCounter_++);
        } else {
            // Hash based on pointer for unique instances
            hashValue = hash_combine(hashValue, static_cast<resultType>(std::hash<IterArgPtr>{}(iterArg)));
        }
    } else if (auto var = As<Var>(node)) {
        if (enableAutoMapping_) {
            hashValue = hash_combine(hashValue, freeVarCounter_++);
        } else {
            hashValue = hash_combine(hashValue, static_cast<resultType>(std::hash<VarPtr>{}(var)));
        }
    }

    if (!dispatched) {
        // Unknown IR node type
        throw TypeError("Unknown IR node type in StructuralHasher::HashNode");
    }
    hashValueMap_.emplace(node, hashValue);
    return hashValue;
}

#undef HASH_DISPATCH
#undef HASH_DISPATCH_BASE

// Public API
uint64_t StructuralHash(const IRNodePtr &node, bool enableAutoMapping) {
    StructuralHasher hasher(enableAutoMapping);
    return hasher(node);
}

uint64_t StructuralHash(const TypePtr &type, bool enableAutoMapping) {
    StructuralHasher hasher(enableAutoMapping);
    return hasher(type);
}

} // namespace ir
} // namespace pypto
