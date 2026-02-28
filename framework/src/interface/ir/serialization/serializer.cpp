/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/serialization/serializer.h"

#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

/**
 * @brief Field visitor for serialization
 *
 * Visits all fields of an IR node and serializes them to MessagePack format.
 */
class FieldSerializerVisitor {
public:
    using resultType = msgpack::object;

    explicit FieldSerializerVisitor(msgpack::zone& zone, class IRSerializer::Impl& ctx) : zone_(zone), ctx_(ctx) {}

    [[nodiscard]] resultType InitResult() const;

    // Visit IRNode pointer fields
    template <typename IRNodePtrType>
    resultType VisitIRNodeField(const IRNodePtrType& field);

    // Visit optional IRNode pointer fields
    template <typename IRNodePtrType>
    resultType VisitIRNodeField(const std::optional<IRNodePtrType>& field);

    // Visit vector of IRNode pointers
    template <typename IRNodePtrType>
    resultType VisitIRNodeVectorField(const std::vector<IRNodePtrType>& field);

    // Visit map of IRNode pointers
    template <typename KeyType, typename ValueType, typename Compare>
    resultType VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& field);

    // Visit leaf fields
    resultType VisitLeafField(const int& field);
    resultType VisitLeafField(const int64_t& field);
    resultType VisitLeafField(const uint64_t& field);
    resultType VisitLeafField(const double& field);
    resultType VisitLeafField(const bool& field);
    resultType VisitLeafField(const std::string& field);
    resultType VisitLeafField(const DataType& field);
    resultType VisitLeafField(const FunctionType& field);
    resultType VisitLeafField(const MemorySpace& field);
    resultType VisitLeafField(const TypePtr& field);
    resultType VisitLeafField(const OpPtr& field);
    resultType VisitLeafField(const Span& field);
    resultType VisitLeafField(const std::vector<TypePtr>& field);
    resultType VisitLeafField(const std::vector<std::pair<std::string, std::any>>& field);

    // Field kind hooks
    template <typename FVisitOp>
    void VisitIgnoreField(FVisitOp&& visitOp) {
        visitOp();
    }

    template <typename FVisitOp>
    void VisitDefField(FVisitOp&& visitOp) {
        visitOp();
    }

    template <typename FVisitOp>
    void VisitUsualField(FVisitOp&& visitOp) {
        visitOp();
    }

    // Combine field results into a map
    template <typename Desc>
    void CombineResult(resultType& acc, resultType fieldResult, const Desc& desc);

private:
    msgpack::zone& zone_;
    class IRSerializer::Impl& ctx_;
    std::map<std::string, msgpack::object> fields_;
};

/**
 * @brief Implementation class for IRSerializer
 */
class IRSerializer::Impl {
public:
    Impl() = default;

    std::vector<uint8_t> Serialize(const IRNodePtr& node) {
        ptrToId_.clear();
        nextId_ = 0;

        msgpack::sbuffer buffer;
        msgpack::packer<msgpack::sbuffer> packer(buffer);

        msgpack::zone zone;
        auto obj = SerializeNode(node, zone);
        packer.pack(obj);

        return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
    }

    msgpack::object SerializeNode(const IRNodePtr& node, msgpack::zone& zone) {
        INTERNAL_CHECK(node) << "Cannot serialize null IR node";

        // Check if we've already serialized this pointer
        auto it = ptrToId_.find(node.get());
        if (it != ptrToId_.end()) {
            // Return a reference to the already-serialized node
            std::map<std::string, msgpack::object> refMap;
            refMap["ref"] = msgpack::object(it->second, zone);
            return msgpack::object(refMap, zone);
        }

        // Assign a new ID to this node
        uint64_t id = nextId_++;
        ptrToId_[node.get()] = id;

        // Serialize the node with its ID and type
        std::map<std::string, msgpack::object> nodeMap;
        nodeMap["id"] = msgpack::object(id, zone);
        nodeMap["type"] = msgpack::object(node->TypeName(), zone);

        // Serialize fields using field visitor
        nodeMap["fields"] = SerializeFields(node, zone);

        return msgpack::object(nodeMap, zone);
    }

    msgpack::object SerializeFields(const IRNodePtr& node, msgpack::zone& zone) {
#define SERIALIZE_FIELDS(Type)                  \
    if (auto p = As<Type>(node)) {              \
        return SerializeFieldsGeneric(p, zone); \
    }

#define SERIALIZE_FIELDS_BASE(Type)             \
    if (auto p = As<Type>(node)) {              \
        return SerializeFieldsGeneric(p, zone); \
    }

        SERIALIZE_FIELDS(IterArg);
        SERIALIZE_FIELDS(Var);
        SERIALIZE_FIELDS(MemRef);
        SERIALIZE_FIELDS(ConstInt);
        SERIALIZE_FIELDS(ConstFloat);
        SERIALIZE_FIELDS(ConstBool);
        SERIALIZE_FIELDS(Call);
        SERIALIZE_FIELDS(MakeTuple);
        SERIALIZE_FIELDS(TupleGetItemExpr);

        // BinaryExpr and UnaryExpr are abstract base classes, use dynamic_pointer_cast
        SERIALIZE_FIELDS_BASE(BinaryExpr);
        SERIALIZE_FIELDS_BASE(UnaryExpr);

        SERIALIZE_FIELDS(AssignStmt);
        SERIALIZE_FIELDS(IfStmt);
        SERIALIZE_FIELDS(YieldStmt);
        SERIALIZE_FIELDS(ReturnStmt);
        SERIALIZE_FIELDS(ForStmt);
        SERIALIZE_FIELDS(SeqStmts);
        SERIALIZE_FIELDS(OpStmts);
        SERIALIZE_FIELDS(EvalStmt);
        SERIALIZE_FIELDS(Function);
        SERIALIZE_FIELDS(Program);

#undef SERIALIZE_FIELDS
#undef SERIALIZE_FIELDS_BASE

        INTERNAL_UNREACHABLE << "Unknown IR node type in serialization: " << node->TypeName();
        return msgpack::object(); // Unreachable, but needed for compilation
    }

    msgpack::object SerializeSpan(const Span& span, msgpack::zone& zone) {
        std::map<std::string, msgpack::object> spanMap;
        spanMap["filename"] = msgpack::object(span.filename_, zone);
        spanMap["begin_line"] = msgpack::object(span.beginLine_, zone);
        spanMap["begin_column"] = msgpack::object(span.beginColumn_, zone);
        spanMap["end_line"] = msgpack::object(span.endLine_, zone);
        spanMap["end_column"] = msgpack::object(span.endColumn_, zone);
        return msgpack::object(spanMap, zone);
    }

    msgpack::object SerializeMemRef(const std::optional<MemRefPtr>& memrefOpt, msgpack::zone& zone) {
        if (!memrefOpt.has_value()) {
            return msgpack::object(); // null
        }

        const auto& memref = *memrefOpt.value();
        std::map<std::string, msgpack::object> memrefMap;
        memrefMap["memory_space"] = msgpack::object(static_cast<uint8_t>(memref.memorySpace_), zone);
        memrefMap["addr"] = SerializeNode(memref.addr_, zone);
        memrefMap["size"] = msgpack::object(memref.size_, zone);
        memrefMap["id"] = msgpack::object(memref.id_, zone);
        return msgpack::object(memrefMap, zone);
    }

    msgpack::object SerializeTileView(const std::optional<TileView>& tileView, msgpack::zone& zone) {
        if (!tileView.has_value()) {
            return msgpack::object(); // null
        }

        std::map<std::string, msgpack::object> tvMap;

        // Serialize valid_shape
        std::vector<msgpack::object> validShapeVec;
        for (const auto& dim : tileView->validShape) {
            validShapeVec.push_back(SerializeNode(dim, zone));
        }
        tvMap["valid_shape"] = msgpack::object(validShapeVec, zone);

        // Serialize stride
        std::vector<msgpack::object> strideVec;
        for (const auto& dim : tileView->stride) {
            strideVec.push_back(SerializeNode(dim, zone));
        }
        tvMap["stride"] = msgpack::object(strideVec, zone);

        // Serialize start_offset
        tvMap["start_offset"] = SerializeNode(tileView->startOffset, zone);

        return msgpack::object(tvMap, zone);
    }

    msgpack::object SerializeType(const TypePtr& type, msgpack::zone& zone) {
        INTERNAL_CHECK(type) << "Cannot serialize null Type";

        std::map<std::string, msgpack::object> typeMap;
        typeMap["type_kind"] = msgpack::object(type->TypeName(), zone);

        if (auto scalarType = As<ScalarType>(type)) {
            typeMap["dtype"] = msgpack::object(scalarType->dtype_.Code(), zone);
        } else if (auto tensorType = As<TensorType>(type)) {
            typeMap["dtype"] = msgpack::object(tensorType->dtype_.Code(), zone);

            std::vector<msgpack::object> shapeVec;
            for (const auto& dim : tensorType->shape_) {
                shapeVec.push_back(SerializeNode(dim, zone));
            }
            typeMap["shape"] = msgpack::object(shapeVec, zone);

            // Serialize memref if present
            if (tensorType->memref_.has_value()) {
                typeMap["memref"] = SerializeMemRef(tensorType->memref_, zone);
            }
        } else if (auto tileType = As<TileType>(type)) {
            typeMap["dtype"] = msgpack::object(tileType->dtype_.Code(), zone);

            std::vector<msgpack::object> shapeVec;
            for (const auto& dim : tileType->shape_) {
                shapeVec.push_back(SerializeNode(dim, zone));
            }
            typeMap["shape"] = msgpack::object(shapeVec, zone);

            // Serialize memref if present
            if (tileType->memref_.has_value()) {
                typeMap["memref"] = SerializeMemRef(tileType->memref_, zone);
            }

            // Serialize tile_view if present
            if (tileType->tileView_.has_value()) {
                typeMap["tile_view"] = SerializeTileView(tileType->tileView_, zone);
            }
        } else if (auto tupleType = As<TupleType>(type)) {
            std::vector<msgpack::object> typesVec;
            for (const auto& t : tupleType->types_) {
                typesVec.push_back(SerializeType(t, zone));
            }
            typeMap["types"] = msgpack::object(typesVec, zone);
        } else if (IsA<MemRefType>(type)) {
            // MemRefType has no additional fields
        } else if (IsA<UnknownType>(type)) {
            // UnknownType has no additional fields
        } else {
            INTERNAL_UNREACHABLE << "Unknown Type subclass: " << type->TypeName();
        }

        return msgpack::object(typeMap, zone);
    }

    msgpack::object SerializeDataType(const DataType& dtype, msgpack::zone& zone) {
        std::map<std::string, msgpack::object> dtypeMap;
        dtypeMap["type"] = msgpack::object("DataType", zone);
        dtypeMap["code"] = msgpack::object(dtype.Code(), zone);
        return msgpack::object(dtypeMap, zone);
    }

    msgpack::object SerializeOp(const OpPtr& op, msgpack::zone& zone) {
        INTERNAL_CHECK(op) << "Cannot serialize null Op";

        std::map<std::string, msgpack::object> opMap;
        opMap["name"] = msgpack::object(op->name_, zone);

        // Check if it's a GlobalVar
        if (IsA<GlobalVar>(op)) {
            opMap["is_global_var"] = msgpack::object(true, zone);
        } else {
            opMap["is_global_var"] = msgpack::object(false, zone);
        }

        return msgpack::object(opMap, zone);
    }

    template <typename NodePtr>
    msgpack::object SerializeFieldsGeneric(const NodePtr& node, msgpack::zone& zone) {
        using NodeType = typename NodePtr::element_type;
        auto descriptors = NodeType::GetFieldDescriptors();

        FieldSerializerVisitor visitor(zone, *this);
        return std::apply(
            [&](auto&&... descs) {
                return reflection::FieldIterator<NodeType, FieldSerializerVisitor, decltype(descs)...>::Visit(
                    *node, visitor, descs...);
            },
            descriptors);
    }

private:
    uint64_t nextId_;
    std::unordered_map<const IRNode*, uint64_t> ptrToId_;
};

// FieldSerializerVisitor implementation

msgpack::object FieldSerializerVisitor::InitResult() const {
    return msgpack::object(fields_, zone_);
}

template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeField(const IRNodePtrType& field) {
    return ctx_.SerializeNode(field, zone_);
}

// Overload for std::optional<IRNodePtr>
template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeField(const std::optional<IRNodePtrType>& field) {
    if (field.has_value() && *field) {
        return ctx_.SerializeNode(*field, zone_);
    } else {
        // Return null object for empty optional
        return msgpack::object();
    }
}

template <typename IRNodePtrType>
msgpack::object FieldSerializerVisitor::VisitIRNodeVectorField(const std::vector<IRNodePtrType>& field) {
    std::vector<msgpack::object> vec;
    for (const auto& item : field) {
        vec.push_back(ctx_.SerializeNode(item, zone_));
    }
    return msgpack::object(vec, zone_);
}

template <typename KeyType, typename ValueType, typename Compare>
msgpack::object FieldSerializerVisitor::VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& field) {
    // Serialize map as array of {key, value} pairs
    std::vector<msgpack::object> entries;
    for (const auto& [key, value] : field) {
        std::map<std::string, msgpack::object> entry;
        entry["key"] = ctx_.SerializeOp(key, zone_);
        entry["value"] = ctx_.SerializeNode(value, zone_);
        entries.emplace_back(entry, zone_);
    }
    return msgpack::object(entries, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const int& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const int64_t& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const uint64_t& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const double& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const bool& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const std::string& field) {
    return msgpack::object(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const DataType& field) {
    return ctx_.SerializeDataType(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const FunctionType& field) {
    return msgpack::object(static_cast<uint8_t>(field), zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const MemorySpace& field) {
    return msgpack::object(static_cast<uint8_t>(field), zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const TypePtr& field) {
    return ctx_.SerializeType(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const OpPtr& field) {
    return ctx_.SerializeOp(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const Span& field) {
    return ctx_.SerializeSpan(field, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const std::vector<TypePtr>& field) {
    std::vector<msgpack::object> vec;
    vec.reserve(field.size());
    for (const auto& type : field) {
        vec.push_back(ctx_.SerializeType(type, zone_));
    }
    return msgpack::object(vec, zone_);
}

msgpack::object FieldSerializerVisitor::VisitLeafField(const std::vector<std::pair<std::string, std::any>>& kwargs) {
    // Use vector to preserve order (msgpack will serialize as array of [key, value] pairs)
    std::vector<msgpack::object> kwargsMsgs;

    auto make_pair = [this](const std::string& key, const msgpack::object& value) -> msgpack::object {
        std::map<std::string, msgpack::object> pairMap;
        pairMap["key"] = msgpack::object(key, zone_);
        pairMap["value"] = value;
        return msgpack::object(pairMap, zone_);
    };

    for (const auto& [key, value] : kwargs) {
        // Serialize common types
        if (value.type() == typeid(int)) {
            kwargsMsgs.push_back(make_pair(key, VisitLeafField(AnyCast<int>(value, "serializing kwarg: " + key))));
        } else if (value.type() == typeid(bool)) {
            kwargsMsgs.push_back(make_pair(key, VisitLeafField(AnyCast<bool>(value, "serializing kwarg: " + key))));
        } else if (value.type() == typeid(std::string)) {
            kwargsMsgs.push_back(
                make_pair(key, VisitLeafField(AnyCast<std::string>(value, "serializing kwarg: " + key))));
        } else if (value.type() == typeid(double)) {
            kwargsMsgs.push_back(make_pair(key, VisitLeafField(AnyCast<double>(value, "serializing kwarg: " + key))));
        } else if (value.type() == typeid(float)) {
            kwargsMsgs.push_back(make_pair(key, VisitLeafField(AnyCast<float>(value, "serializing kwarg: " + key))));
        } else if (value.type() == typeid(DataType)) {
            kwargsMsgs.push_back(make_pair(key, VisitLeafField(AnyCast<DataType>(value, "serializing kwarg: " + key))));
        } else {
            throw TypeError("Invalid kwarg type for key: " + key +
                            ", expected int, bool, std::string, double, float, or DataType, but got " +
                            DemangleTypeName(value.type().name()));
        }
    }

    return msgpack::object(kwargsMsgs, zone_);
}

template <typename Desc>
void FieldSerializerVisitor::CombineResult(resultType& acc, resultType fieldResult, const Desc& desc) {
    fields_[desc.name] = fieldResult;
    acc = msgpack::object(fields_, zone_);
}

// IRSerializer implementation

IRSerializer::IRSerializer() : impl_(std::make_unique<Impl>()) {}

IRSerializer::~IRSerializer() = default;

std::vector<uint8_t> IRSerializer::Serialize(const IRNodePtr& node) {
    return impl_->Serialize(node);
}

// Public API functions

std::vector<uint8_t> Serialize(const IRNodePtr& node) {
    IRSerializer serializer;
    return serializer.Serialize(node);
}

void SerializeToFile(const IRNodePtr& node, const std::string& path) {
    auto data = Serialize(node);
    std::ofstream file(path, std::ios::binary);
    INTERNAL_CHECK(file) << "Failed to open file for writing: " + path;
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    INTERNAL_CHECK(file) << "Failed to write to file: " + path;
}

} // namespace serialization
} // namespace ir
} // namespace pypto
