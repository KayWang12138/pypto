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

#include "ir/serialization/deserializer.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// clang-format off
#include <msgpack.hpp>
// clang-format on

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/serialization/type_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace serialization {

/**
 * \brief Implementation class for IRDeserializer
 */
class IRDeserializer::Impl : public detail::DeserializerContext {
public:
    Impl() = default;

    IRNodePtr Deserialize(const std::vector<uint8_t> &data) {
        idToPtr_.clear();

        try {
            msgpack::object_handle oh = msgpack::unpack(reinterpret_cast<const char *>(data.data()), data.size());
            msgpack::object obj = oh.get();
            return DeserializeNode(obj, *oh.zone());
        } catch (const msgpack::parse_error &e) {
            throw RuntimeError(std::string("MessagePack parse error: ") + e.what());
        } catch (const msgpack::type_error &e) {
            throw RuntimeError(std::string("MessagePack type error: ") + e.what());
        }
    }

    IRNodePtr DeserializeNode(const msgpack::object &obj, msgpack::zone &zone) override {
        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for IR node";

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;

        // Check if this is a reference
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "ref") {
                uint64_t id;
                p->val.convert(id);
                auto it = idToPtr_.find(id);
                INTERNAL_CHECK(it != idToPtr_.end()) << "Invalid reference ID: " << id;
                return it->second;
            }
        }

        // Parse full node
        uint64_t id = 0;
        std::string typeName;
        msgpack::object fieldsObj;
        bool hasId = false;
        bool hasType = false;
        bool hasFields = false;

        p = obj.via.map.ptr;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "id") {
                p->val.convert(id);
                hasId = true;
            } else if (key == "type") {
                p->val.convert(typeName);
                hasType = true;
            } else if (key == "fields") {
                fieldsObj = p->val;
                hasFields = true;
            }
        }

        INTERNAL_CHECK(hasId && hasType && hasFields) << "Missing required fields (id, type, or fields) in node";

        // Use type registry to create the node
        IRNodePtr node = TypeRegistry::Instance().Create(typeName, fieldsObj, zone, *this);

        // Store in reference table
        idToPtr_[id] = node;

        return node;
    }

    Span DeserializeSpan(const msgpack::object &obj) override {
        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for Span";
        std::string filename;
        int beginLine = -1, beginColumn = -1, endLine = -1, endColumn = -1;

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "filename") {
                p->val.convert(filename);
            } else if (key == "begin_line") {
                p->val.convert(beginLine);
            } else if (key == "begin_column") {
                p->val.convert(beginColumn);
            } else if (key == "end_line") {
                p->val.convert(endLine);
            } else if (key == "end_column") {
                p->val.convert(endColumn);
            }
        }

        return Span(filename, beginLine, beginColumn, endLine, endColumn);
    }

    std::optional<MemRefPtr> DeserializeMemRef(const msgpack::object &obj, msgpack::zone &zone) {
        if (obj.is_nil()) {
            return std::nullopt;
        }

        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for MemRef";

        MemorySpace memorySpace = MemorySpace::DDR;
        ExprPtr addr = nullptr;
        uint64_t size = 0;
        uint64_t id = 0;
        bool hasAddr = false;
        bool hasSize = false;
        bool hasId = false;

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "memory_space") {
                uint8_t memorySpaceCode = 0;
                p->val.convert(memorySpaceCode);
                memorySpace = static_cast<MemorySpace>(memorySpaceCode);
            } else if (key == "addr") {
                addr = std::static_pointer_cast<const Expr>(DeserializeNode(p->val, zone));
                hasAddr = true;
            } else if (key == "size") {
                p->val.convert(size);
                hasSize = true;
            } else if (key == "id") {
                p->val.convert(id);
                hasId = true;
            }
        }

        INTERNAL_CHECK(hasAddr && hasSize && hasId) << "MemRef missing required fields (addr, size, or id)";

        return std::make_shared<MemRef>(memorySpace, addr, size, id);
    }

    std::optional<TileView> DeserializeTileView(const msgpack::object &obj, msgpack::zone &zone) {
        if (obj.is_nil()) {
            return std::nullopt;
        }

        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for TileView";

        TileView tileView;

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "valid_shape") {
                if (p->val.type == msgpack::type::ARRAY) {
                    for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
                        tileView.validShape.push_back(
                            std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
                    }
                }
            } else if (key == "stride") {
                if (p->val.type == msgpack::type::ARRAY) {
                    for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
                        tileView.stride.push_back(
                            std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
                    }
                }
            } else if (key == "start_offset") {
                tileView.startOffset = std::static_pointer_cast<const Expr>(DeserializeNode(p->val, zone));
            }
        }

        return tileView;
    }

    TypePtr DeserializeType(const msgpack::object &obj, msgpack::zone &zone) override {
        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for Type";

        std::string typeKind;
        uint8_t dtypeCode = 0;
        std::vector<ExprPtr> shape;
        std::vector<TypePtr> types;
        msgpack::object memrefObj;
        msgpack::object tileViewObj;
        bool hasMemref = false;
        bool hasTileView = false;

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "type_kind") {
                p->val.convert(typeKind);
            } else if (key == "dtype") {
                p->val.convert(dtypeCode);
            } else if (key == "shape") {
                if (p->val.type == msgpack::type::ARRAY) {
                    for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
                        shape.push_back(
                            std::static_pointer_cast<const Expr>(DeserializeNode(p->val.via.array.ptr[i], zone)));
                    }
                }
            } else if (key == "types") {
                if (p->val.type == msgpack::type::ARRAY) {
                    for (uint32_t i = 0; i < p->val.via.array.size; ++i) {
                        types.push_back(DeserializeType(p->val.via.array.ptr[i], zone));
                    }
                }
            } else if (key == "memref") {
                memrefObj = p->val;
                hasMemref = true;
            } else if (key == "tile_view") {
                tileViewObj = p->val;
                hasTileView = true;
            }
        }

        if (typeKind == "ScalarType") {
            return std::make_shared<ScalarType>(DataType(dtypeCode));
        } else if (typeKind == "TensorType") {
            if (hasMemref) {
                std::optional<MemRefPtr> memref = DeserializeMemRef(memrefObj, zone);
                return std::make_shared<TensorType>(shape, DataType(dtypeCode), memref);
            }
            return std::make_shared<TensorType>(shape, DataType(dtypeCode));
        } else if (typeKind == "TileType") {
            std::optional<MemRefPtr> memref;
            std::optional<TileView> tileView;

            if (hasMemref) {
                memref = DeserializeMemRef(memrefObj, zone);
            }
            if (hasTileView) {
                tileView = DeserializeTileView(tileViewObj, zone);
            }

            if (hasMemref && hasTileView) {
                return std::make_shared<TileType>(shape, DataType(dtypeCode), memref, tileView);
            } else if (hasMemref) {
                return std::make_shared<TileType>(shape, DataType(dtypeCode), memref);
            }
            return std::make_shared<TileType>(shape, DataType(dtypeCode));
        } else if (typeKind == "TupleType") {
            return std::make_shared<TupleType>(types);
        } else if (typeKind == "MemRefType") {
            return GetMemRefType();
        } else if (typeKind == "UnknownType") {
            return GetUnknownType();
        } else {
            throw RuntimeError("Unknown Type kind: " + typeKind);
        }
    }

    OpPtr DeserializeOp(const msgpack::object &obj) override {
        INTERNAL_CHECK(obj.type == msgpack::type::MAP) << "Expected map for Op";

        std::string name;
        bool isGlobalVar = false;

        msgpack::object_kv *p = obj.via.map.ptr;
        msgpack::object_kv *const pend = obj.via.map.ptr + obj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == "name") {
                p->val.convert(name);
            } else if (key == "is_global_var") {
                p->val.convert(isGlobalVar);
            }
        }

        if (isGlobalVar) {
            return std::make_shared<GlobalVar>(name);
        } else {
            return std::make_shared<Op>(name);
        }
    }

    msgpack::object GetFieldObj(const msgpack::object &fieldsObj, const std::string &fieldName) override {
        INTERNAL_CHECK(fieldsObj.type == msgpack::type::MAP) << "Expected map for fields";
        msgpack::object_kv *p = fieldsObj.via.map.ptr;
        msgpack::object_kv *const pend = fieldsObj.via.map.ptr + fieldsObj.via.map.size;
        for (; p < pend; ++p) {
            std::string key;
            p->key.convert(key);
            if (key == fieldName) {
                return p->val;
            }
        }
        throw RuntimeError("Missing required field: " + fieldName);
    }

private:
    std::unordered_map<uint64_t, IRNodePtr> idToPtr_;
};

// IRDeserializer implementation

IRDeserializer::IRDeserializer() : impl_(std::make_unique<Impl>()) {}

IRDeserializer::~IRDeserializer() = default;

IRNodePtr IRDeserializer::Deserialize(const std::vector<uint8_t> &data) {
    return impl_->Deserialize(data);
}

// Public API functions

IRNodePtr Deserialize(const std::vector<uint8_t> &data) {
    IRDeserializer deserializer;
    return deserializer.Deserialize(data);
}

IRNodePtr DeserializeFromFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    INTERNAL_CHECK(file.is_open()) << "Failed to open file for reading: " + path;
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    INTERNAL_CHECK(!file.fail()) << "Failed to read from file: " + path;

    return Deserialize(data);
}

} // namespace serialization
} // namespace ir
} // namespace pypto
