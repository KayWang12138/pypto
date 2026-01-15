/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file object.h
 * \brief
 */

#pragma once

#include <iostream>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "ir/utils.h"

namespace pto {

// Object type categories for ID generation.
enum class ObjectType {
    Program,
    Function,
    Statement,
    Operation,
    Value,
    Memory,
    Type,
};

// Attribute value supports multiple basic types.
using AttributeValue = std::variant<std::string, int64_t, double, bool>;

// Simple key/value attribute bag used across IR nodes.
using AttributeMap = std::map<std::string, AttributeValue>;

// Helper: streaming for attribute values.
inline std::ostream& operator<<(std::ostream& os, const AttributeValue& value)
{
    std::visit([&os](auto&& v) { os << v; }, value);
    return os;
}

// Base class for all IR objects.
class Object {
public:
    explicit Object(ObjectType type, const std::string &name = "")
        : id_(IDGen::NextID(type)), name_(name) {}
    virtual ~Object() = default;

    int GetID() const { return id_; }
    const std::string& GetName() const { return name_; }
    void SetName(std::string name) { name_ = name; }

    // Get the name with prefix for display/printing
    // - Program and Function: add @ prefix
    // - Value: add % prefix
    // - Other types: no prefix
    std::string GetPrefixedName() const {
        ObjectType type = GetObjectType();
        char prefix = '\0';
        if (type == ObjectType::Program || type == ObjectType::Function) {
            prefix = '@';
        } else if (type == ObjectType::Value) {
            prefix = '%';
        }

        if (prefix != '\0') {
            return std::string(1, prefix) + name_;
        }

        return name_;
    }

    // Get the SSA name for the object
    std::string GetSSAName() const {
        return GetPrefixedName() + "_" + std::to_string(id_);
    }

    // Each derived class must specify its object type.
    virtual ObjectType GetObjectType() const = 0;

    // Check if the attribute key can be set on this object.
    bool CanSetAttr(const std::string& key) const
    {
        auto itType = registry_.find(std::type_index(typeid(*this)));
        if (itType == registry_.end()) {
            return false;
        }
        const auto& attrInfo = itType->second;
        return attrInfo.find(key) != attrInfo.end();
    }

    // Get the expected type of an attribute key.
    std::optional<std::type_index> GetAttrKeyType(const std::string& key) const
    {
        auto itType = registry_.find(std::type_index(typeid(*this)));
        if (itType == registry_.end()) {
            return std::nullopt;
        }
        const auto& attrInfo = itType->second;
        auto it = attrInfo.find(key);
        if (it == attrInfo.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    // Set an attribute by key. Return false if the attribute key is not registered or type mismatches.
    template <typename T>
    bool SetAttr(const std::string& key, const T& value)
    {
        // Check if key is registered
        if (!CanSetAttr(key)) {
            std::cerr << "Attempt to set unregistered attribute key '" << key
                      << "' on type '" << typeid(*this).name() << "'" << std::endl;
            return false;
        }

        // Get expected type
        auto expectedType = GetAttrKeyType(key);
        if (!expectedType.has_value()) {
            return false;
        }

        // Get actual type index - no automatic conversion
        std::type_index actualType = std::type_index(typeid(T));

        // Verify type matches exactly
        if (actualType != expectedType.value()) {
            std::cerr << "Type mismatch: attribute key '" << key
                      << "' expects type " << expectedType.value().name()
                      << " but got type " << actualType.name() << std::endl;
            return false;
        }

        // Set the attribute value
        attributes_[key] = AttributeValue{value};

        return true;
    }

    // Check if the attribute key is set on this object.
    bool HasAttr(const std::string& key) const
    {
        return attributes_.find(key) != attributes_.end();
    }

    // Get an attribute by key. Return false if the attribute key is not set.
    template <typename T>
    bool GetAttr(const std::string& key, T& value) const
    {
        auto it = attributes_.find(key);
        if (it == attributes_.end()) {
            return false;
        }
        if (const auto* p = std::get_if<T>(&it->second)) {
            value = *p;
            return true;
        }
        return false;
    }

    AttributeValue GetAttr(const std::string& key) const
    {
        auto it = attributes_.find(key);
        if (it == attributes_.end()) {
            return AttributeValue{std::string("undefined")};
        }
        return it->second;
    }

    // Unset an attribute by key.
    void UnsetAttr(const std::string& key)
    {
        attributes_.erase(key);
    }

    // Get all set attribute keys on this object.
    std::vector<std::string> GetSetAttrKeys() const
    {
        std::vector<std::string> keys;
        for (const auto& [key, value] : attributes_) {
            if (value.index() != 0) {
                keys.push_back(key);
            }
        }
        return keys;
    }

protected:
    int id_;
    std::string name_;

private:
    // Attribute registry API (per C++ derived type) - only accessible by RegisterAllAttributes
    template <typename T, typename ValueType>
    static void RegisterAttrKey(const std::string& key)
    {
        // Check if ValueType is a supported type in AttributeValue variant
        static_assert(
            std::is_same_v<ValueType, std::string> ||
            std::is_same_v<ValueType, int64_t> ||
            std::is_same_v<ValueType, double> ||
            std::is_same_v<ValueType, bool>,
            "ValueType must be one of: std::string, int64_t, double, bool"
        );

        auto& attrInfo = registry_[std::type_index(typeid(T))];
        auto it = attrInfo.find(key);
        if (it != attrInfo.end()) {
            std::cerr << "Attribute key '" << key << "' already registered for type '" << typeid(T).name() << "'" << std::endl;
            return;
        }
        attrInfo.emplace(key, std::type_index(typeid(ValueType)));
    }

    // Friend function for centralized attribute registration
    friend void RegisterAllAttributes();

    using AttributeKeyTypeMap = std::unordered_map<std::string, std::type_index>;
    using AttributeRegistry = std::unordered_map<std::type_index, AttributeKeyTypeMap>;

    static AttributeRegistry registry_;
    AttributeMap attributes_;
};

// Centralized attribute registration function - called once during initialization
void RegisterAllAttributes();

}