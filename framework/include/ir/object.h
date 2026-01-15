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

    // Attribute registry API (per C++ derived type).
    template <typename T>
    static void RegisterAttrKey(const std::string& key)
    {
        auto& keySet = registry_[std::type_index(typeid(T))];
        keySet.insert(key);
    }

    bool CanSetAttr(const std::string& key) const
    {
        auto itType = registry_.find(std::type_index(typeid(*this)));
        if (itType == registry_.end()) {
            return false;
        }
        const auto& keySet = itType->second;
        return keySet.find(key) != keySet.end();
    }

    template <typename T>
    bool SetAttr(const std::string& key, const T& value)
    {
        if (!CanSetAttr(key)) {
            // 未注册的属性 key，打印日志并拒绝设置。
            std::cerr << "Attempt to set unregistered attribute key '" << key
                      << "' on type '" << typeid(*this).name() << "'" << std::endl;
            return false;
        }
        attributes_[key] = AttributeValue{value};
        return true;
    }

    // Attribute helpers.
    bool HasAttr(const std::string& key) const
    {
        return attributes_.find(key) != attributes_.end();
    }

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

protected:
    int id_;
    std::string name_;
    AttributeMap attributes_;

private:
    using AttributeKeySet = std::unordered_set<std::string>;
    using AttributeRegistry = std::unordered_map<std::type_index, AttributeKeySet>;
    
    static AttributeRegistry registry_;
};

// Attribute registry API (per C++ derived type).
template <typename T>
void RegisterAttrKey(const std::string& key) {
    Object::RegisterAttrKey<T>(key);
}

}