#pragma once

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

// Simple key/value attribute bag used across IR nodes.
using AttributeMap = std::map<std::string, std::string>;

// Base class for all IR objects.
class Object {
public:
    explicit Object(ObjectType type, std::string name = "")
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
        if (name_.empty()) {
            return name_;
        }
        
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

    // Each derived class must specify its object type.
    virtual ObjectType GetObjectType() const = 0;

    AttributeMap& Attributes() { return attributes_; }
    const AttributeMap& Attributes() const { return attributes_; }

protected:
    int id_;
    std::string name_;
    AttributeMap attributes_;
};

}