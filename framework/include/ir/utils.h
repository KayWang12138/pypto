// PTO-IR prototype: utility functions.
// All comments must remain in English for consistency.

#pragma once

#include <map>
#include <ostream>
#include <string>

namespace pto {

// Forward declaration
enum class DataType;

// Simple key/value attribute bag used across IR nodes.
using AttributeMap = std::map<std::string, std::string>;

// Object type categories for ID generation.
enum class ObjectType {
    Program,
    Function,
    Statement,
    Operation,
    Value,
    Memory,
};

// ID generator for different object types.
// Each object type maintains its own independent ID counter.
class IDGen {
public:
    // Get the next ID for the given object type.
    static int NextID(ObjectType type);

    // Reset the ID counter for a specific type.
    static void Reset(ObjectType type);

    // Reset all ID counters.
    static void ResetAll();

private:
    static std::map<ObjectType, int> counters_;
};

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

protected:
    int id_;
    std::string name_;
};

// Base class for IR objects that can carry attributes.
class AttributeHolder {
public:
    AttributeHolder() = default;
    virtual ~AttributeHolder() = default;

    AttributeMap& Attributes() { return attributes_; }
    const AttributeMap& Attributes() const { return attributes_; }

protected:
    AttributeMap attributes_;
};

// Utility to help printing indentation.
void PrintIndent(std::ostream& os, int indent);

// Helper function to convert DataType enum to string name.
std::string DataTypeToString(DataType type);

// Helper function to convert string name to DataType enum.
DataType StringToValueType(const std::string& name);

} // namespace pto

