// PTO-IR prototype: utility functions.
// All comments must remain in English for consistency.

#pragma once

#include <map>
#include <ostream>
#include <string>

namespace pto {

// Forward declaration
enum class DataType;
enum class ObjectType;

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

// Utility to help printing indentation.
void PrintIndent(std::ostream& os, int indent);

// Helper function to convert DataType enum to string name.
std::string DataTypeToString(DataType type);

// Helper function to convert string name to DataType enum.
DataType StringToValueType(const std::string& name);

} // namespace pto

