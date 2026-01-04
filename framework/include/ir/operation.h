// PTO-IR prototype: operation-level IR structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/utils.h"
#include "ir/value.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_schema.h"
#include "ir/op/op_payload.h"

#include <memory>
#include <ostream>
#include <string>

namespace pto {

// Base class for operations inside statements.
class Operation : public Object {
public:
    /// Default: invalid / placeholder op
    Operation();
    explicit Operation(Opcode opcode);
    Operation(Opcode opcode, std::string name);

    /// Full construction
    Operation(Opcode opcode,
              ValuePtrs ioprands,
              ValuePtrs ooprands,
              std::string name = "");
    
    ~Operation() override = default;

    ObjectType GetObjectType() const override { return ObjectType::Operation; }
    // ---- OpCode ----
    Opcode GetOpcode() const { return opcode_; }

    // ---- OpSchema ---- 
    const OpSchema& GetSchema() const {
        return GetOpSchema(opcode_);
    }

    // ---- IOprands ----
    const ValuePtrs& GetInputs() const { return ioprands_; }
    ValuePtrs& MutableInputs()  { return ioprands_; }
    void SetInputs(ValuePtrs inputs) { ioprands_ = std::move(inputs); }

    void AddInput(const ValuePtr& value) {
        ioprands_.push_back(value);
    }

    size_t GetNumInputs() const { return ioprands_.size(); }

    ValuePtr GetInput(size_t idx) const {
        return ioprands_.at(idx);
    }

    void SetInput(size_t idx, const ValuePtr& value) {
        ioprands_.at(idx) = value;
    }
    
    // ---- OOprands ----
    const ValuePtrs& GetOutputs() const { return ooprands_; }
    ValuePtrs&  MutableOutputs()  { return ooprands_; }

    void SetOutputs(ValuePtrs outputs) { ooprands_ = std::move(outputs); }

    void AddOutput(const ValuePtr& value) {
        ooprands_.push_back(value);
    }

    size_t GetNumOutputs() const { return ooprands_.size(); }

    ValuePtr GetOutput(size_t idx) const {
        return ooprands_.at(idx);
    }

    void SetOutput(size_t idx, const ValuePtr& value) {
        ooprands_.at(idx) = value;
    }

    // ---- Payload ----
    bool HasPayload() const { return payload_ != nullptr; }
    PayloadKind GetPayloadKind() const { return payload_ ? payload_->Kind() : PayloadKind::None; }

    void SetPayload(std::shared_ptr<OpPayload> p) { payload_ = std::move(p); }

    OpPayload* GetPayload() { return payload_.get(); }
    const OpPayload* GetPayload() const { return payload_.get(); }

    // ---- Verify (schema + payload) ----
    bool Verify(std::string* err = nullptr) const;

    // Pretty-print with the given indentation (in spaces).
    void Print(std::ostream& os, int indent = 0) const;
public:
    ValuePtrs ioprands_;
    ValuePtrs ooprands_;
    Opcode opcode_;

private:
    std::shared_ptr<OpPayload> payload_;
};

using OperationPtr = std::shared_ptr<Operation>;
} // namespace pto


