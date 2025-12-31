// ir/op_build.cpp
#include "ir/builder/op_builder.h"
#include "ir/builder/ir_builder.h"  
#include "ir/op/op_factory.h"
#include "ir/op/op_trait_rule.h"
#include "ir/op/op_schema.h"

#include <stdexcept>

namespace pto {

static void CheckPayloadRequirement(const OpSchema& schema, const OpPayload* payload) {
    if (schema.requiredPayload == PayloadKind::None) {
        if (payload) {
            throw std::runtime_error("BuildBySchema: payload provided but schema requires none");
        }
        return;
    }
    if (!payload) {
        throw std::runtime_error("BuildBySchema: missing required payload");
    }
    if (payload->Kind() != schema.requiredPayload) {
        throw std::runtime_error("BuildBySchema: payload kind mismatch");
    }
}


// ---------- main entry ----------
ValuePtrs BuildBySchema(IRBuilder& builder,
                        Opcode opcode,
                        ValuePtrs inputs,
                        std::shared_ptr<OpPayload> payload,
                        std::string name) {

    const auto& schema = GetOpSchema(opcode);
    CheckPayloadRequirement(schema, payload.get());
    
    // ---- arity check ----
    if (schema.arity != OpArity::Variadic) {
        if (inputs.size() != static_cast<size_t>(schema.arity)) {
            throw std::runtime_error("Arity mismatch for op " + std::string(schema.name));
        }
    }

    // ---- infer via trait rules ----
    ResultDesc desc;
    desc.dataType = inputs.empty() ? DataType::UNKNOWN : inputs[0]->GetDataType();

    const auto& rules = GetRulesForTraits(schema.traits);
    for (auto* r : rules) {
        r->Apply(schema, inputs, payload.get(), desc);
    }
    ValuePtrs autoOutputs;

    for (uint8_t i = 0; i < schema.numResults; ++i) {
        ValuePtr out;
        
        std::string outName = name;
        if (!name.empty() && schema.numResults > 1) {
            outName = name + "#" + std::to_string(i);
        }
        switch (desc.kind) {
        case ValueKind::Scalar:
            out = std::make_shared<Scalar>(desc.dataType, outName, ScalarValueKind::Symbolic);
            break;
        case ValueKind::Tensor:
            out = std::make_shared<Tensor>(desc.tensorShape, desc.dataType, outName);
            break;
        case ValueKind::Tile:
            out = std::make_shared<Tile>(desc.tileShape, desc.dataType, outName);
            break;
        default:
            throw std::runtime_error("BuildBySchema: unsupported result kind");
        }

        builder.AddToCompound(out);
        autoOutputs.push_back(out);
    }

    // CreateOp already does schema verify
    auto opPtr = CreateOp(opcode, inputs, autoOutputs, std::move(payload));
    builder.Emit(std::move(opPtr));
    return autoOutputs;
}

} // namespace pto
