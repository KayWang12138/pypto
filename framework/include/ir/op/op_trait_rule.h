// ir/op/trait_rule.h
#pragma once

#include "ir/value.h"
#include <vector>
#include "ir/op/op_schema.h"
#include "ir/op/op_payload.h"     // OpPayload

namespace pto {

struct ResultDesc {
    ValueKind kind = ValueKind::Scalar;
    DataType  dataType = DataType::UNKNOWN;

    // Only for Tensor / Tile
    std::vector<Scalar> tensorShape;
    std::vector<size_t> tileShape;
    std::vector<size_t> tileValidShape;

    bool finalized = false;
};

class TraitRule {
public:
    virtual ~TraitRule() = default;

    // Apply rule to ResultDesc. Rule may set/refine desc.
    virtual void Apply(const OpSchema& schema,
                       const ValuePtrs& inputs,
                       const OpPayload* payload,
                       ResultDesc& desc) const = 0;

    // Higher runs earlier.
    virtual int Priority() const = 0;
};

using RuleList = std::vector<const TraitRule*>;

// Return ordered rules for a given trait-mask.
const RuleList& GetRulesForTraits(uint32_t traits);

} // namespace pto