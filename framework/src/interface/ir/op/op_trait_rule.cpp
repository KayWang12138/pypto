#include "ir/op/op_trait_rule.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace pto {

static DataType PickDataType(const ValuePtrs& xs) {
    for (auto& v : xs) {
        if (v && v->GetDataType() != DataType::UNKNOWN) return v->GetDataType();
    }
    return DataType::UNKNOWN;
}

// ---------------- Rules ----------------

class ProducesScalarRule final : public TraitRule {
public:
    int Priority() const override { return 100; }

    void Apply(const OpSchema&, const ValuePtrs& inputs,
               const OpPayload*, ResultDesc& desc) const override {
        desc.kind = ValueKind::Scalar;
        desc.dataType = PickDataType(inputs);
        desc.finalized = true;
    }
};

class ProducesTileRule final : public TraitRule {
public:
    int Priority() const override { return 90; }

    void Apply(const OpSchema&, const ValuePtrs& inputs,
               const OpPayload* payload, ResultDesc& desc) const override {
        if (desc.finalized) return;

        // Case 1: payload provides tile shape (e.g. View)
        if (payload && payload->Kind() == PayloadKind::View) {
            auto* vp = static_cast<const ViewPayload*>(payload);
            const auto& spec = vp->Spec();

            desc.kind = ValueKind::Tile;
            desc.dataType = PickDataType(inputs);
            desc.tileShape.clear();
            desc.tileShape.reserve(spec.shape.size());

            for (auto d : spec.shape) {
                if (d <= 0) throw std::runtime_error("ProducesTileRule: view shape dim must be > 0");
                desc.tileShape.push_back(static_cast<size_t>(d));
            }
            desc.tileValidShape = desc.tileShape;
            desc.finalized = true;
            return;
        }

        // Case 2: fallback: like first tile input
        for (auto& v : inputs) {
            if (v && v->GetValueKind() == ValueKind::Tile) {
                auto* t = static_cast<Tile*>(v.get());
                desc.kind = ValueKind::Tile;
                desc.dataType = t->GetDataType();
                desc.tileShape = t->GetShape();
                desc.tileValidShape = desc.tileShape;
                desc.finalized = true;
                return;
            }
        }

        throw std::runtime_error("ProducesTileRule: no shape source (need payload or tile input)");
    }
};

class ProducesTensorRule final : public TraitRule {
public:
    int Priority() const override { return 80; }

    void Apply(const OpSchema&, const ValuePtrs& inputs,
               const OpPayload* payload, ResultDesc& desc) const override {
        if (desc.finalized) return;

        // Case 1: TensorCreate payload provides shape and dtype
        if (payload && payload->Kind() == PayloadKind::TensorCreate) {
            auto* tcp = static_cast<const TensorCreatePayload*>(payload);
            const auto& spec = tcp->Spec();

            desc.kind = ValueKind::Tensor;
            desc.dataType = spec.dtype;
            desc.tensorShape = spec.shape;
            desc.finalized = true;
            return;
        }

        // Case 2: payload gives new shape (Reshape)
        if (payload && payload->Kind() == PayloadKind::Reshape) {
            auto* rp = static_cast<const ReshapePayload*>(payload);
            const auto& spec = rp->Spec();

            desc.kind = ValueKind::Tensor;
            desc.dataType = PickDataType(inputs);
            desc.tensorShape.clear();
            desc.tensorShape.reserve(spec.shape.size());

            if (!spec.validShape.empty()) {
                desc.tensorShape = spec.validShape;
            } else {
                for (auto d : spec.shape) {
                    if (d == -1) {
                        // new symbolic dim (minimal; you can replace with your SymbolDim helper)
                        desc.tensorShape.emplace_back(DataType::INT32, "%d", ScalarValueKind::Symbolic);
                    } else {
                        desc.tensorShape.emplace_back(static_cast<int64_t>(d));
                    }
                }
            }

            desc.finalized = true;
            return;
        }

        // Case 3: Reduce payload gives reduced shape
        if (payload && payload->Kind() == PayloadKind::Reduce) {
            auto* reducep = static_cast<const ReducePayload*>(payload);
            const auto& spec = reducep->Spec();

            // Get input shape (support both Tensor and Tile)
            if (inputs.empty() || !inputs[0]) {
                throw std::runtime_error("ProducesTensorRule (Reduce): no input provided");
            }
            
            std::vector<Scalar> inputShape;
            DataType inputDataType;
            ValueKind inputKind = inputs[0]->GetValueKind();
            
            if (inputKind == ValueKind::Tensor) {
                auto* inputTensor = static_cast<Tensor*>(inputs[0].get());
                inputShape = inputTensor->GetShape();
                inputDataType = inputTensor->GetDataType();
            } else if (inputKind == ValueKind::Tile) {
                auto* inputTile = static_cast<Tile*>(inputs[0].get());
                // Convert Tile's size_t shape to Scalar shape
                const auto& tileShape = inputTile->GetShape();
                for (size_t dim : tileShape) {
                    inputShape.push_back(Scalar(static_cast<int64_t>(dim)));
                }
                inputDataType = inputTile->GetDataType();
            } else {
                throw std::runtime_error("ProducesTensorRule (Reduce): input must be a Tensor or Tile");
            }
            
            int ndim = static_cast<int>(inputShape.size());

            // Normalize axis
            int axis = spec.axis;
            if (axis < 0) {
                axis = ndim + axis;
            }
            if (axis < 0 || axis >= ndim) {
                throw std::runtime_error("ProducesTensorRule (Reduce): axis out of range");
            }

            // Compute output shape
            desc.kind = ValueKind::Tensor;
            desc.dataType = inputDataType;
            desc.tensorShape.clear();

            if (spec.keepDim) {
                // Keep dimension with size 1
                for (int i = 0; i < ndim; ++i) {
                    if (i == axis) {
                        desc.tensorShape.emplace_back(int64_t{1});
                    } else {
                        desc.tensorShape.push_back(inputShape[i]);
                    }
                }
            } else {
                // Remove the reduced dimension
                for (int i = 0; i < ndim; ++i) {
                    if (i != axis) {
                        desc.tensorShape.push_back(inputShape[i]);
                    }
                }
            }

            desc.finalized = true;
            return;
        }

        // Case 4: like first tensor input
        for (auto& v : inputs) {
            if (v && v->GetValueKind() == ValueKind::Tensor) {
                auto* t = static_cast<Tensor*>(v.get());
                desc.kind = ValueKind::Tensor;
                desc.dataType = t->GetDataType();
                desc.tensorShape = t->GetShape();
                desc.finalized = true;
                return;
            }
        }

        throw std::runtime_error("ProducesTensorRule: no tensor shape source (need payload or tensor input)");
    }
};

class ElementwiseRule final : public TraitRule {
public:
    int Priority() const override { return 10; }

    void Apply(const OpSchema&, const ValuePtrs& inputs,
               const OpPayload*, ResultDesc& desc) const override {
        if (desc.finalized) return;

        // Priority: Tile > Tensor > Scalar
        for (auto& v : inputs) {
            if (v && v->GetValueKind() == ValueKind::Tile) {
                auto* t = static_cast<Tile*>(v.get());
                desc.kind = ValueKind::Tile;
                desc.dataType = t->GetDataType();
                desc.tileShape = t->GetShape();
                desc.tileValidShape = desc.tileShape;
                return;
            }
        }
        for (auto& v : inputs) {
            if (v && v->GetValueKind() == ValueKind::Tensor) {
                auto* t = static_cast<Tensor*>(v.get());
                desc.kind = ValueKind::Tensor;
                desc.dataType = t->GetDataType();
                desc.tensorShape = t->GetShape();
                return;
            }
        }

        desc.kind = ValueKind::Scalar;
        desc.dataType = PickDataType(inputs);
    }
};

class PureRule final : public TraitRule {
public:
    int Priority() const override { return 0; }
    void Apply(const OpSchema&, const ValuePtrs&,
               const OpPayload*, ResultDesc&) const override {
        // no-op (used by passes later)
    }
};

class TileBroadcastRule final : public TraitRule {
public:
    // Priority:
    // - higher than ElementwiseRule
    // - lower than ProducesTileRule (view)
    int Priority() const override { return 30; }

    void Apply(const OpSchema& schema,
               const ValuePtrs& inputs,
               const OpPayload*,
               ResultDesc& desc) const override {

        if (desc.finalized) return;

        // Only for elementwise ops
        if (!HasTrait(schema.traits, OpTrait::Elementwise)) return;

        const Tile* tile = nullptr;
        const Value* other = nullptr;

        // Find tile input
        for (auto& v : inputs) {
            if (v && v->GetValueKind() == ValueKind::Tile) {
                tile = static_cast<const Tile*>(v.get());
            } else if (v) {
                other = v.get();
            }
        }

        if (!tile || !other) return;

        // Case 1: tile ⊗ scalar
        if (other->GetValueKind() == ValueKind::Scalar) {
            desc.kind = ValueKind::Tile;
            desc.dataType = tile->GetDataType();
            desc.tileShape = tile->GetShape();
            desc.tileValidShape = desc.tileShape;
            desc.finalized = true;
            return;
        }

        // Case 2: tile ⊗ tile (same shape only)
        if (other->GetValueKind() == ValueKind::Tile) {
            auto* t2 = static_cast<const Tile*>(other);
            if (tile->GetShape() != t2->GetShape()) {
                throw std::runtime_error(
                    "TileBroadcastRule: tile-tile broadcast requires same shape");
            }
            desc.kind = ValueKind::Tile;
            desc.dataType = tile->GetDataType();
            desc.tileShape = tile->GetShape();
            desc.tileValidShape = desc.tileShape;
            desc.finalized = true;
            return;
        }
    }
};

class ViewPolymorphicRule final : public TraitRule {
public:
    int Priority() const override {
        // Must be higher than ProducesTile / ProducesTensor
        return 110;
    }

    void Apply(const OpSchema& schema,
               const ValuePtrs& inputs,
               const OpPayload* payload,
               ResultDesc& desc) const override {
        if (desc.finalized) return;
        if (schema.opcode != Opcode::OP_VIEW) return;
        if (inputs.size() != 1) return;

        const auto& in = inputs[0];
        if (!in) return;
        if (!payload) return;

        const auto* vp = static_cast<const ViewPayload*>(payload);
        const auto& spec = vp->Spec();

        // -------- Tensor -> Tensor (for Python frontend) --------
        // View operation extracts a sub-tensor from the input tensor.
        // Returns Tensor to keep operations at the Python frontend level.
        // Compiler passes can later convert this to Tile representation.
        if (in->GetValueKind() == ValueKind::Tensor) {
            const auto* t = static_cast<const Tensor*>(in.get());

            desc.kind       = ValueKind::Tensor;
            desc.dataType   = t->GetDataType();
            
            // Create tensor shape from view spec
            desc.tensorShape.clear();
            desc.tensorShape.reserve(spec.shape.size());
            for (auto d : spec.shape) {
                if (d <= 0) {
                    throw std::runtime_error("ViewPolymorphicRule: view shape dim must be > 0");
                }
                desc.tensorShape.emplace_back(static_cast<int64_t>(d));
            }
            
            desc.finalized  = true;
            return;
        }
        
        // -------- Tile -> Tile (for compiler passes) --------
        // When operating on Tile objects (introduced by compiler),
        // view returns another Tile.
        if (in->GetValueKind() == ValueKind::Tile) {
            const auto* tile = static_cast<const Tile*>(in.get());

            desc.kind       = ValueKind::Tile;
            desc.dataType   = tile->GetDataType();
            desc.tileShape  = spec.shape;
            desc.tileValidShape = desc.tileShape;
            desc.finalized  = true;
            return;
        }
    }
};

class AssemblePolymorphicRule final : public TraitRule {
public:
    int Priority() const override {
        return 110;
    }
    void Apply(const OpSchema& schema,
               const ValuePtrs& inputs,
               const OpPayload* payload,
               ResultDesc& desc) const override {
        if (desc.finalized) return;
        if (schema.opcode != Opcode::OP_ASSEMBLE) return;
        if (inputs.size() != 2) return;

        const auto& base  = inputs[0];
        const auto& patch = inputs[1];
        if (!base || !patch) return;
        if (!payload) return;

        // -------- Tensor + Tensor -> Tensor (for Python frontend) --------
        // Assemble updates a sub-region of the base tensor with patch tensor.
        // Both inputs and output are Tensors for Python frontend compatibility.
        // Returns a new Tensor (SSA functional update semantics).
        if (base->GetValueKind() == ValueKind::Tensor &&
            patch->GetValueKind() == ValueKind::Tensor) {

            const auto* t = static_cast<const Tensor*>(base.get());

            desc.kind         = ValueKind::Tensor;
            desc.dataType     = t->GetDataType();
            desc.tensorShape  = t->GetShape();   // SSA functional update
            desc.finalized    = true;
            return;
        }

        // -------- Tensor + Tile -> Tensor (for mixed mode) --------
        // When patch is a Tile (from compiler pass), assemble into Tensor base.
        if (base->GetValueKind() == ValueKind::Tensor &&
            patch->GetValueKind() == ValueKind::Tile) {

            const auto* t = static_cast<const Tensor*>(base.get());

            desc.kind         = ValueKind::Tensor;
            desc.dataType     = t->GetDataType();
            desc.tensorShape  = t->GetShape();   // SSA functional update
            desc.finalized    = true;
            return;
        }

        // -------- Tile + Tile -> Tile (for compiler passes) --------
        // When both are Tiles (compiler optimization), result is also Tile.
        if (base->GetValueKind() == ValueKind::Tile &&
            patch->GetValueKind() == ValueKind::Tile) {

            const auto* tile = static_cast<const Tile*>(base.get());

            desc.kind       = ValueKind::Tile;
            desc.dataType   = tile->GetDataType();
            desc.tileShape  = tile->GetShape();
            desc.tileValidShape = desc.tileShape;
            desc.finalized  = true;
            return;
        }
    }
};



// ---------------- registry ----------------
static const ViewPolymorphicRule     kViewPolymorphicRule;
static const AssemblePolymorphicRule kAssemblePolymorphicRule;

static const ProducesScalarRule kProducesScalarRule;
static const ProducesTileRule   kProducesTileRule;
static const ProducesTensorRule kProducesTensorRule;
static const ElementwiseRule    kElementwiseRule;
static const PureRule           kPureRule;
static const TileBroadcastRule  kTileBroadcasrRule;

const RuleList& GetRulesForTraits(uint32_t traits) {
    static std::unordered_map<uint32_t, RuleList> cache;

    auto it = cache.find(traits);
    if (it != cache.end()) return it->second;

    RuleList rules;

    rules.push_back(&kViewPolymorphicRule);
    rules.push_back(&kAssemblePolymorphicRule);


    if (HasTrait(traits, OpTrait::ProducesScalar)) rules.push_back(&kProducesScalarRule);
    if (HasTrait(traits, OpTrait::ProducesTile))   rules.push_back(&kProducesTileRule);
    if (HasTrait(traits, OpTrait::ProducesTensor)) rules.push_back(&kProducesTensorRule);
    if (HasTrait(traits, OpTrait::Elementwise)){ 
        rules.push_back(&kElementwiseRule);
        rules.push_back(&kTileBroadcasrRule);
    }
    if (HasTrait(traits, OpTrait::Pure))           rules.push_back(&kPureRule);

    std::sort(rules.begin(), rules.end(),
              [](const TraitRule* a, const TraitRule* b) {
                  return a->Priority() > b->Priority();
              });

    auto [insIt, _] = cache.emplace(traits, std::move(rules));
    return insIt->second;
}

} // namespace pto
