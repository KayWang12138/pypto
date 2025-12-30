#include "ir/op/op_verify_rule.h"
#include "ir/operation.h"
#include "ir/type.h"
#include "ir/op/op_payload.h"
#include <unordered_map>

namespace pto {

class ViewVerifyRule final : public VerifyRule {
public:
    bool Apply(const Operation& op, std::string* err) const override {
        const auto& inputs  = op.GetInputs();
        const auto& outputs = op.GetOutputs();

        if (inputs.size() != 1 || outputs.size() != 1) {
            if (err) *err = "View: expects 1 input and 1 result";
            return false;
        }

        const auto& in  = inputs[0];
        const auto& out = outputs[0];
        if (!in || !out) {
            if (err) *err = "View: null operand";
            return false;
        }

        const auto* vp = static_cast<const ViewPayload*>(op.GetPayload());
        const auto& spec = vp->Spec();

        // ---- Tensor -> Tensor (for Python frontend) ----
        if (in->GetValueKind() == ValueKind::Tensor) {
            if (out->GetValueKind() != ValueKind::Tensor) {
                if (err) *err = "View(Tensor): result must be Tensor";
                return false;
            }
            
            const auto* t = static_cast<const Tensor*>(in.get());
            if (spec.offset.size() != t->GetShape().size()) {
                if (err) *err = "View(Tensor): offset rank mismatch";
                return false;
            }
            return true;
        }

        // ---- Tile -> Tile (for compiler passes) ----
        if (in->GetValueKind() == ValueKind::Tile) {
            if (out->GetValueKind() != ValueKind::Tile) {
                if (err) *err = "View(Tile): result must be Tile";
                return false;
            }
            
            const auto* tile = static_cast<const Tile*>(in.get());
            if (spec.offset.size() != tile->GetShape().size()) {
                if (err) *err = "View(Tile): offset rank mismatch";
                return false;
            }
            return true;
        }

        if (err) *err = "View: input must be Tensor or Tile";
        return false;
    }
};


class AssembleVerifyRule final : public VerifyRule {
public:
    bool Apply(const Operation& op, std::string* err) const override {
        const auto& inputs  = op.GetInputs();
        const auto& outputs = op.GetOutputs();

        if (inputs.size() != 2 || outputs.size() != 1) {
            if (err) *err = "Assemble: expects 2 inputs and 1 result";
            return false;
        }

        const auto& base  = inputs[0];
        const auto& patch = inputs[1];
        const auto& out   = outputs[0];

        if (!base || !patch || !out) {
            if (err) *err = "Assemble: null operand";
            return false;
        }

        const auto* ap = static_cast<const AssemblePayload*>(op.GetPayload());
        const auto& offset = ap->Spec().offset;

        // ---- Tensor + Tensor -> Tensor (for Python frontend) ----
        if (base->GetValueKind() == ValueKind::Tensor &&
            patch->GetValueKind() == ValueKind::Tensor) {

            if (out->GetValueKind() != ValueKind::Tensor) {
                if (err) *err = "Assemble(Tensor,Tensor): result must be Tensor";
                return false;
            }

            const auto* t = static_cast<const Tensor*>(base.get());
            if (offset.size() != t->GetShape().size()) {
                if (err) *err = "Assemble(Tensor,Tensor): offset rank mismatch";
                return false;
            }
            return true;
        }

        // ---- Tensor + Tile -> Tensor (for mixed mode) ----
        if (base->GetValueKind() == ValueKind::Tensor &&
            patch->GetValueKind() == ValueKind::Tile) {

            if (out->GetValueKind() != ValueKind::Tensor) {
                if (err) *err = "Assemble(Tensor,Tile): result must be Tensor";
                return false;
            }

            const auto* t = static_cast<const Tensor*>(base.get());
            if (offset.size() != t->GetShape().size()) {
                if (err) *err = "Assemble(Tensor,Tile): offset rank mismatch";
                return false;
            }
            return true;
        }

        // ---- Tile + Tile -> Tile (for compiler passes) ----
        if (base->GetValueKind() == ValueKind::Tile &&
            patch->GetValueKind() == ValueKind::Tile) {

            if (out->GetValueKind() != ValueKind::Tile) {
                if (err) *err = "Assemble(Tile,Tile): result must be Tile";
                return false;
            }

            const auto* t = static_cast<const Tile*>(base.get());
            if (offset.size() != t->GetShape().size()) {
                if (err) *err = "Assemble(Tile,Tile): offset rank mismatch";
                return false;
            }
            return true;
        }

        if (err) *err = "Assemble: invalid operand kinds (supported: Tensor+Tensor, Tensor+Tile, Tile+Tile)";
        return false;
    }
};

const static ViewVerifyRule kViewVerifyRule;
const static AssembleVerifyRule kAssembleVerifyRule;

using VerifyRuleTable = std::unordered_map<Opcode, const VerifyRule*>;

static const VerifyRuleTable kVerifyRules = {
    {Opcode::OP_VIEW,     &kViewVerifyRule},
    {Opcode::OP_ASSEMBLE, &kAssembleVerifyRule},
};

const VerifyRule* GetVerifyRuleForOpcode(Opcode op) {
    auto it = kVerifyRules.find(op);
    if (it != kVerifyRules.end())
        return it->second;
    return nullptr;
}

} // namespace