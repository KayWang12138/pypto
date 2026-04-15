# FillPad Inplace Manual-Line Design

## Context

The current `pypto_block.language.op.manual` line exposes `fillpad(dst, src)` but does not expose a separate `fillpad_inplace(dst, src)`.

Today the block implementation incorrectly treats `fillpad(src, src)` as an inplace path in some backends. That behavior is not aligned with PTOAS or pto-isa. In ISA and PTOAS, `fillpad_inplace` is a distinct two-operand operation where:

- `src` provides the valid-region bounds.
- `dst` provides the target padded bounds and pad value.
- `src` and `dst` share the same backing storage address.
- `src` and `dst` do not need to be the same tile object and may carry different valid-shape metadata.

The same-tile call `fillpad(src, src)` is therefore not the semantic definition of inplace. It is only a degenerate case where source and destination metadata are also the same, which means the call does not produce useful padding.

## Goals

- Keep scope limited to the manual `plm.*` line.
- Add an explicit `plm.fillpad_inplace(dst, src)` API.
- Make `manual.fillpad` and `manual.fillpad_inplace` parallel, separate operations.
- Remove backend special-casing that rewrites `fillpad(src, src)` into inplace behavior.
- Add strong block-side validation that `fillpad_inplace(dst, src)` requires shared backing storage.
- Lower the PTO path to explicit `pto.tfillpad_inplace`.
- Lower the CCE path to explicit `TFILLPAD_INPLACE`.

## Non-Goals

- No auto/SSA API changes.
- No new helper API for alias construction.
- No changes to shared non-block monorepo layers.
- No attempt to preserve the old mistaken meaning of `fillpad(src, src)`.

## User-Facing Semantics

### `plm.fillpad(dst, src)`

- Remains the normal non-inplace fillpad operation.
- `dst` and `src` may use different backing storage.
- `dst` pad metadata controls the padding result.
- No special same-address or same-object behavior is implied.

### `plm.fillpad_inplace(dst, src)`

- New explicit inplace fillpad operation.
- `dst` and `src` must share the same underlying tile address.
- `dst` and `src` may have different valid-shape metadata.
- `src` valid-shape defines the source valid region.
- `dst` valid-shape defines the destination padded bounds.
- `dst` pad metadata controls the filled padding value.

### `plm.fillpad(src, src)`

- No longer lowers to inplace.
- It is treated as an ordinary non-inplace `fillpad` call.
- Because source and destination metadata are identical in that case, the call is valid but not useful for padding expansion.

## API Shape

Add the following manual API:

```python
def fillpad_inplace(out: Tile, tile: Tile) -> None:
    _op("manual.fillpad_inplace", [tile.unwrap()], out)
```

This matches the manual style already used by `fillpad(dst, src)` and keeps the destination as the rebound output tile.

## IR Design

Add a new manual op registration:

- `manual.fillpad_inplace`

Type deduction and validation rules:

- Require exactly two arguments: `src`, `out`.
- Require both operands to be `TileType`.
- Require `out.tile_view.pad != TilePad.null`.
- Require rank-2 static tile shapes.
- Require `src` and `out` rows and cols to match.
- Require `src` and `out` data type size compatibility matching current fillpad-family rules.
- Require `src` and `out` backing storage to be the same logical tile address.

The same-address validation should happen in block code before backend lowering so incorrect user code fails early and consistently across PTO and CCE.

## Address-Sharing Validation

`fillpad_inplace(dst, src)` is only legal when `dst` and `src` refer to the same backing storage.

The validation rule is:

- If both tiles resolve to concrete memref-backed manual tiles with distinct addresses, raise `ValueError`.
- If the block IR can prove the addresses differ, fail in the IR type-deduction path.
- If the block layer cannot resolve the address at type-deduction time, fail in backend codegen before emitting PTO or CCE.

For the initial implementation, manual tiles created through existing `make_tile(..., addr=...)` flows are expected to carry enough address information for this check in current block scope.

## Lowering Design

### CCE backend

Change `manual.fillpad` lowering:

- Remove the `src == dst` branch that currently emits `TFILLPAD_INPLACE`.
- Keep only the normal `TFILLPAD(dst, src)` lowering plus the existing null-pad source alias handling needed for non-inplace fillpad.

Add `manual.fillpad_inplace` lowering:

- Reuse the existing null-pad source alias construction logic for the source tile view when needed.
- Emit explicit `TFILLPAD_INPLACE(dst, src_alias_or_src)`.
- Do not restore `dst` to full valid-shape as part of lowering.
- Use the destination tile's own valid-shape metadata as the padded bounds.

The previous `SetValidShape(full_rows, full_cols)` step was only compensating for the old incorrect `fillpad(src, src)` reinterpretation and must not survive in the new design.

### PTO backend

Change `manual.fillpad` lowering:

- Remove the `src == dst` special case.
- Stop relying on implicit PTOAS recognition of same-value `pto.tfillpad`.
- Keep ordinary `pto.tfillpad` lowering plus null-pad source alias handling for non-inplace fillpad.

Add `manual.fillpad_inplace` lowering:

- Reuse the same source alias strategy when a null-pad source view is required.
- Emit explicit `pto.tfillpad_inplace ins(src : ...) outs(dst : ...)`.
- Preserve separate `src` and `dst` tile metadata, especially valid-shape.

## Tests

### Remove or rewrite old tests

Delete or rewrite tests that assert old behavior:

- CCE tests that expect `fillpad(src, src)` to become `TFILLPAD_INPLACE`.
- PTO tests that expect `fillpad(src, src)` to be special-cased.
- PTO-to-PTOAS tests that use `fillpad(src, src)` as the inplace trigger.

### Add new tests

Add manual-line tests for:

1. `plm.fillpad_inplace(dst, src)` lowers to `TFILLPAD_INPLACE` in CCE.
2. `plm.fillpad_inplace(dst, src)` lowers to `pto.tfillpad_inplace` in PTO.
3. `plm.fillpad(src, src)` does not lower to inplace and no longer triggers any inplace-specific backend path.
4. `plm.fillpad_inplace(dst, src)` rejects different backing addresses.
5. `plm.fillpad_inplace(dst, src)` preserves separate source and destination valid-shape semantics in generated code.

Runtime-style frontend examples may be updated to show the correct two-tile same-address pattern if a current test fixture already covers manual datacopy examples.

## Documentation Updates

Update manual operation documentation to reflect:

- `fillpad(dst, src)` is not inplace.
- `fillpad_inplace(dst, src)` is a separate manual operation.
- Inplace means shared backing storage with potentially different tile metadata, not necessarily the same tile object.

## Migration Notes

Existing manual user code that relied on `fillpad(src, src)` as inplace behavior must be migrated to:

1. Create two tile handles that share the same backing address.
2. Give `src` the source valid-shape metadata.
3. Give `dst` the target padded bounds and pad metadata.
4. Call `plm.fillpad_inplace(dst, src)`.

This is an intentional behavior correction, not a backward-compatible alias.

## Implementation Plan Summary

1. Add manual Python API export for `fillpad_inplace`.
2. Add `manual.fillpad_inplace` IR registration and validation.
3. Remove `manual.fillpad` same-tile inplace lowering from CCE.
4. Remove `manual.fillpad` same-tile inplace lowering from PTO.
5. Add explicit CCE lowering for `manual.fillpad_inplace`.
6. Add explicit PTO lowering for `manual.fillpad_inplace`.
7. Rewrite and add UT coverage.
8. Update manual docs.
