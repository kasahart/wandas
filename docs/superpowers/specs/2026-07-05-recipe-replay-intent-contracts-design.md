# Recipe Replay Intent Contracts Design

## Goal

Define the small Recipe contract needed for issue #259: a Recipe should only store work that can be replayed later with the same user-visible meaning.

If Wandas cannot preserve that meaning, extraction must fail with `RecipeExtractionError` instead of saving a misleading recipe.

## Scope

This PR continues from the Recipe support matrix and locks down the risky replay boundaries that are already visible in the prototype:

- `fix_length(duration=...)`
- channel selection
- time selection
- `add_channel(...)`
- `add_with_snr(...)`

This PR does not define WDF Recipe persistence, sklearn export, joblib/skops export, or a broad new Recipe serialization schema. Those remain in #257 and #258.

## Core Rule

Recipe extraction stores user intent, not incidental runtime details.

When the only available record is a runtime helper detail and not a stable user intent, the extractor must either:

- convert it to an explicit public replay argument, if that preserves meaning, or
- reject extraction with `RecipeExtractionError`.

Recipe replay must continue to call existing frame APIs. Recipe code must not duplicate numerical processing, metadata propagation, source-time updates, or Dask graph construction.

## Contract Table

| Boundary | Contract | Why |
| --- | --- | --- |
| `fix_length(length=...)` | Store `length`. | The user asked for a sample count, and replaying the same sample count is stable. |
| `fix_length(duration=...)` | Store the resolved `length`, not `duration`. | Replaying the same duration on a different sampling rate would produce a different sample count. The current Recipe contract promises the same output length, not the same wall-clock duration request. |
| Channel selection by name, index, range, index list, or boolean mask | Store the public selection form and replay through `get_channel(...)` or `frame[...]`. | These selections have stable replay meaning. |
| Channel selection by callable or regex | Reject extraction. | The selected channels are the result of executable or pattern logic that may not be portable or stable. |
| Literal dict channel query | Store the dict if all values are simple Recipe literals. | This preserves query intent without storing executable code. |
| Dict channel query containing regex or non-literal values | Reject extraction. | The query cannot be represented as portable Recipe data. |
| Time selection by continuous slice | Store the slice and replay through `frame[...]`. | Existing frame indexing owns source-time updates. |
| Time point selection or fancy time selection | Reject extraction. | The source-time contract for non-contiguous time picks is not defined. |
| `add_channel(ChannelFrame, ...)` | Store the named added frame input and public options such as `align`, `label`, and `suffix_on_dup`. Do not store the added frame data. | Frame data is a runtime input, not Recipe payload. Existing `add_channel` owns metadata and source-time behavior. |
| `add_channel(ndarray|dask.array, ...)` | Store the named raw data input and public options, including `source_time_offset`. Do not store raw array values. | Raw data is a runtime input. `source_time_offset` is part of the user's public add-channel request. |
| `add_with_snr(...)` | Store only `snr` and the two frame inputs. Do not store implicit length alignment as an extra `fix_length` step. | Length alignment is helper behavior inside `add_with_snr`; adding it to the Recipe would change the user-visible replay contract. |

## User-Facing Documentation

The support matrix remains the first page for beginners. It should say what is supported, what is not supported, and which Recipe class to use.

The detailed extraction-boundaries page remains the deeper engineering reference. It may use internal terms such as `operation_graph`, `MethodStep`, and `IndexingStep`.

## Implementation Units

1. Update the support matrix so the contract table above is visible in beginner language.
2. Add focused tests for any listed boundary that is not already covered.
3. Keep existing passing behavior where it already follows this contract.
4. Change extractor behavior only where it silently accepts a boundary that this design marks as unsupported.

## Error Handling

Unsupported extraction boundaries must raise `RecipeExtractionError` when a processed frame exists and Recipe extraction is attempted.

Some invalid frame operations, such as non-contiguous time indexing with source-time offsets, may already fail in the frame API before a processed frame exists. Those cases do not need to be converted into `RecipeExtractionError`; the Recipe contract should document that they are outside the supported Recipe boundary.

The error message should say what kind of selection or graph is unsupported and point users toward a supported form when possible. It should not suggest WDF persistence or sklearn export as a workaround.

## Testing

Use focused tests in `tests/pipeline/test_recipe.py` unless a smaller existing test file is more appropriate.

Tests should prove:

- supported cases replay through public frame APIs and preserve the relevant user intent;
- unsupported cases raise `RecipeExtractionError`;
- `add_channel` and `add_with_snr` do not hide extra frame operations or array payloads inside the Recipe.

Full test coverage for all Recipe classes is not required in this PR; this PR should only cover the contract boundaries above.

## Issue Closure

This PR should keep `Related #259` while the implementation is limited to documenting and pinning these contract boundaries.

Switch to `Closes #259` only if all acceptance criteria in #259 are satisfied:

- accepted cases preserve intent and replay through frame APIs;
- unsupported cases fail with `RecipeExtractionError`;
- new intent representation, if any, is documented and tested;
- Recipe code does not duplicate numerical logic, metadata propagation, or Dask graph construction.

The expected default for this PR is `Related #259`.
