# Index-Wise Source-Time Operations Design

## Context

Issue #227 asks for a clear contract when frames or channels with different
`source_time_offset` values are combined. The current implementation already
performs frame-frame binary operations by array index, checks sampling rate,
channel count, and shape, and lets the result inherit the left operand's
`source_time_offset`.

That behavior is intentional for common workflows. Users may trim separate
source regions and then combine the resulting arrays by index. Rejecting offset
mismatches would make those workflows cumbersome and would imply that
`source_time_offset` is an automatic alignment mechanism, which it is not.

## Goal

Clarify and test the existing index-wise operation contract without changing
runtime behavior.

The work should make it explicit that `source_time_offset` describes the source
timeline carried by a frame. It does not cause binary operators or
`channel_difference()` to align, trim, pad, or reject data.

## Contract

Frame-frame binary operators such as `+`, `-`, `*`, `/`, and `**` operate on the
current array indices. They do not perform source-time alignment.

The existing compatibility checks remain:

- sampling rates must match
- channel counts must match
- full frame shapes must match

`source_time_offset` values are not compared and mismatches are allowed. The
result inherits the left operand's source-time offset, matching the existing
`_create_new_instance()` behavior.

`channel_difference()` is also index-wise. It subtracts the selected reference
channel from each channel in the same frame by array index. Channel-to-channel
`source_time_offset` mismatches are allowed, and the result keeps the input
frame's source-time offsets.

## Non-Goals

This issue must not add automatic source-time alignment to operators. It must
not silently trim, pad, resample, or reject data based on offset differences.

This issue also should not add a new public alignment API. A future explicit API
could provide a source-time intersection workflow, but that API should be
designed separately so users can choose alignment intentionally.

## Implementation Shape

No numerical logic change is required.

Update frame-facing documentation and docstrings where they currently read as
implicit behavior:

- `BaseFrame._binary_operand_op()` should state that frame-frame operations are
  index-wise, preserve the left operand's source-time offset, and do not compare
  offsets.
- `ChannelProcessingMixin.channel_difference()` should state that it is
  index-wise within one frame and does not reject per-channel offset mismatch.
- Public docs should include a short note that `source_time_offset` is metadata,
  not an automatic alignment key for binary operators.

Keep the existing sampling-rate, channel-count, and shape errors unchanged.

## Tests

Update the existing binary operation test that currently demonstrates left
offset inheritance so the name and docstring describe the public contract:
offset mismatches are allowed, the operation is index-wise, and the result keeps
the left operand's `source_time_offset`.

Add or update a focused `channel_difference()` test using per-channel
`source_time_offset` values. It should verify that the method does not reject
mismatches, computes by array index, and preserves the input offsets.

Relevant verification targets:

- `uv run pytest tests/frames/test_channel_frame.py`
- `uv run pytest tests/core/test_base_frame_lineage.py`
- `uv run ruff check wandas tests --config=pyproject.toml -v`
- `uv run ty check wandas tests`

## Future Work

A future explicit source-time alignment API may be useful for workflows that
want to combine only the overlapping source-time interval. That should be an
opt-in API with clear trimming and sample-grid rules, not an implicit behavior
of Python operators.
