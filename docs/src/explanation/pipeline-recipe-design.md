# How RecipePlan separates workflow intent from data

A Recipe is a portable description of public Frame calls. It is not a saved
Frame, a Dask graph, a runtime operation object, or a copy of display history.

## One owner for each concern

- `lineage` is the complete provenance of Frame inputs.
- `previous` is the immediate, process-local receiver reference used for
  before-and-after notebook comparisons; it is not serialized.
- `operation_history` is a derived compatibility view for display and does not
  become executable Recipe structure.
- `RecipePlan` owns reusable invocation intent and named runtime input slots.

Keeping these roles separate lets replay rebuild the receiver-side workflow
without pretending that a process-local reference or a display record is a
portable execution graph.

## External arrays and persistence

NumPy and Dask operands have no sampling rate, channel metadata, or source-time
meaning. Recipe models them as named `array` inputs rather than inventing a
temporary Frame and its metadata. Callers provide the concrete array again at
replay, preserving the array-level lazy boundary.

WDF stores one concrete typed Frame and display history. Recipe stores reusable
operation intent. Use WDF when the artifact is a result to inspect, and Recipe
when the artifact is a workflow to replay; the two schemas evolve independently.

## Safety and extension

The Recipe loader validates the complete graph and fails closed for unknown
operations, versions, bindings, fields, or malformed values. This keeps a loaded
plan deterministic instead of importing hidden executable code.

Portable extensions use an immutable registry derived from the default registry;
they never mutate process-wide state. The current extension and handler contract
is maintained in the [Frame and Operation extension guide](../contributing/frame-operation-extensions.md).

The durable low-level contract is recorded in the
[Recipe v2 architecture ADR](https://github.com/kasahart/wandas/blob/main/docs/design/2026-07-13-recipe-v2-architecture.md).
