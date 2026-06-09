# ADR: xarray-native Migration Plan / xarray-native 移行計画

- **Status**: Proposed / In progress
- **Date**: 2026-06-09
- **Related PR**: https://github.com/kasahart/wandas/pull/213
- **Related ADR**: `docs/design/2025-11-19-channel-wise-chunking.md`

## Summary / 要約

Wandas は長期的に、独自の `BaseFrame + Dask array + metadata + axis-like state` を中核データモデルとして育て続けるのではなく、xarray の `DataArray` / `Dataset` を中核表現として使う方向へ移行する。

ただし、Wandas の価値は xarray 自体ではなく、音声・波形解析のドメイン API、操作履歴、channel metadata、signal-safe chunk policy にある。そのため、既存 API を一気に廃止せず、段階的に xarray-native な内部表現と I/O / operation 実行へ寄せる。

PR #213 はこの移行の **Phase 1: Bridge** を主に実装する。Phase 2-4 は一部の足場だけ入り、完了扱いにはしない。

## Problem Statement / 背景と問題

現行 Wandas は以下を独自に管理している。

- Dask-backed frame data
- channel labels / units / refs / extra metadata
- sampling rate and frame metadata
- operation history
- time / frequency / band-like axes
- signal-safe chunking convention
- WDF-centered persistence

この設計はプロトタイプとして有効だったが、長期的には xarray がすでに解いている labelled multi-dimensional array、dims、coords、attrs、Dask integration、NetCDF/Zarr I/O、selection/alignment と責務が重なる。

Wandas がこの領域を独自に再実装し続けると、保守コストが増え、科学技術 Python stack との接続性も弱くなる。

## Decision / 方針

Wandas は xarray の subclass にはしない。xarray の公式方針に従い、まずは composition と bridge API で進める。

```python
frame = wd.read_wav("audio.wav")
da = frame.to_xarray()
restored = wd.from_xarray(da)
```

将来的には accessor も検討できる。

```python
da.wd.stft()
da.wd.describe()
```

ただし、最初から accessor 中心にすると既存 Wandas API の互換性とプロダクト感を失いやすいため、Phase 1-2 では frame class を残す。

## Non-goals for PR #213 / PR #213 でやらないこと

PR #213 は full xarray-native migration ではない。以下は未完了であり、follow-up とする。

- `BaseFrame` の独自責務の完全撤去
- xarray `DataArray` を唯一の authoritative storage にすること
- operations の `xr.apply_ufunc` / `map_blocks` / `map_overlap` 移行
- strict / blockwise execution mode の正式設計
- Zarr support
- WDF の再定義または非推奨化
- xarray accessor `.wd` の提供

## Target xarray Schema / 目標 schema

### ChannelFrame

```python
xr.DataArray(
    data,
    dims=("channel", "time"),
    coords={
        "channel": labels,
        "time": time_seconds,
        "unit": ("channel", units),
        "ref": ("channel", refs),
    },
    attrs={
        "wandas_frame_type": "ChannelFrame",
        "sampling_rate": sampling_rate,
        "label": label,
        "metadata": metadata,
        "operation_history": history,
        "channel_metadata": channel_metadata,
    },
)
```

### SpectralFrame

```python
xr.DataArray(
    data,
    dims=("channel", "frequency"),
    coords={
        "channel": labels,
        "frequency": np.fft.rfftfreq(n_fft, 1 / sampling_rate),
        "unit": ("channel", units),
        "ref": ("channel", refs),
    },
    attrs={
        "wandas_frame_type": "SpectralFrame",
        "sampling_rate": sampling_rate,
        "n_fft": n_fft,
        "window": window,
        "operation_history": history,
    },
)
```

### SpectrogramFrame

```python
xr.DataArray(
    data,
    dims=("channel", "frequency", "time"),
    coords={
        "channel": labels,
        "frequency": np.fft.rfftfreq(n_fft, 1 / sampling_rate),
        "time": frame_times,
        "unit": ("channel", units),
        "ref": ("channel", refs),
    },
    attrs={
        "wandas_frame_type": "SpectrogramFrame",
        "sampling_rate": sampling_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": window,
        "operation_history": history,
    },
)
```

### NOctFrame

```python
xr.DataArray(
    data,
    dims=("channel", "band"),
    coords={
        "channel": labels,
        "band": center_frequencies,
        "unit": ("channel", units),
        "ref": ("channel", refs),
    },
    attrs={
        "wandas_frame_type": "NOctFrame",
        "sampling_rate": sampling_rate,
        "fmin": fmin,
        "fmax": fmax,
        "n": n,
        "G": G,
        "fr": fr,
        "operation_history": history,
    },
)
```

### RoughnessFrame

Mono roughness is represented without a `channel` dimension because data shape is `(bark, time)`, but channel metadata must still round-trip through attrs.

```python
xr.DataArray(
    data,
    dims=("bark", "time"),
    coords={
        "bark": bark_axis,
        "time": time_seconds,
    },
    attrs={
        "wandas_frame_type": "RoughnessFrame",
        "sampling_rate": sampling_rate,
        "overlap": overlap,
        "channel_metadata": singleton_channel_metadata,
    },
)
```

Multi-channel roughness uses `dims=("channel", "bark", "time")`.

## coords vs attrs / coords と attrs の責務

`coords` に置くもの:

- selection / alignment / validation に効く軸情報
- `channel`, `time`, `frequency`, `band`, `bark`
- channel-scoped `unit`, `ref`

`attrs` に置くもの:

- `wandas_frame_type`
- `sampling_rate`
- `label`
- `metadata`
- `operation_history`
- frame constructor parameters such as `n_fft`, `hop_length`, `fmin`, `fmax`, `n`, `G`, `fr`, `overlap`
- `channel_metadata` for metadata that cannot be represented as simple numeric/string coords

Important: xarray does not interpret or reliably maintain `attrs` through arbitrary operations. Wandas must explicitly update and validate attrs when reconstructing frames.

## Coordinate Validation / 座標検証方針

Wandas frame classes currently cannot represent arbitrary xarray-selected axes such as shifted time origins, decimated time axes, reversed frequencies, or reordered octave bands. Therefore `from_xarray()` should reject non-canonical coordinates instead of silently rebuilding a misleading frame.

Validation rules:

- `time`: must start at 0 and match `sampling_rate` or `hop_length` spacing.
- `frequency`: must match canonical ascending `np.fft.rfftfreq(n_fft, 1 / sampling_rate)`.
- `band`: must match canonical NOct center frequencies from `fmin`, `fmax`, `n`, `G`, `fr`.
- length mismatches must fail before frame construction.

This is intentionally strict. A future frame model may preserve selected coordinate offsets, but PR #213 does not introduce that capability.

## Signal-safe Chunk Policy / 信号処理として安全な chunk policy

xarray adoption must not mean `chunks="auto"` for waveform operations. Time and frequency axes have signal-processing semantics.

Default storage/analysis chunks:

```python
ChannelFrame:     {"channel": 1, "time": -1}
SpectralFrame:    {"channel": 1, "frequency": -1}
SpectrogramFrame: {"channel": 1, "frequency": -1, "time": -1}
NOctFrame:        {"channel": 1, "band": -1}
RoughnessFrame:   mono {"bark": -1, "time": -1}, multi {"channel": 1, "bark": -1, "time": -1}
```

Operation policy:

| Operation type | Split time/frequency chunks | Default policy |
| --- | --- | --- |
| Elementwise gain / abs / power | Safe | allow |
| trim / channel selection | Safe | allow |
| global normalize / RMS scalar | Unsafe unless reduction is explicit | require contiguous core dim |
| FFT / IFFT | Unsafe across split transform dim | require contiguous core dim |
| STFT / Welch | Conditionally safe with overlap | strict first, blockwise later |
| FIR / convolution / moving RMS | Conditionally safe with overlap | future `map_overlap` mode |
| IIR / `filtfilt` / resampling | Generally unsafe blockwise | strict single core dim first |
| NOct synthesis | Requires whole frequency spectrum | require contiguous `frequency` |

Strict mode should be default. Any blockwise/overlap implementation must be explicit and recorded in `operation_history`.

## Phase Plan / 移行計画

### Phase 1: Bridge - completed in PR #213

Goal: expose xarray interoperability while keeping existing Wandas API compatible.

Delivered:

- `frame.to_xarray()`
- `frame.xr`
- `wd.from_xarray()`
- `frame.to_netcdf(path)`
- `wd.open_netcdf(path)`
- xarray schema for current frame classes
- strict coordinate validation on import
- NetCDF-safe attrs encoding/decoding
- complex real/imag encoding for NetCDF3 compatibility
- signal-safe chunk validation based on xarray dims

### Phase 2: Internal xarray as authoritative storage - not complete

Goal: make xarray `DataArray` the long-term core data representation.

Required work:

1. Decide `_xr` ownership rules.
2. Remove or reduce direct `_data` mutation paths.
3. Convert `.data`, `.compute()`, `.labels`, `.channels`, `.metadata`, `.operation_history` to compatibility properties over `_xr` where possible.
4. Define when frame constructors accept raw arrays vs `DataArray`.
5. Add tests that direct xarray storage updates preserve all legacy API behavior.
6. Update documentation to state that frame semantics are defined by xarray schema.

Exit criteria:

- There is a single authoritative storage object per frame.
- Legacy `_data` access is either removed, deprecated, or clearly marked as an escape hatch.
- Existing frame operations no longer rely on independent duplicated metadata state where xarray schema can provide it.

### Phase 3: xarray-aware operations - completed in PR #213 follow-up

Goal: move safe representative operations to xarray/Dask-native execution while preserving signal correctness. This phase does not mean every existing operation has been rewritten; it means the dispatch path, strict execution semantics, representative operations, and verification pattern are in place.

Delivered:

1. Added `AudioOperation.process_xarray()` and `AudioOperation.process_dataarray()` so real operation instances can opt into xarray-native execution while mocks/stubs and unsupported operations keep the legacy `process()` path.
2. Added execution metadata to operation history for xarray-aware operations without changing the existing `params` structure.
3. Implemented `xr.apply_ufunc` strict core-dim execution for:
   - `normalize(axis=-1)` / time-axis normalization
   - `fft()` from `ChannelFrame` to `SpectralFrame`
   - `rms_trend()` moving-window RMS in strict whole-time mode
4. Implemented exact xarray/Dask reduction execution for:
   - `remove_dc()` over the `time` dimension
5. Kept unsafe split-core operations guarded by strict chunk validation.
6. Added numerical equivalence tests and execution-history tests for the xarray-aware paths.

Blockwise / overlap decision:

- No approximate blockwise mode is enabled in this phase.
- `map_overlap` is intentionally not added until a specific operation has a reviewed boundary/trim definition.
- FIR or moving-window blockwise execution should be a separate follow-up with operation-specific numerical equivalence tests.

Exit criteria status:

- Representative `xr.apply_ufunc` strict operations: complete.
- Representative exact xarray reduction: complete.
- Unsafe split core chunks fail loudly by default: complete for operations in `STRICT_CORE_DIMS_BY_OPERATION`.
- Approximate/blockwise mode is explicit: complete by absence; no public blockwise mode is exposed.

### Phase 4: I/O repositioning - not complete

Goal: reposition Wandas persistence around xarray-compatible storage without breaking WDF users.

Required work:

1. Decide whether WDF remains HDF5-specific or becomes a Wandas profile over xarray storage.
2. Add `to_zarr()` / `open_zarr()` if optional dependency policy allows it.
3. Separate storage chunk policy from analysis chunk policy.
4. Document rechunking expectations before signal-processing operations.
5. Preserve existing WDF compatibility or define a deprecation timeline.
6. Add round-trip tests for large chunked datasets.

Exit criteria:

- Users can store large xarray-backed Wandas data without materializing everything eagerly.
- Storage chunks may be optimized for I/O, while analysis chunks are validated/rechunked for signal correctness.
- WDF/NetCDF/Zarr roles are clearly documented.

## Accessor Plan / accessor 方針

Accessor is optional and should follow after Phase 2 stabilizes.

Possible future API:

```python
da.wd.describe()
da.wd.stft()
da.wd.to_frame()
```

Do not implement this before the frame schema and operation semantics are stable.

## Risks / リスク

- Bridge code can become permanent complexity if Phase 2-4 are not completed.
- Strict coordinate validation may reject legitimate xarray workflows until Wandas supports arbitrary coordinate offsets.
- `attrs` can be dropped or changed by xarray operations; Wandas must treat attrs as schema data only at controlled boundaries.
- NetCDF support does not automatically solve large-data storage; Zarr is still needed for chunked cloud/local stores.
- Direct `_data` mutation compatibility makes `_xr` ownership harder to reason about.

## Current State After PR #213 / PR #213 後の状態

Completed:

- Phase 1 bridge is implemented and verified.
- Some Phase 2 scaffolding exists through internal `_xr` sync.
- Phase 3 xarray-aware operation dispatch is implemented for representative strict operations and exact reductions.
- Some Phase 4 scaffolding exists through NetCDF helpers.

Not completed:

- Full xarray-native internal model.
- Full xarray-native operation coverage for every operation.
- Public blockwise/overlap execution modes.
- Zarr support.
- WDF repositioning.
- Accessor API.

## Verification Expectations / 検証方針

For every follow-up phase:

1. Write failing regression tests first.
2. Preserve existing public Wandas API unless a breaking change is explicitly approved.
3. Run focused tests for changed modules.
4. Run full suite before merging.
5. Keep `wandas/xarray_bridge.py` and `wandas/processing/chunk_policy.py` coverage at 100% while they remain small enough for this expectation to be reasonable.

Current PR #213 final verification:

```text
uv run ruff check wandas tests
uv run ty check wandas tests
uv run pytest -q
uv run coverage report -m wandas/xarray_bridge.py wandas/processing/chunk_policy.py
```

Latest recorded result:

```text
1482 passed, 3 skipped, 69 warnings
wandas/xarray_bridge.py: 100%
wandas/processing/chunk_policy.py: 100%
```

## Open Questions / 未決事項

- Should WDF become a Wandas profile over Zarr/NetCDF, or should it remain a legacy HDF5 format?
- Should `attrs` hold full `channel_metadata`, or should complex metadata eventually move to a sidecar `Dataset` variable?
- What is the deprecation policy for direct `_data` mutation?
- Which operations are safe to expose with `mode="blockwise"` after operation-specific boundary tests?
- Should xarray-selected non-zero time origins become representable in Wandas frames?
- Should `.wd` accessor become public API after Phase 2?
