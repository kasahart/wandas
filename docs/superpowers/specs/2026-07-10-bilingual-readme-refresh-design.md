# Bilingual README Refresh Design

## Goal

Refresh `README.md` and `README.ja.md` together so first-time users can understand Wandas, run the first example from either a repository checkout or a PyPI installation, and distinguish exploratory normalization from calibrated acoustic analysis.

## Scope

- Use the user-provided Japanese README as the content baseline.
- Produce a natural English counterpart with the same structure, examples, claims, and cautions.
- Keep the existing generated README figures and known-signal example.
- Update README-focused tests to describe and verify the revised contract.
- Do not change runtime APIs or generated figures unless verification reveals a mismatch that cannot be resolved in documentation or tests.

## Content Structure

Both READMEs will follow the same progression:

1. Explain the frame-first value proposition and reviewable analysis context.
2. Put installation and optional extras before executable examples.
3. Start with `describe()` on the bundled sample audio, falling back to its public raw GitHub URL outside a checkout.
4. Explain the method-centered path from inspection to processing and plotting.
5. Validate the workflow with a deterministic two-channel known signal.
6. Show a compact path for users' own recordings.
7. Summarize the top-level API, core objects, suitable use cases, further reading, project status, contribution path, and license.

## Accuracy Contracts

- `wd.read()` is described for registered external formats, URLs, bytes, and file-like inputs; WDF uses `wd.load()`.
- `normalize=True` is limited to listening and shape inspection. SPL and psychoacoustic analysis require correctly calibrated pressure values.
- Optional functionality is labeled with its corresponding extra: `io`, `effects`, `marimo`, `psychoacoustic`, or `ml`.
- Dask-backed laziness is described for frames and folder-backed datasets without implying that conversion to NumPy or tensors stays lazy.
- `remove_dc()` is documented as immutable and connected to `previous` and `operation_history`.
- The English and Japanese examples remain semantically equivalent.

## Verification

README tests will verify:

- all published Python blocks execute in the repository test environment;
- the sample-source expression selects the local file when present and exposes a public fallback URL;
- the known-signal calculation removes DC and produces the documented Welch peaks;
- plot titles, limits, labels, and signal semantics match the narrative;
- installation precedes examples and optional features are labeled;
- both languages retain matching workflow sections and core claims;
- repository links and committed figure files remain valid.

Run the focused README test module, the full docs test directory, Ruff, and the repository type check before completion. Any skipped check must be reported with its reason.

## GitHub Review Outcome

The public raw URL fallback directly addresses the unresolved PR #281 review thread about the repository-relative sample path. Replying to or resolving the thread is outside this edit unless explicitly requested.
