# Pull request review instructions

Used by `.github/workflows/claude-code-review.yml` and by "@claude review"
(`.github/workflows/claude.yml`). It lives under `.claude/` because
claude-code-action restores that directory from the base branch on pull
requests, so a PR cannot rewrite the rules it is reviewed by.

Review the pull request as if no other reviewer will look for bugs after you.
Read AGENTS.md first and judge the change against its conventions and
guardrails. Write every comment in Japanese.

## How to review

1. Read the whole diff with `gh pr diff`, then read the code each changed
   function is called from and calls into. Do not review hunks in isolation.
2. For every changed behaviour, try to break it. Walk the checklist below and,
   for each item, either name a concrete failing input or state, or move on.
3. When a claim depends on runtime behaviour (numerics, dtype, shapes, laziness,
   metadata, warnings), verify it: write a short script under
   `review-scratch/` (gitignored) and run it with
   `uv run --no-sync python`, or run the relevant tests with
   `uv run --no-sync pytest`. Say in the finding what you ran and what it
   printed.
4. Dependencies may have been installed from the default branch. If the PR
   changes dependency files, say which results could differ with the PR's
   own dependencies.
5. Drop any finding you cannot back with a concrete scenario.

## Checklist

- Parallel entry points: a new parameter or option added to one API but not
  forwarded by the other routes to the same operation: `ChannelFrame` vs
  `ChannelFrameDataset` and other wrappers or mixins, `SpectrogramFrame` vs
  `ChannelFrame` paths, Recipe recording and replay, WDF save/load state, and
  Recipe op versions.
- Numerics: float32/complex64 underflow or overflow in products and divisions,
  NaN or inf from 0/0, padded or partial frames treated as data, off-by-one
  frame or sample origins, axis lengths that do not match `n_fft` accepted
  silently, units or sampling rates combined as if comparable. Compare against
  SciPy where an equivalent exists.
- Metadata and lineage after slicing or derivation: `source_time_offset`,
  `frame_time_origin`, units, labels and `operation_history` copied from the
  wrong source; Dask laziness broken by an early compute.
- Edge inputs: empty, very short, single-channel, zero or negative spans,
  extreme magnitudes, unsupported codecs or backends, invalid arguments that
  crash instead of raising a clear error.
- Performance traps: per-element Python loops or quadratic-or-worse lookups
  over channels, pairs or frames while building a graph.
- Robustness: ignored return values (short writes), `warnings.warn`
  `stacklevel` pointing at internal frames, paths and URLs not validated.
- Docs, docstrings and comments that state something false about the code or
  a dependency (for example SciPy defaults).
- Tests that pass without exercising the new behaviour. Changes to existing
  test expectations or tolerances need the maintainer's approval; flag them.

## Severity

- **blocking**: any defect you can describe with a concrete input or state
  that gives wrong output, a crash, a hang, data loss, a security gap, or a
  false statement in public docs, even if the scenario is uncommon. If you
  reproduced it, it is blocking. Do not downgrade a reproducible defect
  because it is rare or the fix is small.
- **optional**: style, naming, refactoring, and concerns without a concrete
  scenario.

For each finding give the file and line, the severity, the failing scenario,
what you ran to confirm it, and the fix.

## Output

Post findings tied to lines with
`mcp__github_inline_comment__create_inline_comment` (with `confirmed: true`)
and one overall summary with `gh pr comment`. The summary lists blocking
findings first. If nothing is blocking, it says which checklist areas you
verified and how. Only post GitHub comments; do not return the review as a
message.
