"""Apply one Recipe to many recordings, recording each failure and going on."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Iterable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from wandas.utils.optional_imports import require_pandas

if TYPE_CHECKING:
    import pandas as pd

    from wandas.pipeline.model import RecipePlan
    from wandas.pipeline.registry import RecipeRegistry

RESULT_COLUMNS = ("path", "status", "output", "error", "seconds")


@dataclass(frozen=True)
class _FileResult:
    path: str
    status: Literal["ok", "error"]
    output: str
    error: str
    seconds: float
    exception: BaseException | None


def _output_paths(paths: list[Path], out_dir: Path, root: Path | None) -> list[Path]:
    """Map each recording to ``out_dir/<relative path>.wdf``, keeping its extension."""
    if root is None:
        relatives = [Path(path.name) for path in paths]
    else:
        relatives = []
        for path in paths:
            try:
                relative = path.resolve().relative_to(root.resolve())
            except ValueError:
                relative = Path()
            if relative == Path():
                raise ValueError(f"Recording is not inside root\n  Path: {str(path)!r}\n  Root: {str(root)!r}")
            relatives.append(relative)
    outputs = [out_dir / f"{relative}.wdf" for relative in relatives]
    seen: dict[Path, Path] = {}
    for path, output in zip(paths, outputs):
        if output in seen:
            if seen[output].resolve() == path.resolve():
                raise ValueError(f"Duplicate recording path\n  Path: {str(path)!r}")
            hint = "; pass root to keep their folders" if root is None else ""
            raise ValueError(
                f"Two recordings would be saved to the same output{hint}\n"
                f"  Paths: {[str(seen[output]), str(path)]!r}\n"
                f"  Output: {str(output)!r}"
            )
        seen[output] = path
    return outputs


def _summary(exc: BaseException) -> str:
    """Return one line naming the error; an operation failure names its operation and cause."""
    from wandas.pipeline.errors import RecipeExecutionError

    prefix = ""
    if isinstance(exc, RecipeExecutionError) and exc.cause is not None:
        prefix = f"{exc.operation}: "
        exc = exc.cause
    lines = str(exc).splitlines()
    return f"{prefix}{type(exc).__name__}: {lines[0]}" if lines else f"{prefix}{type(exc).__name__}"


def _apply_one(
    plan: RecipePlan,
    path: Path,
    output: Path,
    overwrite: bool,
    registry: RecipeRegistry | None,
) -> _FileResult:
    """Read, apply, and save one recording; return its failure instead of raising it."""
    import wandas

    started = time.perf_counter()
    try:
        result = plan.apply(wandas.read(path), registry=registry)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - one recording's failure is recorded, the rest go on.
        return _FileResult(str(path), "error", "", _summary(exc), time.perf_counter() - started, exc)
    return _FileResult(str(path), "ok", str(output), "", time.perf_counter() - started, None)


def _apply_one_in_worker(
    plan: RecipePlan,
    path: Path,
    output: Path,
    overwrite: bool,
    keep_exception: bool,
) -> _FileResult:
    """Run :func:`_apply_one` in a worker; only a picklable exception is sent back."""
    import pickle

    outcome = _apply_one(plan, path, output, overwrite, None)
    if outcome.exception is None:
        return outcome
    exception: BaseException | None = None
    if keep_exception:
        try:
            # Some exceptions pickle but cannot be rebuilt, such as one whose __init__ takes
            # other arguments than its args.
            pickle.loads(pickle.dumps(outcome.exception))  # noqa: S301 - our own bytes.
            exception = outcome.exception
        except Exception:  # noqa: BLE001 - some exceptions hold values that cannot cross processes.
            exception = RuntimeError(outcome.error)
    return _FileResult(outcome.path, outcome.status, outcome.output, outcome.error, outcome.seconds, exception)


def apply_files(
    plan: RecipePlan,
    paths: Iterable[str | Path],
    out_dir: str | Path,
    *,
    root: str | Path | None = None,
    jobs: int = 1,
    on_error: Literal["continue", "raise"] = "continue",
    overwrite: bool = False,
    registry: RecipeRegistry | None = None,
) -> pd.DataFrame:
    """Implement :meth:`RecipePlan.apply_files`."""
    from wandas.pipeline.errors import RecipeExecutionError

    if len(plan.inputs) != 1:
        raise RecipeExecutionError(
            "apply_files requires a Recipe with one input\n"
            f"  Inputs: {[recipe_input.name for recipe_input in plan.inputs]!r}"
        )
    if on_error not in ("continue", "raise"):
        raise ValueError(f"on_error must be 'continue' or 'raise'\n  Got: {on_error!r}")
    if type(jobs) is not int or jobs < 1:
        raise ValueError(f"jobs must be a positive integer\n  Got: {jobs!r}")
    if jobs > 1 and registry is not None:
        raise ValueError("A custom registry cannot be sent to worker processes; use jobs=1 with registry")

    pd = require_pandas("RecipePlan.apply_files")
    recordings = [Path(path) for path in paths]
    outputs = _output_paths(recordings, Path(out_dir), None if root is None else Path(root))

    results: list[_FileResult] = []
    if jobs == 1:
        for path, output in zip(recordings, outputs):
            outcome = _apply_one(plan, path, output, overwrite, registry)
            if outcome.exception is not None and on_error == "raise":
                raise outcome.exception
            results.append(outcome)
    else:
        # Forking a process whose Dask threads already ran can deadlock, so start clean workers.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures: list[Future[_FileResult]] = [
                pool.submit(_apply_one_in_worker, plan, path, output, overwrite, on_error == "raise")
                for path, output in zip(recordings, outputs)
            ]
            for path, future in zip(recordings, futures):
                try:
                    outcome = future.result()
                except BrokenProcessPool as exc:
                    # A worker died (for example killed for memory); its recording and the
                    # ones not finished yet fail, and those already finished keep their result.
                    outcome = _FileResult(str(path), "error", "", _summary(exc), float("nan"), exc)
                if outcome.status == "error" and on_error == "raise":
                    pool.shutdown(cancel_futures=True)
                    assert outcome.exception is not None
                    raise outcome.exception
                results.append(outcome)

    rows: list[dict[str, Any]] = [
        {column: getattr(outcome, column) for column in RESULT_COLUMNS} for outcome in results
    ]
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
