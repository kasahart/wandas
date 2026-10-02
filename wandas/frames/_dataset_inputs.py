"""Internal normalization of explicit local files and recording catalogs."""

from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wandas.utils.optional_imports import require_pandas

if TYPE_CHECKING:
    import pandas as pd

_RESERVED_METADATA_KEYS = frozenset({"_source_file"})
_InputItems = list[tuple[Path, Mapping[str, object]]]


def _anchored_path(path: Path, base: Path) -> Path:
    # Preserve .. for the filesystem to interpret after resolving symlinks.
    if path.is_absolute():
        return path
    if path.drive:
        # Remove the relative drive before joining (including on Python 3.14).
        if path.drive.casefold() != base.drive.casefold():
            base = Path(os.path.abspath(path.drive + "."))
        path = path.relative_to(path.drive)
    return base / path


def _base_directory(base_dir: str | Path | None) -> Path:
    return _anchored_path(Path(base_dir), Path.cwd()) if base_dir is not None else Path.cwd()


def _local_path(value: object, base: Path, *, location: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{location}: expected a local str or Path, not {type(value).__name__}")
    if not str(value).strip() or "\0" in str(value):
        raise ValueError(f"{location}: path must be nonempty and contain no null characters")
    path = Path(value)
    is_windows_drive = bool(re.fullmatch(r"[A-Za-z]:", path.drive))
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", str(value)) and not is_windows_drive:
        raise ValueError(f"{location}: collection inputs require local paths; use read() for individual URLs")
    return _anchored_path(path, base)


def _file_items(paths: Iterable[str | Path], base_dir: str | Path | None) -> _InputItems:
    if isinstance(paths, (str, Path, bytes, Mapping)):
        raise TypeError("paths must be a finite iterable of local paths, not a single path or mapping")
    base = _base_directory(base_dir)
    return [(_local_path(value, base, location=f"item {index}"), {}) for index, value in enumerate(paths)]


def _validate_columns(columns: Sequence[object]) -> None:
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("Column names must be nonempty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate column names are not supported")


def _scalar_metadata(value: object, missing_values: tuple[object, object], *, location: str) -> object:
    if value is None or value is missing_values[0] or value is missing_values[1]:
        return None
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError(f"{location}: infinite metadata is unsupported; convert or exclude this column")
        return value
    raise TypeError(f"{location}: basic scalar metadata required; convert or exclude this column")


def _table_items(
    table: pd.DataFrame | str | Path,
    path_column: str,
    base_dir: str | Path | None,
    metadata_columns: Sequence[str] | None,
) -> _InputItems:
    if not isinstance(path_column, str) or not path_column:
        raise ValueError("path_column must explicitly name a nonempty column")
    is_csv = isinstance(table, (str, Path))
    missing_values: tuple[object, object] = (None, None)
    if is_csv:
        csv_path = _local_path(table, _base_directory(None), location="CSV catalog")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                columns = next(reader, [])
                _validate_columns(columns)
                # Like DictReader, ignore entirely blank physical rows.
                values = [row for row in reader if row]
            except csv.Error as exc:
                raise ValueError(f"Malformed CSV catalog near line {reader.line_num}: {exc}") from exc
        base = _base_directory(base_dir) if base_dir is not None else csv_path.parent
    else:
        pandas = require_pandas("Dataset table input")
        missing_values = (pandas.NA, pandas.NaT)
        if not isinstance(table, pandas.DataFrame):
            raise TypeError("table must be a pandas DataFrame or a local CSV catalog path")
        columns = list(table.columns)
        _validate_columns(columns)
        values = list(table.itertuples(index=False, name=None))
        base = _base_directory(base_dir)
    if path_column not in columns:
        raise ValueError(f"Unknown path_column: {path_column}")
    if isinstance(metadata_columns, (str, bytes)):
        raise TypeError("metadata_columns must be a sequence of column names, not a string")
    selected = (
        list(metadata_columns) if metadata_columns is not None else [name for name in columns if name != path_column]
    )
    _validate_columns(selected)
    if path_column in selected:
        raise ValueError("path_column cannot also be a metadata column")
    if set(selected).difference(columns):
        raise ValueError("Unknown metadata column(s): " + ", ".join(sorted(set(selected).difference(columns))))
    reserved = _RESERVED_METADATA_KEYS.intersection(selected)
    if reserved:
        raise ValueError("Reserved metadata column(s): " + ", ".join(sorted(reserved)) + "; rename or exclude them")
    path_index = columns.index(path_column)
    metadata_indices = [(name, columns.index(name)) for name in selected]
    items: _InputItems = []
    for index, row in enumerate(values):
        if len(row) != len(columns):
            raise ValueError(f"row {index}: CSV cell count does not match its header")
        source = _local_path(row[path_index], base, location=f"row {index}.{path_column}")
        metadata = {
            name: row[position]
            if is_csv
            else _scalar_metadata(row[position], missing_values, location=f"row {index}.{name}")
            for name, position in metadata_indices
        }
        items.append((source, metadata))
    return items
