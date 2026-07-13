"""Report Recipe v2 production PLOC/SLOC and structural guardrails."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FILES = sorted(path for path in (ROOT / "wandas" / "pipeline").glob("*.py") if path.name != "sklearn.py")
FILES = [*PIPELINE_FILES, ROOT / "wandas" / "processing" / "semantic.py"]
CORE_PLOC_LIMIT = 1_753


def counts(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ploc = len(lines)
    ignored: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline):
        if token.type in {tokenize.COMMENT, tokenize.STRING}:
            ignored.update(range(token.start[0], token.end[0] + 1))
    sloc = sum(bool(line.strip()) and number not in ignored for number, line in enumerate(lines, 1))
    return ploc, sloc


def private_test_imports() -> int:
    total = 0
    for path in (ROOT / "tests" / "pipeline").glob("test_recipe*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                total += sum(alias.name.startswith("_") for alias in node.names)
    return total


def main() -> None:
    total_ploc = total_sloc = 0
    for path in FILES:
        ploc, sloc = counts(path)
        total_ploc += ploc
        total_sloc += sloc
        print(f"{path.relative_to(ROOT)}: PLOC={ploc} SLOC={sloc}")
    pipeline = "\n".join(path.read_text(encoding="utf-8") for path in FILES if "pipeline" in path.parts)
    print(f"Recipe responsibility total: PLOC={total_ploc} SLOC={total_sloc}")
    sklearn_ploc, sklearn_sloc = counts(ROOT / "wandas" / "pipeline" / "sklearn.py")
    print(f"Optional sklearn adapter: PLOC={sklearn_ploc} SLOC={sklearn_sloc}")
    print(f"operation_graph extraction references: {pipeline.count('operation_graph')}")
    print(f"public contract private imports: {private_test_imports()}")
    if total_ploc > CORE_PLOC_LIMIT:
        raise SystemExit(f"Recipe core PLOC exceeds {CORE_PLOC_LIMIT}: {total_ploc}")


if __name__ == "__main__":
    main()
