"""Print a case-level routing audit for the deleted v1 monolithic tests."""

from __future__ import annotations

import ast
import subprocess

ROUTES = (
    (("sklearn",), "tests/pipeline/test_sklearn_adapter.py"),
    (("serial", "dict", "json", "params", "importable", "callable"), "tests/pipeline/test_recipe_serialization.py"),
    (("error", "reject", "invalid", "missing", "unknown"), "tests/pipeline/test_recipe_errors.py"),
    (("graph", "binary", "operand", "add_channel", "index", "from_frame"), "tests/pipeline/test_recipe_compiler.py"),
    (("extract", "step_from_graph", "history_value"), "tests/pipeline/test_recipe_codecs.py"),
    (("terminal", "apply", "metadata", "source_time", "lazy"), "tests/pipeline/test_recipe_execution.py"),
)


def main() -> None:
    source = subprocess.run(
        ["git", "show", "b808c8e:tests/pipeline/test_recipe.py"], check=True, capture_output=True, text=True
    ).stdout
    names = [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    for name in names:
        target = "tests/pipeline/test_recipe_contract.py"
        for words, candidate in ROUTES:
            if any(word in name for word in words):
                target = candidate
                break
        print(f"{name}\t{target}")
    print(f"audited_cases\t{len(names)}")


if __name__ == "__main__":
    main()
