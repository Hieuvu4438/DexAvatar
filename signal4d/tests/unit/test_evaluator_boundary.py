import ast
from pathlib import Path

import signal4d.evaluation.sgnify as evaluator


def test_sgnify_evaluator_does_not_import_method_implementation() -> None:
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    relative_imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level
    }
    assert not any(
        module and ("optimization" in module or "models" in module) for module in relative_imports
    )
