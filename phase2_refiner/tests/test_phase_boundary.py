import ast
from pathlib import Path


def test_phase2_does_not_import_phase3_posterior():
    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "phase3_posterior" or name.startswith("phase3_posterior.") for name in names):
                violations.append(str(path.relative_to(root)))
    assert not violations, f"Phase 2 imports Phase 3: {sorted(set(violations))}"
