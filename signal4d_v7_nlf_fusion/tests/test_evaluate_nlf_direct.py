import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "evaluate_nlf_direct.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nlf_direct", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_translation_relative_errors_remove_only_translation():
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    translated = target + np.asarray([9.0, -4.0, 2.0])
    assert np.allclose(MODULE.translation_relative_errors(translated, target), 0.0)


def test_translation_relative_errors_do_not_remove_rotation():
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotated = np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert MODULE.translation_relative_errors(rotated, target).mean() > 0.0
