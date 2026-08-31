from pathlib import Path

from signeft.io_utils import load_config, resolve_path


def test_refinement_config_has_no_gt_or_evaluator_region_path():
    config = load_config(Path(__file__).parents[1] / "configs" / "ablations" / "c0_a3f.yaml")
    forbidden = [key for key in config["paths"] if "gt" in key.lower() or "evaluator_assets" in key]
    assert forbidden == []
    assert config["method"]["use_gt_in_fit"] is False
    assert config["method"]["temporal_pose_loss"] is False


def test_inherited_paths_resolve_at_their_declaration_file():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "ablations" / "c3_engineering12.yaml")
    assert resolve_path(config, config["paths"]["manifest"]) == root / "manifests" / "splits" / "engineering12.jsonl"
    assert resolve_path(config, config["paths"]["evaluator"]) == (
        root.parent / "data" / "evaluation_from_author" / "evaluate_new_fitting.py"
    )
