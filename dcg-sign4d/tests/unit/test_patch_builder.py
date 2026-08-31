from pathlib import Path

import pytest

from dcg_sign4d.geometry.patch_builder import build_provisional_smplx_patch_map, write_patch_map
from dcg_sign4d.geometry.patch_map import PatchMap

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
SEGMENTATION = ROOT / "DPoser-X/lib/data/smplx_vert_segmentation.json"
VERTEX_IDS = ROOT / "SMPLer-X/common/utils/smplx/smplx/vertex_ids.py"


@pytest.mark.skipif(not MODEL.is_file(), reason="licensed local SMPL-X asset unavailable")
def test_provisional_patch_builder_is_deterministic_and_complete(tmp_path):
    first = build_provisional_smplx_patch_map(
        model_path=MODEL, segmentation_path=SEGMENTATION, vertex_ids_path=VERTEX_IDS
    )
    second = build_provisional_smplx_patch_map(
        model_path=MODEL, segmentation_path=SEGMENTATION, vertex_ids_path=VERTEX_IDS
    )
    assert first == second
    assert first["development_only"] is True
    assert first["scientific_status"] == "UNFROZEN_AUTHOR_REVIEW_REQUIRED"
    assert len(first["patches"]) == 14
    assert len(first["admissible_edges"]) == 60
    assert len(first["excluded_edges"]) == 31
    output = write_patch_map(first, tmp_path / "map.yaml")
    loaded = PatchMap.load(output)
    assert loaded.content_hash == first["sha256"]
    assert loaded.source_assets is not None
    with pytest.raises(FileExistsError):
        write_patch_map(first, output)
