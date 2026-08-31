from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "signpccx.teacher.v1"

REQUIRED = {
    "K_full": (3, 3),
    "crop_to_full": (3, 3),
    "smplx_vertices_cam": (10475, 3),
    "smplx_body_pose_aa": (21, 3),
    "smplx_left_hand_pose_aa": (15, 3),
    "smplx_right_hand_pose_aa": (15, 3),
    "smplx_global_orient_aa": (1, 3),
    "smplx_betas": (10,),
    "smplx_transl": (3,),
    "left_mano_vertices_cam": (778, 3),
    "right_mano_vertices_cam": (778, 3),
    "left_mano_joints_cam": (21, 3),
    "right_mano_joints_cam": (21, 3),
    "keypoints_2d_full": (133, 3),
    "left_bbox_full_xyxy": (4,),
    "right_bbox_full_xyxy": (4,),
}


@dataclass(frozen=True)
class TeacherMeta:
    schema_version: str
    sign: str
    frame_id: int
    image_sha256: str
    repo_commit: str
    checkpoint_sha256: str
    coord_frame: str
    unit_3d: str
    image_width: int
    image_height: int


def validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED).difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        for key, shape in REQUIRED.items():
            value = archive[key]
            if value.shape != shape:
                raise ValueError(f"{path}:{key} {value.shape} != {shape}")
            if value.dtype.kind == "O":
                raise TypeError(f"{path}:{key} has object dtype")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{path}:{key} contains NaN/Inf")


def validate_sidecar(path: Path) -> TeacherMeta:
    meta = TeacherMeta(**json.loads(path.read_text(encoding="utf-8")))
    if meta.schema_version != SCHEMA_VERSION:
        raise ValueError(meta.schema_version)
    if meta.coord_frame != "opencv_camera_xright_ydown_zforward":
        raise ValueError(meta.coord_frame)
    if meta.unit_3d != "meter":
        raise ValueError(meta.unit_3d)
    return meta

