from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..data.manifest import load_manifest
from ..utils.hashing import sha256_file


def _composite_rgba(
    background: np.ndarray, rendered_rgba: np.ndarray, mesh_opacity: float
) -> np.ndarray:
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("background must be HxWx3 RGB")
    if rendered_rgba.shape != (*background.shape[:2], 4):
        raise ValueError("rendered image shape does not match background")
    if not 0.0 <= mesh_opacity <= 1.0:
        raise ValueError("mesh_opacity must be within [0, 1]")
    alpha = rendered_rgba[..., 3:4].astype(np.float32) / 255.0
    alpha *= mesh_opacity
    foreground = rendered_rgba[..., :3].astype(np.float32)
    result = foreground * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _render_one(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    from PIL import Image

    image_path = Path(task["image_path"])
    mesh_path = Path(task["mesh_path"])
    camera_path = Path(task["camera_path"])
    output_path = Path(task["output_path"])

    background = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = background.shape[:2]
    with camera_path.open("rb") as handle:
        camera_parameters = pickle.load(handle, encoding="latin1")
    focal = np.asarray(camera_parameters["focal"], dtype=np.float64).reshape(2)
    principal = np.asarray(camera_parameters["princpt"], dtype=np.float64).reshape(2)

    loaded = trimesh.load(mesh_path, process=False, maintain_order=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected a triangular mesh: {mesh_path}")
    if loaded.vertices.shape != (10475, 3) or loaded.faces.shape != (20908, 3):
        raise ValueError(f"unexpected SMPL-X topology: {mesh_path}")

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5)
    )
    camera_pose = np.eye(4)
    camera = pyrender.IntrinsicsCamera(
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(principal[0]),
        cy=float(principal[1]),
    )
    scene.add(camera, pose=camera_pose)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode="OPAQUE",
        baseColorFactor=(0.5, 0.5, 0.7, 1.0),
    )
    scene.add(pyrender.Mesh.from_trimesh(loaded, material=material), "mesh")
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    for translation in ([0.0, -1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]):
        light_pose = np.eye(4)
        light_pose[:3, 3] = translation
        scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    composited = _composite_rgba(background, color, float(task["mesh_opacity"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    Image.fromarray(composited, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, output_path)
    return {
        "clip_id": task["clip_id"],
        "frame_id": task["frame_id"],
        "image_relpath": task["image_relpath"],
        "image_sha256": sha256_file(image_path),
        "camera_sha256": sha256_file(camera_path),
        "mesh_relpath": task["mesh_relpath"],
        "mesh_sha256": sha256_file(mesh_path),
        "overlay_relpath": task["overlay_relpath"],
        "overlay_sha256": sha256_file(output_path),
        "width": width,
        "height": height,
    }


def run(
    manifest_path: str,
    mesh_root: str,
    image_root: str,
    camera_root: str,
    output_root: str,
    method_name: str,
    workers: int = 4,
    mesh_opacity: float = 0.9,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 <= mesh_opacity <= 1.0:
        raise ValueError("mesh_opacity must be within [0, 1]")
    manifest = load_manifest(manifest_path)
    meshes = Path(mesh_root)
    images = Path(image_root)
    cameras = Path(camera_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reconstruction root: {output}")

    tasks: list[dict[str, Any]] = []
    for item in manifest:
        if len(item.image_relpaths) != len(item.frame_ids):
            raise ValueError(f"image/frame count mismatch for {item.clip_id}")
        for frame_id, image_relpath in zip(item.frame_ids, item.image_relpaths, strict=True):
            filename = f"low_{frame_id}.obj"
            mesh_path = meshes / item.clip_id / "smplifyx" / "meshes" / filename
            image_path = images / image_relpath
            camera_path = (
                cameras / item.clip_id / "smplerx" / "smplx" / f"low_{frame_id}.pkl"
            )
            overlay_relpath = (
                Path(item.clip_id) / "smplifyx" / "images" / f"low_{frame_id}.png"
            )
            for required in (mesh_path, image_path, camera_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            tasks.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "image_path": str(image_path),
                    "image_relpath": str(image_path.relative_to(images)),
                    "mesh_path": str(mesh_path),
                    "mesh_relpath": str(mesh_path.relative_to(meshes)),
                    "camera_path": str(camera_path),
                    "output_path": str(output / overlay_relpath),
                    "overlay_relpath": str(overlay_relpath),
                    "mesh_opacity": mesh_opacity,
                }
            )

    output.mkdir(parents=True)
    incomplete = output / ".render_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_render_one, tasks, chunksize=1))

    for row in rows:
        source = (meshes / row["mesh_relpath"]).resolve()
        target = output / Path(row["mesh_relpath"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)

    report = {
        "schema_version": "1.0",
        "method_name": method_name,
        "format": "dexavatar_rgb_mesh_overlay",
        "layout": "<sign>/smplifyx/{images,meshes}/low_<frame>.{png,obj}",
        "renderer": "pyrender_intrinsics_camera",
        "camera_source": str(cameras),
        "image_source": str(images),
        "mesh_source": str(meshes),
        "mesh_color_rgba": [0.5, 0.5, 0.7, 1.0],
        "mesh_opacity": mesh_opacity,
        "manifest_sha256": sha256_file(manifest_path),
        "clips": len(manifest),
        "frames": len(rows),
        "files": rows,
    }
    (output / "render_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report
