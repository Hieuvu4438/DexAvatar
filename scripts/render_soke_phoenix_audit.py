"""Render SOKE SMPL-X mesh overlays on PHOENIX-2014T video frames."""

import os
import sys
import argparse
import pickle
from pathlib import Path
import cv2
import numpy as np
import torch

os.environ["PYOPENGL_PLATFORM"] = "egl"
import trimesh
import pyrender
from phase2_refiner.render import create_smplx_model


def render_soke_clip(
    model,
    faces,
    device,
    signer_id: str,
    clip_name: str,
    video_path: Path,
    soke_pose_dir: Path,
    out_dir: Path,
    num_sample_frames: int = 6,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_files = sorted(
        list(soke_pose_dir.glob("images*.pkl")),
        key=lambda p: int(p.stem.replace("images", "")),
    )
    if not pose_files:
        print(f"[Warning] No pose files found in {soke_pose_dir}")
        return

    total_frames = len(pose_files)
    indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int)
    selected_pkls = [pose_files[i] for i in indices]

    print(
        f"Processing {clip_name} ({signer_id}): {len(selected_pkls)} frames from {total_frames} total frames..."
    )

    cap = cv2.VideoCapture(str(video_path))
    rendered_panels = []

    for rank, pkl_path in enumerate(selected_pkls):
        frame_number_1based = int(pkl_path.stem.replace("images", ""))
        frame_idx_0based = frame_number_1based - 1

        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_0based)
        ret, frame = cap.read()
        if not ret:
            print(f"  [Warning] Failed to decode frame {frame_idx_0based} from {video_path}")
            continue

        H, W = frame.shape[:2]

        betas = torch.from_numpy(d["smplx_shape"])[None].float().to(device)
        body_pose = torch.from_numpy(d["smplx_body_pose"])[None].float().to(device)
        global_orient = torch.from_numpy(d["smplx_root_pose"])[None].float().to(device)
        left_hand_pose = torch.from_numpy(d["smplx_lhand_pose"])[None].float().to(device)
        right_hand_pose = torch.from_numpy(d["smplx_rhand_pose"])[None].float().to(device)
        jaw_pose = torch.from_numpy(d["smplx_jaw_pose"])[None].float().to(device)
        expression = torch.from_numpy(d["smplx_expr"])[None].float().to(device)

        with torch.no_grad():
            out = model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                expression=expression,
            )
            vertices = out.vertices[0].cpu().numpy()

        cam_trans = d["cam_trans"]
        verts_cam = vertices + cam_trans

        mesh_tri = trimesh.Trimesh(verts_cam, faces, process=False)
        rot_gl = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh_tri.apply_transform(rot_gl)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.15,
            roughnessFactor=0.6,
            alphaMode="BLEND",
            baseColorFactor=(0.25, 0.70, 0.90, 0.8),
        )
        mesh_pyr = pyrender.Mesh.from_trimesh(mesh_tri, material=material, smooth=True)

        scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.08, 1.0], ambient_light=(0.4, 0.4, 0.4))
        scene.add(mesh_pyr, "mesh")

        fx = 5000.0 / 192.0 * W
        fy = 5000.0 / 256.0 * H
        cx = W / 2.0
        cy = H / 2.0

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera)

        light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light1_pose = np.eye(4)
        light1_pose[:3, 3] = [0, 0, 1]
        scene.add(light1, pose=light1_pose)

        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [1, 1, 1]
        scene.add(light2, pose=light2_pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        mask = (depth > 0)[:, :, None]
        mesh_rgb = rgb[:, :, :3]
        alpha = 0.65

        overlay = np.where(mask, (alpha * mesh_rgb + (1.0 - alpha) * frame).astype(np.uint8), frame)
        mesh_view = np.where(mask, mesh_rgb, np.full_like(frame, 20))

        header_h = 24
        header = np.zeros((header_h, W * 3, 3), dtype=np.uint8)
        cv2.putText(
            header,
            f"{signer_id} | {clip_name[:25]} | F{frame_number_1based}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        triptych = np.hstack([frame, overlay, mesh_view])
        composite = np.vstack([header, triptych])

        frame_out_path = out_dir / f"frame_{rank+1:02d}_img{frame_number_1based:04d}.png"
        cv2.imwrite(str(frame_out_path), composite)
        cv2.imwrite(str(out_dir / f"overlay_{rank+1:02d}_img{frame_number_1based:04d}.png"), overlay)

        rendered_panels.append(composite)

    cap.release()

    if len(rendered_panels) >= 4:
        mid = len(rendered_panels) // 2
        row1 = np.hstack(rendered_panels[:mid])
        row2 = np.hstack(rendered_panels[mid:])
        summary_grid = np.vstack([row1, row2])
        cv2.imwrite(str(out_dir / f"summary_grid.png"), summary_grid)

    print(f"Done {clip_name} -> saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/haipd/DexAvatar/data/phonex_extract"),
        help="Target output directory",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=6,
        help="Number of sample frames per clip",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SMPL-X model on {device}...")
    model = create_smplx_model("data/ARCTIC/body_models", device)
    faces = model.faces

    test_clips = [
        ("Signer01", "25October_2010_Monday_tagesschau-14"),
        ("Signer03", "26January_2010_Tuesday_heute-90"),
        ("Signer04", "24September_2010_Friday_tagesschau-203"),
        ("Signer05", "15December_2010_Wednesday_tagesschau-38"),
        ("Signer08", "11August_2010_Wednesday_tagesschau-1"),
    ]

    base_out_dir = args.output_dir.resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (signer, clip) in enumerate(test_clips, 1):
        vid_path = Path(f"/home/dongvk/datasets/phoenix14T/videos_phoenix/videos/train/{clip}.mp4")
        pose_dir = Path(f"data/SignAvatar_SOKE/extracted/soke_phoenix_frame_poses/train/{clip}")
        clip_out_dir = base_out_dir / f"clip_{idx:02d}_{signer}"
        render_soke_clip(
            model=model,
            faces=faces,
            device=device,
            signer_id=signer,
            clip_name=clip,
            video_path=vid_path,
            soke_pose_dir=pose_dir,
            out_dir=clip_out_dir,
            num_sample_frames=args.frames_per_clip,
        )

    print("All 5 videos rendered successfully!")
    print(f"Results are available at: {base_out_dir}")


if __name__ == "__main__":
    main()
