"""Diagnostic: measure INIT quality (no fitting).

For each frame, render the NLF+WiLoR init and the SMPLer-X init to SMPLX meshes
(pure forward pass) and measure TR-V2V vs the GT mesh for UBody(-face)/LHand/RHand.

This directly tests whether the NLF+WiLoR init is worse than the SMPLer-X init
(i.e. whether the "better expert" belief holds at the init stage), independent of
the fitting config. Runs on CPU to avoid contending with the GPU fitting job.
"""
from __future__ import annotations
import os, glob, pickle
import numpy as np
import torch
import smplx

DEX = "/home/haipd/DexAvatar"
SMPLX_PKL = f"{DEX}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl"
GT_ROOT = f"{DEX}/data/smplx_gt"
MASK_DIR = f"{DEX}/dexavatar_fitting/assets"
DATA_ROOT = f"{DEX}/data/evaluation_from_author/data/data"
SIGNS = ["Glas", "Tisch", "Ablehnen", "Muell"]
MM = 1000.0

def load_obj_verts(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line[:2] == "v ":
                pts.append(line[2:].rstrip())
    return np.fromstring(" ".join(pts), sep=" ", dtype=np.float64).reshape(-1, 3)

def tr_v2v(p, g):
    p = p - p.mean(0); g = g - g.mean(0)
    return float(np.linalg.norm(p - g, axis=1).mean() * MM)

def pa_v2v(p, g):
    """Procrustes (rotation+translation, no scale) — isolates articulation."""
    mu_p, mu_g = p.mean(0), g.mean(0)
    P, G = p - mu_p, g - mu_g
    U, _, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pa = (R @ P.T).T
    return float(np.linalg.norm(Pa - G, axis=1).mean() * MM)

def make_model():
    return smplx.create(
        model_path=SMPLX_PKL, ext="pkl", gender="neutral",
        use_face_contour=True, flat_hand_mean=True, use_pca=False, batch_size=1,
    )

def fwd(model, p):
    kw = dict(
        global_orient=torch.tensor(p["global_orient"], dtype=torch.float32)[None],
        body_pose=torch.tensor(p["body_pose"], dtype=torch.float32)[None],
        left_hand_pose=torch.tensor(p["left_hand_pose"], dtype=torch.float32)[None],
        right_hand_pose=torch.tensor(p["right_hand_pose"], dtype=torch.float32)[None],
        betas=torch.tensor(p["betas"], dtype=torch.float32)[None],
        transl=torch.tensor(p["transl"], dtype=torch.float32)[None],
    )
    with torch.no_grad():
        out = model(**{k: v for k, v in kw.items() if v.shape[-1] > 0 or k == "transl"})
    return out.vertices[0].numpy().astype(np.float64)

def main():
    mano = pickle.load(open(f"{DATA_ROOT}/MANO_SMPLX_vertex_ids.pkl", "rb"))
    lhand = np.asarray(mano["left_hand"]); rhand = np.asarray(mano["right_hand"])
    ubody = np.load(f"{MASK_DIR}/smplx_upper_body_minus_face_vidx.npy")
    model = make_model()

    acc = {"nlf": {"ub": [], "lh": [], "rh": [], "ub_pa": [], "lh_pa": [], "rh_pa": []},
           "smp": {"ub": [], "lh": [], "rh": [], "ub_pa": [], "lh_pa": [], "rh_pa": []}}
    for sign in SIGNS:
        gt_dir = f"{GT_ROOT}/{sign}"
        nlf_dir = f"{DEX}/outputs/method_nlf_wilor/{sign}/nlf/smplx"
        smp_dir = f"{DEX}/outputs/shared/{sign}/smplerx/smplx"
        if not (os.path.isdir(gt_dir) and os.path.isdir(nlf_dir) and os.path.isdir(smp_dir)):
            print(f"[skip] {sign}: missing dirs"); continue
        n_nlf = 0
        for np_path in sorted(glob.glob(f"{nlf_dir}/low_*.pkl")):
            base = os.path.basename(np_path)[:-4]              # low_<N>
            n = int(base.split("_")[-1])
            gt_path = f"{gt_dir}/{2*n:05d}.obj"
            sp_path = f"{smp_dir}/{base}.pkl"
            if not (os.path.exists(gt_path) and os.path.exists(sp_path)):
                continue
            gt = load_obj_verts(gt_path)
            nlf_p = pickle.load(open(np_path, "rb"))
            smp_p = pickle.load(open(sp_path, "rb"))
            try:
                v_nlf = fwd(model, nlf_p); v_smp = fwd(model, smp_p)
            except Exception as e:
                print(f"  [fwd fail] {sign}/{base}: {e}"); continue
            for tag, v in (("nlf", v_nlf), ("smp", v_smp)):
                acc[tag]["ub"].append(tr_v2v(v[ubody], gt[ubody]))
                acc[tag]["lh"].append(tr_v2v(v[lhand], gt[lhand]))
                acc[tag]["rh"].append(tr_v2v(v[rhand], gt[rhand]))
                acc[tag]["ub_pa"].append(pa_v2v(v[ubody], gt[ubody]))
                acc[tag]["lh_pa"].append(pa_v2v(v[lhand], gt[lhand]))
                acc[tag]["rh_pa"].append(pa_v2v(v[rhand], gt[rhand]))
            n_nlf += 1
        print(f"[{sign}] {n_nlf} init frames compared")

    print("\n=== INIT quality TR-V2V (mm), mean over all compared frames ===")
    print(f"{'src':<6}{'UBody-F':>10}{'LHand':>10}{'RHand':>10}")
    for tag in ("smp", "nlf"):
        r = acc[tag]
        print(f"{tag:<6}{np.mean(r['ub']):>10.2f}{np.mean(r['lh']):>10.2f}{np.mean(r['rh']):>10.2f}  (n={len(r['ub'])})")
    print("\n=== INIT quality PA-V2V (mm, Procrustes — isolates articulation) ===")
    print(f"{'src':<6}{'UBody-F':>10}{'LHand':>10}{'RHand':>10}")
    for tag in ("smp", "nlf"):
        r = acc[tag]
        print(f"{tag:<6}{np.mean(r['ub_pa']):>10.2f}{np.mean(r['lh_pa']):>10.2f}{np.mean(r['rh_pa']):>10.2f}")
    print("\nSMPLer-X = baseline init ; NLF = nlf_wilor init (body=NLF, hands=WiLoR)")

if __name__ == "__main__":
    main()
