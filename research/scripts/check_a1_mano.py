"""CPU deployment check and full raw OakInk example decoding; not a physical task qualification."""
import ast
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from a1_data_sources import safe_pickle
from a1_mano import ManoDecoder, MODEL_SHA256, asset_root, quaternion_matrix

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/a1_full_20261028"
REFERENCE_SHA = "0a447b1cd4ddf889993b7e9d892cd003d752354e52d37a58295ca0616d941615"


def reference_skinning():
    path = ROOT / "references/manotorch_decoder/manotorch/manolayer.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA
    # Run only the reviewed upstream pure tensor skinning function; no legacy
    # pickle loader, constructor, or Chumpy dependencies are imported.
    tree = ast.parse(path.read_text())
    pad = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "th_with_zeros")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ManoLayer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "skinning_layer")
    ns = {"torch": torch, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[pad, fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["skinning_layer"]


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    source = next((ROOT / "data/raw/oakink2/anno_preview").glob("*.pkl"))
    raw = safe_pickle(source)
    frames = sorted(raw["raw_mano"])
    ref = reference_skinning()
    rows = {}
    processed = {"mocap_frame_id": np.array(frames)}
    for side, prefix in [("left", "lh"), ("right", "rh")]:
        decoder = ManoDecoder(side)
        q = torch.cat([raw["raw_mano"][f][prefix + "__pose_coeffs"] for f in frames]).float()
        shape = torch.cat([raw["raw_mano"][f][prefix + "__betas"] for f in frames]).float()
        tsl = torch.cat([raw["raw_mano"][f][prefix + "__tsl"] for f in frames]).float()
        all_joints = []
        vertex_finite = True
        max_radius = 0.
        wrist_error = 0.
        with torch.no_grad():
            for j in range(0, len(frames), 64):
                out = decoder(q[j:j+64], shape[j:j+64], tsl[j:j+64])
                vertex_finite &= bool(torch.isfinite(out["vertices"]).all())
                assert torch.isfinite(out["joints"]).all()
                max_radius = max(max_radius, float(torch.linalg.vector_norm(out["vertices"] - tsl[j:j+64, None], dim=-1).max()))
                wrist_error = max(wrist_error, float((out["joints"][:, 0] - tsl[j:j+64]).abs().max()))
                all_joints.append(out["joints"].numpy())
            picks = np.linspace(0, len(frames)-1, 16, dtype=int)
            actual = decoder(q[picks], shape[picks], tsl[picks])
            # Rotation conversion checked independently against SciPy.
            rot_np = Rotation.from_quat(q[picks].numpy().reshape(-1, 4)[:, [1,2,3,0]]).as_matrix().reshape(-1,16,3,3)
            rot_err = float(np.max(np.abs(quaternion_matrix(q[picks]).numpy() - rot_np)))
            upstream = SimpleNamespace(side=side, center_idx=0,
                th_shapedirs=decoder.shapedirs, th_posedirs=decoder.posedirs,
                th_v_template=decoder.v_template[None], th_J_regressor=decoder.J_regressor,
                th_weights=decoder.weights)
            expected = ref(upstream, torch.tensor(rot_np, dtype=torch.float32), shape[picks])
            v_err = float((actual["vertices"] - expected["verts"] - tsl[picks, None]).abs().max())
            j_err = float((actual["joints"] - expected["joints"] - tsl[picks, None]).abs().max())
            identity = torch.zeros(1, 16, 4); identity[..., 0] = 1
            neutral = decoder(identity, torch.zeros(1, 10), torch.zeros(1, 3))
            neutral_expected = decoder.v_template - (decoder.J_regressor @ decoder.v_template)[0]
            neutral_err = float((neutral["vertices"][0] - neutral_expected).abs().max())
        assert vertex_finite and wrist_error < 1e-6
        assert max(v_err, j_err, neutral_err, rot_err) < 2e-6
        q_grad = q[picks[:2]].clone().requires_grad_()
        loss = decoder(q_grad, shape[picks[:2]], tsl[picks[:2]])["joints"].square().mean()
        loss.backward()
        assert q_grad.grad is not None and torch.isfinite(q_grad.grad).all()
        processed[prefix + "_joints_world_m"] = np.concatenate(all_joints)
        rows[side] = {"frames": len(frames), "vertices_per_frame": 778, "joints_per_frame": 21,
            "all_vertices_and_joints_finite": vertex_finite, "wrist_translation_max_error_m": wrist_error,
            "max_vertex_distance_from_wrist_m": max_radius, "upstream_vertex_max_error_m": v_err,
            "upstream_joint_max_error_m": j_err, "neutral_vertex_max_error_m": neutral_err,
            "rotation_scipy_max_abs_error": rot_err, "pose_gradient_finite": True,
            "model_sha256": MODEL_SHA256[side]}
    out = ROOT / "data/a1_sources/oakink2/decoded_preview" / source.stem
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mano_joints_raw.npz", **processed)
    report = {"status": "PASSED_CPU_MODEL_DEPLOYMENT_AND_RAW_DECODE", "asset_root": str(asset_root()),
        "source_annotation": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "upstream_oakink_commit": "502a02809b50b7f5d0037f92a1ccd2a479051ad1",
        "upstream_manotorch_commit": "933be97d6fa05c729656669640084251ce644b0a",
        "upstream_manolayer_sha256": REFERENCE_SHA, "upstream_comparison_frames_per_hand": 16,
        "convention": "metres; original OakInk world coordinates; wxyz; flat full quaternion rotations; center_idx=0 then add tsl; SNAP joints",
        "decoded_joints": str((out / "mano_joints_raw.npz").relative_to(ROOT)),
        "limitations": ["Raw frame IDs retained; no unverified timestamp assumptions", "Not yet resampled or Wuji-retargeted", "Does not qualify a physical task or pass formal A1/G1"],
        "checks": rows, "gpu_process_seconds": 0, "cpu_wall_seconds": time.monotonic() - start}
    (RUN / "mano_deployment_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
