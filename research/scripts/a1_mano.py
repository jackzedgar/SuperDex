"""MANO v1.2 data-only loader and quaternion LBS for OakInk2.

Licensed arrays remain outside Git. No Chumpy code is imported or executed.
OakInk2 convention: wxyz rotations, full local rotations (no PCA/mean offset),
center_idx=0, then add the supplied world-space wrist translation, in metres.
"""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse
import torch

MODEL_SHA256 = {
    "left": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "right": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
SNAP_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class _ArrayRecord:
    """Inert stand-in for a serialized Chumpy array or selection."""


class _SparseRecord:
    """Inert stand-in for a serialized CSC matrix."""


class _ModelReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("__builtin__", "set"): set,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("scipy.sparse.csc", "csc_matrix"): _SparseRecord,
            ("chumpy.ch", "Ch"): _ArrayRecord,
            ("chumpy.reordering", "Select"): _ArrayRecord,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported MANO data class: {module}.{name}")
        return allowed[module, name]


def _array(value):
    if isinstance(value, _ArrayRecord):
        if hasattr(value, "x"):
            return np.asarray(value.x)
        if all(hasattr(value, k) for k in ("a", "idxs", "preferred_shape")):
            return _array(value.a).reshape(-1)[value.idxs].reshape(value.preferred_shape)
        raise ValueError("Unsupported serialized array expression")
    if isinstance(value, _SparseRecord):
        return scipy.sparse.csc_matrix(
            (value.data, value.indices, value.indptr), shape=value._shape
        ).toarray()
    return np.asarray(value)


def asset_root():
    default = Path(__file__).resolve().parents[2] / "assets/mano_v1_2"
    return Path(os.environ.get("SUPERDEX_MANO_ROOT", str(default))).resolve()


def load_arrays(side, root=None):
    if side not in MODEL_SHA256:
        raise ValueError(side)
    path = Path(root or asset_root()) / "models" / f"MANO_{side.upper()}.pkl"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != MODEL_SHA256[side]:
        raise ValueError(f"MANO {side} differs from the verified official v1.2 asset")
    with path.open("rb") as f:
        raw = _ModelReader(f, encoding="latin1").load()
    names = ["v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"]
    arrays = {k: _array(raw[k]) for k in names}
    expected = {"v_template": (778, 3), "shapedirs": (778, 3, 10),
                "posedirs": (778, 3, 135), "J_regressor": (16, 778),
                "weights": (778, 16), "kintree_table": (2, 16), "f": (1538, 3)}
    for k, shape in expected.items():
        if arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError(f"Invalid model array {k}")
    if not np.allclose(arrays["weights"].sum(-1), 1, atol=1e-6):
        raise ValueError("Invalid skinning weights")
    return arrays


def quaternion_matrix(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class ManoDecoder(torch.nn.Module):
    def __init__(self, side, root=None, dtype=torch.float32):
        super().__init__()
        self.side = side
        arrays = load_arrays(side, root)
        for k in ["v_template", "shapedirs", "posedirs", "J_regressor", "weights"]:
            self.register_buffer(k, torch.as_tensor(arrays[k].copy(), dtype=dtype))
        self.register_buffer("faces", torch.as_tensor(arrays["f"].astype(np.int64)))
        tree = arrays["kintree_table"]
        id_to_index = {int(j): i for i, j in enumerate(tree[1])}
        self.parents = [-1] + [id_to_index[int(j)] for j in tree[0, 1:]]
        if any(p >= i for i, p in enumerate(self.parents[1:], 1)):
            raise ValueError("Kinematic tree must be parent-before-child")

    def forward(self, pose_wxyz, betas, wrist_world):
        q = pose_wxyz.reshape(-1, 16, 4)
        b = q.shape[0]
        if betas.shape != (b, 10) or wrist_world.shape != (b, 3):
            raise ValueError("Expected matching pose [B,16,4], shape [B,10], wrist [B,3]")
        if not all(torch.isfinite(x).all() for x in [q, betas, wrist_world]):
            raise ValueError("Non-finite MANO input")
        if torch.any(torch.linalg.vector_norm(q, dim=-1) < 1e-8):
            raise ValueError("Zero quaternion")
        r = quaternion_matrix(q)
        shaped = self.v_template + torch.einsum("vck,bk->bvc", self.shapedirs, betas)
        joints0 = torch.einsum("jv,bvc->bjc", self.J_regressor, shaped)
        identity = torch.eye(3, dtype=q.dtype, device=q.device)
        posed = shaped + torch.einsum("vck,bk->bvc", self.posedirs, (r[:, 1:] - identity).reshape(b, 135))
        transforms = []
        for j, parent in enumerate(self.parents):
            relative = joints0[:, j] if parent < 0 else joints0[:, j] - joints0[:, parent]
            top = torch.cat((r[:, j], relative[..., None]), -1)
            bottom = q.new_tensor([0, 0, 0, 1]).expand(b, 1, 4)
            local = torch.cat((top, bottom), -2)
            transforms.append(local if parent < 0 else transforms[parent] @ local)
        absolute = torch.stack(transforms, 1)
        rotated_rest = torch.einsum("bjcd,bjd->bjc", absolute[..., :3, :3], joints0)
        corrected = absolute.clone()
        corrected[..., :3, 3] = absolute[..., :3, 3] - rotated_rest
        blended = torch.einsum("vj,bjcd->bvcd", self.weights, corrected)
        homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
        vertices = torch.einsum("bvcd,bvd->bvc", blended, homogeneous)[..., :3]
        tips = [745, 317, 444 if self.side == "right" else 445, 556, 673]
        joints = torch.cat((absolute[..., :3, 3], vertices[:, tips]), 1)[:, SNAP_ORDER]
        center = joints[:, :1]
        return {"vertices": vertices - center + wrist_world[:, None],
                "joints": joints - center + wrist_world[:, None],
                "rotations": absolute[..., :3, :3]}
