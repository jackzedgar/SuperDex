"""Bounded synthetic MuJoCo Warp preflight; NOT the D0 benchmark or a policy.

Uses the pinned training URDF, retains its source, and records all scene changes.
Only homogeneous worlds are measured. Timing excludes startup/JIT and learning.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from runtime_config import GPU as EXPECTED_GPU, ROOT
if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_GPU:
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the approved physical GPU 0 UUID")

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


@wp.kernel
def set_targets(ctrl: wp.array2d(dtype=float), centers: wp.array(dtype=float),
                amplitudes: wp.array(dtype=float), clock: wp.array(dtype=float)):
    world, joint = wp.tid()
    ctrl[world, joint] = centers[joint] + amplitudes[joint] * wp.sin(6.2831853 * clock[world])


def build_scene(base, scene, generated):
    generated.mkdir(parents=True, exist_ok=True)
    provenance = {"scene": scene, "synthetic": True, "wrist": "fixed"}
    if scene == "wuji":
        source = base / "assets/wuji-training/urdf/right.urdf"
        tree = ET.parse(source)
        compiler = tree.getroot().find("mujoco/compiler")
        meshdir = (source.parent / "../meshes/right").resolve()
        compiler.set("meshdir", str(meshdir))
        for mesh in tree.getroot().iter("mesh"):
            mesh.set("filename", Path(mesh.get("filename")).name)
        temporary = generated / "right_path_fixed.urdf"
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        imported = mujoco.MjModel.from_xml_path(str(temporary))
        assert imported.nv == 20
        converted = generated / "right_converted.xml"
        mujoco.mj_saveLastXML(str(converted), imported)
        root = ET.parse(converted).getroot()
        root.find("compiler").set("meshdir", str(meshdir))
        world = root.find("worldbody")
        mount = ET.Element("body", name="test_mount", pos="0 0 0.16",
                           quat="0.7071067812 0 0.7071067812 0")
        for child in list(world):
            world.remove(child)
            mount.append(child)
        world.append(mount)
        for geom in mount.iter("geom"):
            visual = geom.get("group") == "1"
            geom.set("contype", "0" if visual else "1")
            geom.set("conaffinity", "0" if visual else "2")
            geom.set("friction", "0.8 0.005 0.0001")
        act = ET.SubElement(root, "actuator")
        for joint in mount.iter("joint"):
            joint.set("armature", "0.0005")
            joint.set("damping", "0.04")
            ET.SubElement(act, "position", name=joint.get("name") + "_test_pd",
                          joint=joint.get("name"), kp="2", kv="0.08",
                          ctrlrange=joint.get("range"), forcerange="-0.2 0.2")
        provenance.update(source_urdf=str(source),
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          modifications=["absolute mesh paths in separate copy",
                                         "fixed rotated test mount", "test PD gains and armature",
                                         "hand-object and object-floor collision masks; no self collision"],
                          pd_is_identified=False)
        object_pos = "0.04 0 0.30"
    else:
        root = ET.Element("mujoco", model="synthetic_cube")
        world = ET.SubElement(root, "worldbody")
        object_pos = "0 0 0.30"
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib.update(timestep="0.002", integrator="implicitfast", solver="Newton",
                         iterations="20", ls_iterations="50", gravity="0 0 -9.81")
    ET.SubElement(world, "geom", name="test_floor", type="plane", size="1 1 0.05",
                  contype="4", conaffinity="2", friction="0.8 0.005 0.0001")
    obj = ET.SubElement(world, "body", name="test_object", pos=object_pos)
    ET.SubElement(obj, "freejoint", name="object_free")
    ET.SubElement(obj, "geom", name="test_cube", type="box", size="0.02 0.02 0.02",
                  mass="0.05", contype="2", conaffinity="5", friction="0.8 0.005 0.0001")
    path = generated / (scene + "_scene.xml")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT.parent)
    parser.add_argument("--scene", choices=["cube", "wuji"], default="cube")
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--controls", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.worlds <= 4096 and 100 <= args.controls <= 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.base / "superdex/cache/warp")
    wp.init()
    assert len(wp.get_cuda_devices()) == 1
    wp.set_device("cuda:0")
    scene_path, provenance = build_scene(args.base, args.scene, args.output.parent / "generated")
    m = mujoco.MjModel.from_xml_path(str(scene_path))
    initial = mujoco.MjData(m)
    joint_ids = m.actuator_trnid[:, 0]
    qadr = m.jnt_qposadr[joint_ids]
    centers = np.mean(m.jnt_range[joint_ids], axis=1).astype(np.float32)
    amplitudes = (0.08 * np.diff(m.jnt_range[joint_ids], axis=1)[:, 0]).astype(np.float32)
    if m.nu:
        initial.qpos[qadr] = centers
        initial.ctrl[:] = centers
    mujoco.mj_forward(m, initial)
    # Independent CPU contact/stability check of the generated scene.
    cpu = mujoco.MjData(m)
    cpu.qpos[:] = initial.qpos
    object_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "test_cube")
    floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "test_floor")
    hand_contact_steps = 0
    for _ in range(1000):
        if m.nu:
            cpu.ctrl[:] = centers + amplitudes * math.sin(2 * math.pi * cpu.time)
        mujoco.mj_step(m, cpu)
        pairs = [(c.geom1, c.geom2) for c in cpu.contact]
        hand_contact_steps += any(object_geom in p and floor_geom not in p for p in pairs)
    assert np.isfinite(cpu.qpos).all() and np.isfinite(cpu.qvel).all()
    if args.scene == "wuji":
        assert hand_contact_steps > 0, "Generated scene did not exercise hand-object contact"
    wm = mjw.put_model(m)
    wm.opt.graph_conditional = wp.is_conditional_graph_supported()
    wd = mjw.put_data(m, initial, nworld=args.worlds, nconmax=128, njmax=512)
    center_w = wp.array(centers, dtype=wp.float32, device="cuda:0")
    amp_w = wp.array(amplitudes, dtype=wp.float32, device="cuda:0")

    def control_step():
        if m.nu:
            wp.launch(set_targets, dim=(args.worlds, m.nu),
                      inputs=[wd.ctrl, center_w, amp_w, wd.time], device="cuda:0")
        for _ in range(10):
            mjw.step(wm, wd)

    compile_start = time.perf_counter()
    control_step()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        control_step()
    for _ in range(20):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    startup = time.perf_counter() - compile_start
    started = time.perf_counter()
    for _ in range(args.controls):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    elapsed = time.perf_counter() - started
    qpos = wd.qpos.numpy()
    qvel = wd.qvel.numpy()
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
    object_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    obj_qadr = m.jnt_qposadr[object_joint]
    z = qpos[:, obj_qadr + 2]
    assert np.all(z > 0.005) and np.all(z < 0.7), ("object escaped", z.min(), z.max())
    if args.scene == "cube":
        assert np.max(np.abs(z - 0.02)) < 0.003
    excursion = float(np.max(np.abs(qpos[:, qadr] - centers))) if m.nu else 0.0
    if m.nu:
        assert excursion > 0.01
        lo, hi = m.jnt_range[joint_ids].T
        assert np.all(qpos[:, qadr] >= lo - 0.15) and np.all(qpos[:, qadr] <= hi + 0.15)
    report = dict(status="PASS", scope="synthetic homogeneous preflight; not full D0",
                  provenance=provenance, gpu_uuid=EXPECTED_GPU,
                  versions={k:importlib.metadata.version(k) for k in ("mujoco", "mujoco-warp", "warp-lang", "numpy")},
                  num_worlds=args.worlds, controls=args.controls, physics_dt=0.002,
                  solver_iterations=20, linesearch_iterations=50,
                  substeps_per_control=10, control_hz=50, nq=m.nq, nv=m.nv, nu=m.nu,
                  cpu_hand_object_contact_steps=hand_contact_steps,
                  final_object_z_range_m=[float(z.min()),float(z.max())],
                  hand_excursion_rad=excursion, measured_seconds=elapsed,
                  environment_control_steps_per_second=args.worlds * args.controls / elapsed,
                  compile_and_warmup_seconds=startup,
                  limitations=["one object geometry and fixed wrist", "test gains, not identified hardware",
                               "no policy learning or tensor interop workload", "not H100 and not 140 heterogeneous objects",
                               "hand contact observed in independent CPU check; GPU validates state stability"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
