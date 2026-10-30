#!/usr/bin/env python3
"""Render ONE already-simulated static frame from several camera angles
("turntable" images), instead of one frame from a moving camera.

Sibling of render_lean.py, same relationship to inference.py: reuses
inference.py's own Simulation/Doll classes and helpers by importing it as a
module (`import inference as inf`), doesn't modify render_lean.py or
inference.py.

Why this is cheap per extra angle, not free but not full price either:
`Simulation.prepare_frame(idx)` (inference.py, unmodified) does the mesh
export + Blender Cycles AO/normal texture bake -- by far the most expensive
step (~60-65% of render_lean.py's own "prep" bucket, per its docstring) --
and that function is entirely camera-independent (it bakes onto the
garment's UV texture space, no camera involved). It's ALSO already
self-caching to disk (`if not os.path.exists(_ambient): ...` /
`if not os.path.exists(_body): ...`), so calling it once per angle with the
SAME frame idx and SAME --output_path (as this script does, on purpose)
only actually bakes on the first angle -- every angle after that just reads
the cached PNGs/PLY back off disk. What's NOT free and genuinely gets
redone per angle: AvatarNet's forward pass (`update_garments`, takes the
camera as an input -- the shading it produces is view-dependent) and the
o3d visibility raycast (`prepare_gaussian`, needs to know what's visible
from THIS camera), plus the cheap rasterization calls. Expect roughly
"first angle full price, each additional angle a fraction of that" -- see
run_outfit_preview_pipeline.py's docstring for a measured estimate.

Must be launched with the gaugar Python environment and cwd=Gaussian-Garments
root, exactly like render_lean.py:

    <gaugar-python> render_turntable.py --traj_path traj.pkl \
        --output_path turntable/ --frame_idx 4 --n-angles 12 --bake-samples 4
"""

import sys
import os
import time
import functools
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera

import inference as inf
from utils.defaults import DEFAULTS
from arguments import ModelParams, PipelineParams
from render_lean import Config, bake_texture_fast


def turntable_camera(base_target: np.ndarray, base_offset: np.ndarray, theta: float, w: int, h: int, viewer):
    """A PinholeCamera at `base_target`, rotated by `theta` radians around the
    world "up" (Y) axis from the default view's own offset -- so theta=0
    reproduces render_lean.py/inference.py's own default "horizontal view"
    exactly (see inference.py's ait_render, cam=None branch)."""
    c, s = np.cos(theta), np.sin(theta)
    rotated_offset = np.array([
        base_offset[0] * c + base_offset[2] * s,
        base_offset[1],
        -base_offset[0] * s + base_offset[2] * c,
    ])
    position = base_target + rotated_offset
    return PinholeCamera(position, base_target, w, h, viewer=viewer)


if __name__ == "__main__":
    parser = ArgumentParser(description="Turntable render (one frame, many angles)")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    cp = Config(parser)
    parser.add_argument("--bake-samples", type=int, default=4,
        help="Same meaning as render_lean.py's --bake-samples.")
    parser.add_argument("--traj_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, default=None,
        help="Which trajectory frame to turn around. Default: the last frame "
             "(the settled pose after however many simulate steps the caller "
             "ran).")
    parser.add_argument("--n-angles", type=int, default=12,
        help="Evenly-spaced camera azimuths around the body, saved as "
             "angle_000.png .. angle_{n_angles-1:03d}.png. angle_000.png "
             "reproduces the same default camera render_lean.py/inference.py "
             "use everywhere else, so it can double as a drop-in preview.png.")
    parser.add_argument("--hide-body", action="store_true",
        help="Ghost-mannequin style product shot: the garment still needs a "
             "real body+pose underneath during simulation to hold its "
             "physically-draped shape (cloth has no shape of its own), but "
             "the body is not drawn in the final image -- just a blank white "
             "canvas composited behind the garment instead of the actual "
             "rendered body panel. Doll.render (inference.py, unmodified) "
             "already punches a hole in the garment layer wherever the body "
             "should show through in front of it (neckline/cuffs/ankles, "
             "via _body_mask) and composites body_image underneath -- this "
             "flag just swaps what's passed as body_image, nothing about the "
             "physics or the mask logic changes.")
    args = parser.parse_args(sys.argv[1:])

    w, h = 940, 1280

    args.source_path = DEFAULTS.data_root
    args.store_path = DEFAULTS.output_root
    viewer = HeadlessRenderer(size=(w, h))
    viewer.playback_fps = 30
    viewer.shadows_enabled = False
    viewer.scene.origin.enabled = False
    cmap = plt.get_cmap('gist_rainbow')

    # Same globals-reassignment trick render_lean.py uses -- inference.py's
    # helpers read these off its own module at call time.
    inf.w, inf.h = w, h
    inf.viewer = viewer
    inf.cmap = cmap
    inf.bake_texture = functools.partial(bake_texture_fast, samples=args.bake_samples)

    simu = inf.Simulation(args)
    frame_idx = args.frame_idx if args.frame_idx is not None else len(simu) - 1
    os.makedirs(args.output_path, exist_ok=True)

    with torch.inference_mode():
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Probe call: adds the body mesh to the scene and hands back
        # inference.py's own default camera for it, purely so we can read
        # its target/offset back out and build our own cameras around the
        # same point -- this is the same "horizontal view" default branch
        # every other script in this project relies on (inference.py's
        # ait_render, cam=None), just inspected instead of used directly.
        _, _, probe_cam = inf.ait_render(vb=simu.simu_pkl['obstacle'][frame_idx],
                                          fb=simu.simu_pkl['obstacle_faces'])
        base_target = np.array(probe_cam.current_target, dtype=np.float64).copy()
        base_offset = np.array(probe_cam.position, dtype=np.float64).copy() - base_target

        total_prep_time = 0.0
        total_render_time = 0.0
        total_io_time = 0.0
        global_start_time = time.time()

        def tick():
            torch.cuda.synchronize()
            return time.time()

        for i in range(args.n_angles):
            theta = 2 * np.pi * i / args.n_angles
            camera = turntable_camera(base_target, base_offset, theta, w, h, viewer)

            t0 = tick()
            _body, _body_depth, ait_cam = inf.ait_render(
                vb=simu.simu_pkl['obstacle'][frame_idx], fb=simu.simu_pkl['obstacle_faces'], cam=camera)
            gs_cam = inf.from_ait_to_gs(ait_cam, w, h)
            # Cheap after i==0: prepare_frame caches its mesh/bake outputs to
            # disk keyed on frame_idx + this script's own --output_path (see
            # module docstring), and camera doesn't factor into what it does
            # at all.
            vert_list, ambient_list, normal_list = simu.prepare_frame(frame_idx)
            style_list = simu.doll.update_garments(vert_list, ambient_list, normal_list, gs_cam)
            simu.doll.body = simu.doll.get_o3d_mesh(simu.simu_pkl['obstacle'][frame_idx],
                                                      faces=simu.simu_pkl['obstacle_faces'])
            simu.doll.prepare_gaussian(gs_cam.camera_center, style_list)

            face_list = [g.mesh.f.cpu().numpy() for g in simu.doll.garments]
            _, _body_garm_depth, _ = inf.ait_render(
                vert_list, face_list, simu.simu_pkl['obstacle'][frame_idx], simu.simu_pkl['obstacle_faces'], ait_cam)
            _, _garm_depth, _ = inf.ait_render(vert_list, face_list, cam=ait_cam)
            t1 = tick()

            _body_mask = _body_garm_depth < _garm_depth
            body_image = torch.ones_like(_body) * 255 if args.hide_body else _body
            _final = simu.doll.render(gs_cam, simu.doll, args, bg, body_image, _body_mask)

            torch.cuda.synchronize()
            t2 = time.time()

            Image.fromarray(np.array(_final, dtype=np.uint8)).save(
                os.path.join(args.output_path, f"angle_{i:03d}.png"))
            t3 = time.time()

            total_prep_time += (t1 - t0)
            total_render_time += (t2 - t1)
            total_io_time += (t3 - t2)

    global_end_time = time.time()
    print("\n" + "=" * 50)
    print(f"Turntable: {args.n_angles} angles of frame {frame_idx}")
    print(f"Total time taken: {global_end_time - global_start_time:.2f} seconds")
    total_tracked = total_prep_time + total_render_time + total_io_time
    print(f"1. prep (GPU, bake cached after angle 0): {total_prep_time:.2f}s ({(total_prep_time/total_tracked)*100:.1f}%)")
    print(f"2. gaussian render (GPU): {total_render_time:.2f}s ({(total_render_time/total_tracked)*100:.1f}%)")
    print(f"3. image I/O (CPU): {total_io_time:.2f}s ({(total_io_time/total_tracked)*100:.1f}%)")
    print("=" * 50 + "\n")
    print("Turntable render finished")
