#!/usr/bin/env python3
"""EXPERIMENTAL alternative to generate_garment_thumbnail.py: render a
catalog thumbnail straight from the garment's own raw GG capture, instead of
re-simulating it onto a synthesized/standing body.

Why this exists (see conversation, not written down elsewhere): generate_
garment_thumbnail.py's --use-source-body still runs a full fit_body -> rebind
-> ContourCraft cloth-simulate pipeline, just retargeted onto the garment's
OWN recovered body shape instead of a synthesized one. That fixed shape-
retarget distortion, but two things it does NOT fix:
  1. It's the WRONG use of --hide-body ghost-mannequin compositing for a
     garment whose hem is naturally open (ankle cuffs, short sleeves, etc.):
     Doll.render (Gaussian-Garments/inference.py) always punches a hole in
     the garment layer wherever body should show through (neckline/cuffs/
     ankles), then composites body_image underneath. --hide-body swaps that
     to a blank WHITE canvas -- against this project's white thumbnail
     background, that reads as a torn hole even though nothing is wrong with
     the garment (confirmed empirically: same "hole" appears when rendering
     the untouched raw trajectory at frame 0, and completely disappears when
     the real body is composited in instead of white).
  2. Any mesh defect already baked into the raw GG reconstruction (checked:
     there wasn't one for 00176_lower -- 0 non-manifold edges, exactly 3
     boundary loops at the expected waist+2-ankle locations) would still
     ride along through fit/rebind/simulate. Re-simulating adds retarget/
     cloth-dynamics risk for zero benefit when all we want is "what does
     this garment look like".

Insight this script acts on: recover_resize_garment_bootstrap.py's own
--frame (garment_dict.pkl's recovery_metadata: source_trajectory +
source_frame) IS already a real, physically-settled, artifact-free capture
of the garment worn on its actual real body -- see this project's own
gaugar_trajectories/individual-garments/<name>_Take*.pkl. No fit_body, no
rebind, no ContourCraft cloth-simulate needed at all to look at it: this
script slices exactly that one frame out of the raw trajectory and renders
it directly with render_turntable.py, body SHOWN (not --hide-body), which
sidesteps problem (1) by construction and can't introduce problem (2) since
nothing touches the mesh between the raw capture and the render.

Trade-off vs. generate_garment_thumbnail.py: the thumbnail shows a grey
SMPL-X mannequin body wearing the garment (not a pure garment cutout) --
that's the plain-white-hole workaround. If a pure cutout look is wanted
later, composite something other than pure white for the hidden-body areas
instead (a separate, smaller change).

NEW SCRIPT, doesn't modify or import from generate_garment_thumbnail.py or
any existing orchestrator -- same convention this project's other
catalog-prep scripts already follow. EXPERIMENTAL: run it against a scratch
--output first and compare against the existing catalog_thumbnails/*.png
before overwriting anything.

Also supports import_official_gg_garment.py's garments (recovery_metadata.
source == "official_actorshq_smplx_import", e.g. Actor05): those don't have
a source_trajectory of already-posed garment frames -- their "source" is raw
SMPL-X body pose params, not a garment mesh sequence, so there's no frame to
slice out of a trajectory the way the recover_resize_garment_bootstrap.py
path works. Instead this reuses <output_root>/<garment_name>/relax_
trajectory.pkl, which import_official_gg_garment.py already wrote (every
step of settling the garment onto the posed body, same pred/obstacle/
cloth_faces/obstacle_faces schema) -- no need to redo SMPL-X forward
kinematics from smplx_source just to get a body mesh that already exists.
Uses the LAST relaxation step by default (most settled / fewest collisions,
see that trajectory's own metrics.ncoll), not frame 0 -- --frame means a
different thing for this source type, see its own help text.

--slot upper/lower additionally crops the render to a modest half-body shot
(see crop_half_body) -- a single garment on a full standing body leaves the
OTHER half bare, which looks indecent rather than just incomplete.

Example:
    python generate_garment_thumbnail_from_source.py --garment-name 00176_lower \\
        --slot lower --output /tmp/00176_lower_from_source.png
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent

RENDER_ENV_EXTRA = dict(
    UCX_MEMTYPE_CACHE="n",
    UCX_TLS="tcp,cuda_copy",
    UCX_ERROR_SIGNALS="",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)

RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


def run(cmd, **kwargs):
    """Same retry-on-failure wrapper generate_garment_thumbnail.py /
    run_outfit_preview_pipeline.py use (WSL2 GPU passthrough flakiness) --
    duplicated, not imported, see this script's own docstring."""
    env = {**os.environ, **kwargs.pop("extra_env", {})}
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        print("+ " + " ".join(str(c) for c in cmd))
        try:
            subprocess.run(cmd, check=True, env=env, **kwargs)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                print(f"!! Command failed (attempt {attempt}/{RETRY_ATTEMPTS}, exit {exc.returncode}). "
                      f"Retrying in {RETRY_DELAY_SECONDS}s in case this was transient GPU flakiness...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"!! Command failed {RETRY_ATTEMPTS} times in a row, giving up.")
    raise last_exc


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def compute_yaw_correction(global_orient: np.ndarray) -> np.ndarray:
    """Same yaw-alignment math as make_display_pose_sequence() (generate_
    garment_thumbnail.py / run_outfit_preview_pipeline.py) -- duplicated, not
    imported, same convention this project's other orchestrators follow.
    That function applies this correction to an SMPL-X root_orient pose
    parameter (then lets forward kinematics repose the whole body); here we
    need a rotation matrix to apply directly to raw XYZ vertices instead, so
    this stops one step earlier and returns the "yaw_raw" matrix -- already
    transformed back into the same (unswapped) frame root_orient itself
    lives in -- rather than composing it into a new root_orient.

    global_orient is source_body_fit.pkl's fitted SMPL-X root axis-angle --
    recover_resize_garment_bootstrap.py's Adam+LBFGS fit it directly against
    THIS garment's own raw obstacle vertices (RMSE ~4-5mm, see recovery_
    metadata), so it's a faithful description of this exact frame's body
    orientation, in the same world frame those raw vertices already live in.
    """
    swap_axes_rotation = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    root_matrix = Rotation.from_rotvec(global_orient).as_matrix()
    forward_final = swap_axes_rotation @ (root_matrix @ np.array([0.0, 0.0, 1.0]))
    yaw = -np.arctan2(forward_final[0], forward_final[2])
    yaw_world = Rotation.from_euler("y", yaw).as_matrix()
    return swap_axes_rotation.T @ yaw_world @ swap_axes_rotation


def apply_yaw(verts: np.ndarray, yaw_raw: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    return (verts - pivot) @ yaw_raw.T + pivot


def crop_half_body(img: Image.Image, slot: str) -> Image.Image:
    """Crop the full standing-body render down to just the half relevant to
    this garment, instead of showing a full body with the OTHER half bare
    (a single lower-body garment on a full body leaves the whole torso bare,
    and vice versa -- looks indecent, not just "incomplete").

    Finds the garment's own pixel rows by color rather than by any 3D/camera
    math: background is near-white, the grey mannequin body is low-
    saturation grey, so anything that's neither of those is treated as
    garment fabric. Cheap and robust across this catalog's colors (denim
    blue, cream, dark grey/brown) -- if a future garment is itself a
    near-white or body-grey color this would misdetect its rows, but that's
    a real edge case to handle then, not preemptively.

    upper: keep from the top (head) down to a bit below the hem -- cropping
    right at the hem left a sliver of bare thigh right at the frame edge,
    which read as MORE awkward than just including a little more leg, so a
    fixed 120px margin is added below the detected hem.
    lower: crop starts EXACTLY at the top of the garment (the waistband) --
    tried a softer margin above it first (to show a sliver of midriff) but
    that cut the frame right through the bare chest, which drew MORE
    attention to it than showing nothing above the waist at all.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w, _ = arr.shape
    is_bg = arr.min(axis=-1) > 245
    ch_max, ch_min = arr.max(axis=-1), arr.min(axis=-1)
    sat = ch_max - ch_min
    is_body_grey = (~is_bg) & (sat < 18) & (ch_max > 120)
    is_garment = (~is_bg) & (~is_body_grey)
    ys, _xs = np.where(is_garment)
    if len(ys) == 0:
        print("WARNING: crop_half_body couldn't find any garment-colored pixels -- "
              "leaving the image uncropped.")
        return img
    top, bottom = int(ys.min()), int(ys.max())
    if slot == "upper":
        crop_top, crop_bottom = 0, min(h, bottom + 120)
    elif slot == "lower":
        crop_top, crop_bottom = top, h
    else:
        raise ValueError(f"--slot must be 'upper' or 'lower' to crop, got {slot!r}")
    return img.crop((0, crop_top, w, crop_bottom))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--garment-name", required=True)
    parser.add_argument("--slot", choices=["upper", "lower", "none"], default="none",
        help="Crop the render to a modest half-body shot instead of a full standing "
             "body (see crop_half_body) -- 'upper'/'lower' match preview_server.py's "
             "GARMENTS slot field. 'none' (default) leaves the full-body render as is "
             "(e.g. for a garment that already covers most of the body).")
    parser.add_argument("--frame", type=int, default=None,
        help="Which frame to render. For a garment with recovery_metadata."
             "source_trajectory: an index into that raw capture, default "
             "whatever recovery_metadata['source_frame'] says (almost always "
             "0). For an official_actorshq_smplx_import garment (e.g. "
             "Actor05): an index into its relax_trajectory.pkl's relaxation "
             "steps instead (supports negative indexing), default -1 (the "
             "last/most-settled step) -- 0 would mean the FIRST, least-"
             "settled step for this source type, not the real capture.")
    parser.add_argument("--align-forward", action=argparse.BooleanOptionalAction, default=False,
        help="KNOWN BROKEN, defaults off -- do not enable without fixing compute_yaw_"
             "correction first (see its docstring). Was meant to rotate the raw capture "
             "so every garment's thumbnail faces the camera consistently instead of each "
             "one sitting at whatever angle its own raw capture happened to be shot at. "
             "Tried borrowing source_body_fit.pkl's fitted global_orient and extracting "
             "just its yaw component (same trick make_display_pose_sequence() uses for "
             "AMASS motion, which IS known to be upright) -- but this raw capture's "
             "global_orient comes from an unconstrained from-scratch fit with no upright "
             "prior, so treating it as yaw-only is invalid and visibly tips the whole "
             "body onto its side. A correct fix would derive facing direction from the "
             "raw obstacle mesh's own geometry (e.g. known SMPL-X shoulder/hip vertex "
             "ids) instead of borrowing this fit.")
    parser.add_argument("--n-angles", type=int, default=1)
    parser.add_argument("--bake-samples", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True, help="Final PNG path (angle_000).")
    parser.add_argument("--work-dir", type=Path, default=None,
        help="Scratch directory for intermediate files. Default: alongside --output.")
    parser.add_argument("--output-root", type=Path, default=Path("/path/to/GG_projects/gg_outputs_fixedsize"),
        help="Where <garment-name>/garment_dict.pkl lives (for its recovery_metadata).")
    parser.add_argument("--gaugar-root", type=Path, default=Path("/path/to/GG_projects/Gaussian-Garments"))
    parser.add_argument("--gaugar-python", default="/path/to/conda/envs/gaugar-sm120/bin/python")
    args = parser.parse_args()

    if args.work_dir is None:
        args.work_dir = args.output.parent / f".thumbnail_from_source_work_{args.garment_name}"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    garment_dict_path = args.output_root / args.garment_name / "garment_dict.pkl"
    if not garment_dict_path.exists():
        raise FileNotFoundError(
            f"{garment_dict_path} doesn't exist -- this garment hasn't been recovered yet "
            f"(recovery_metadata, which this script reads, is written at that step).")
    garment_dict = load_pickle(garment_dict_path)
    meta = garment_dict.get("recovery_metadata")
    if not meta:
        raise ValueError(f"{garment_dict_path} has no recovery_metadata -- "
                          f"can't find the raw capture this garment came from.")

    if meta.get("source") == "official_actorshq_smplx_import":
        # See this script's module docstring for why this source type reuses
        # relax_trajectory.pkl instead of slicing a source_trajectory.
        relax_path = args.output_root / args.garment_name / "relax_trajectory.pkl"
        if not relax_path.exists():
            raise FileNotFoundError(
                f"{relax_path} doesn't exist -- import_official_gg_garment.py should "
                f"have written it alongside {garment_dict_path.name}.")
        raw = load_pickle(relax_path)
        n_steps = raw["pred"].shape[0]
        frame = args.frame if args.frame is not None else -1
        frame_idx = frame if frame >= 0 else n_steps + frame
        print(f"Using relaxation trajectory: {relax_path} step {frame_idx}/{n_steps - 1} "
              f"(import_official_gg_garment.py's own settle of the real smplx_source "
              f"capture, template_frame={meta.get('template_frame')})")
        sliced = {
            "pred": raw["pred"][frame_idx:frame_idx + 1],
            "obstacle": raw["obstacle"][frame_idx:frame_idx + 1],
            "cloth_faces": raw["cloth_faces"],
            "obstacle_faces": raw["obstacle_faces"],
            "garment_names": [args.garment_name],
        }
    elif "source_trajectory" in meta:
        source_trajectory = Path(meta["source_trajectory"])
        frame = args.frame if args.frame is not None else meta["source_frame"]
        print(f"Using raw capture: {source_trajectory} frame {frame} "
              f"(recover_resize_garment_bootstrap.py used this exact frame originally, "
              f"body_fit RMSE={meta.get('body_fit_rmse')})")

        raw = load_pickle(source_trajectory)
        sliced = {
            "pred": raw["pred"][frame:frame + 1],
            "obstacle": raw["obstacle"][frame:frame + 1],
            "cloth_faces": raw["cloth_faces"],
            "obstacle_faces": raw["obstacle_faces"],
            # Explicit rather than relying on inference.py's own filename-inference
            # fallback (Simulation.__init__, "MODIFICATION" branch) -- this script
            # always knows the real name already.
            "garment_names": [args.garment_name],
        }
    else:
        raise ValueError(f"{garment_dict_path}'s recovery_metadata has neither "
                          f"source_trajectory nor a recognized source= tag -- "
                          f"can't find the raw capture this garment came from.")

    if args.align_forward:
        body_fit_path = args.output_root / args.garment_name / "source_body_fit.pkl"
        if not body_fit_path.exists():
            print(f"WARNING: {body_fit_path} not found -- skipping --align-forward, "
                  f"rendering at this capture's own raw angle.")
        elif "global_orient" not in load_pickle(body_fit_path):
            print(f"WARNING: {body_fit_path} has no 'global_orient' (e.g. an "
                  f"official_actorshq_smplx_import garment's source_body_fit.pkl only "
                  f"stores betas) -- skipping --align-forward, rendering at this "
                  f"capture's own raw angle.")
        else:
            body_fit = load_pickle(body_fit_path)
            yaw_raw = compute_yaw_correction(np.asarray(body_fit["global_orient"], dtype=np.float64))
            pivot = sliced["obstacle"][0].mean(axis=0)
            sliced["obstacle"] = apply_yaw(sliced["obstacle"][0], yaw_raw, pivot)[None].astype(np.float32)
            sliced["pred"] = apply_yaw(sliced["pred"][0], yaw_raw, pivot)[None].astype(np.float32)
            print(f"Applied yaw correction from {body_fit_path}")

    sliced_path = args.work_dir / "sliced_trajectory.pkl"
    with sliced_path.open("wb") as f:
        pickle.dump(sliced, f)

    render_dir = args.work_dir / "render"
    # Deliberately NOT --hide-body -- see this script's module docstring for why.
    run([args.gaugar_python, "render_turntable.py",
         "--traj_path", str(sliced_path.resolve()),
         "--output_path", str(render_dir.resolve()),
         "--frame_idx", "0",  # sliced trajectory has exactly 1 frame
         "--n-angles", str(args.n_angles),
         "--bake-samples", str(args.bake_samples)],
        cwd=str(args.gaugar_root), extra_env=RENDER_ENV_EXTRA)

    if args.slot in ("upper", "lower"):
        img = Image.open(render_dir / "angle_000.png")
        img = crop_half_body(img, args.slot)
        img.save(args.output)
    else:
        args.output.write_bytes((render_dir / "angle_000.png").read_bytes())
    print(f"\nWrote thumbnail: {args.output}")


if __name__ == "__main__":
    main()
