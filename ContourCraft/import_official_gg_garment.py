#!/usr/bin/env python3
"""Import a Gaussian-Garments garment using the REAL captured SMPL-X body
params for its template frame, instead of run_multigarment_pipeline_fixedsize.py's
recover_resize_garment_bootstrap.py (which re-fits a body via Adam+LBFGS
optimization as a fallback for when real params aren't available).

Body params come from the OFFICIAL ActorsHQ SMPL-X release used by the
"ActorsHQ-for-Gaussian-Garments" companion repo
(https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments, see its
data_preparation.py::unpack_smplx() -- downloads
<subject>/<sequence>/smpl_params.npz from Google Drive, file id
1DVk3k-eNbVqVCkLhGJhD_e9ILLCwhspR), NOT the raw per-frame ActorsHQ
smplx/*.json export (EasyMocap Rh/Th/poses/shapes format). Verified
empirically for Actor05/Sequence1 frame 156: this official npz's SMPL-X
forward pass matches the real captured body mesh (obstacle[156] in the
GG-rendered trajectory) to ~1mm mean vertex error; the raw per-frame json
was off by ~16mm at the same nominal frame index because it has a different
total frame count (2500 vs the official npz's 2338, which matches
stage2/<sequence>/meshes/ exactly) -- i.e. the two aren't even the same
real-world timestamp at a given index. Use fetch_official_actorshq_smplx.py
(or the manual gdown steps in this repo's notes) to obtain smpl_params.npz
for a new subject before running this script.

Calls utils.mesh_creation.GarmentCreator._add_posed_garment_raw() directly
(the same official, unmodified function GauGarConverter._import_garment()
calls) rather than going through GauGarConverter/_convert_smplx_subject().
That wrapper's CMU-format round-trip assumes 6-dim PCA hand pose input
(_convert_smplx_dir() does `pose[:, :6] @ left_hand_components` to expand
it) -- but the official npz's left_hand_pose/right_hand_pose are already
full 45-dim axis-angle, and GarmentCreator's internal SMPL-X body model is
built with use_pca=False (utils/mesh_creation.py:124), so feeding it PCA-
sliced-then-expanded garbage would be wrong. Full 45-dim poses need no
expansion at all -- calling _add_posed_garment_raw() directly sidesteps the
mismatch entirely instead of monkeypatching official code. leye_pose/
reye_pose aren't in the official npz; zeroed, matching unpack_smplx()'s own
convention.

The garment is relaxed with the PRETRAINED ContourCraft checkpoint, same as
every other non-finetuned garment in run_multigarment_pipeline_fixedsize.py
-- this script has nothing to do with GG's own Stage 4 (behavior finetune),
which is skipped entirely.

Output layout matches run_multigarment_pipeline_fixedsize.py's per-garment
cache exactly, so once this script has run for a garment, invoking that
pipeline normally with the same --output-root sees the cache already
populated (ensure_garment()'s is_relaxed()/is_pinned() checks read the
recovery_metadata this script stamps) and skips straight to Stage 3
(untangle) for it -- recover_resize_garment_bootstrap.py,
relax_recovered_garment.py, and pin_garment_waistband.py are never invoked
for this garment:

    <output-root>/<subject>/garment_dict.pkl       relaxed garment dict
    <output-root>/<subject>/source_body_fit.pkl     {"betas": <real captured betas>, ...}
    <output-root>/<subject>/relax_trajectory.pkl    relaxation rollout (for inspection)

Assumes gaugar_output_root/<subject>/ already has stage1/template_uv.obj
(true for any subject that finished GG Stage 1).

Waistband pinning (--pin-waistband) ranks this garment's open boundary
loops by mean height in the RAW POSED template.obj (not the canonical/
unposed LBS space pin_garment_waistband.py ranks in) because pinned_indices
must be decided before _add_posed_garment_raw() unposes+relaxes in one
un-splittable call. Fine for a roughly-upright capture pose (true for
ActorsHQ-style scans) where the waist sits well above ankle cuffs either
way; use --dry-run to sanity-check the candidate loops first, and if the
pin ever looks wrong, don't pass --pin-waistband here -- instead let this
script relax unpinned, then run the ordinary pin_garment_waistband.py +
relax_recovered_garment.py on the result (it operates on any garment_dict.pkl
regardless of how it was produced).

KNOWN BLOCKER, not handled by this script: stage1/template_uv.obj for
Actor05 (and every other stage1 template variant checked so far for this
subject) has at least one small fully-disconnected mesh fragment (e.g. a
stray 3-vertex/1-face island) alongside the main garment body.
GarmentCreator.add_coarse_edges() (utils/mesh_creation.py:128), called
inside _make_garment_dict_from_verts() since this script builds with
coarse=True, splits the mesh into connected components and builds a coarse
edge hierarchy per component -- a degenerate tiny component produces a
differently-shaped edge array than the main body and crashes
np.concatenate(). Needs a mesh cleanup pass (keep only the largest
connected component) before this script can complete; not included here.

Example:
    python import_official_gg_garment.py \\
        --contourcraft-root . \\
        --smplx-npz /path/to/GG_projects/actorshq_official_smplx/Actor05/Sequence1/smpl_params.npz \\
        --gaugar-output-root /path/to/GG_projects/training_result/gg_outputs_actor05_pants_tf000156 \\
        --output-root /path/to/GG_projects/gg_outputs_fixedsize \\
        --subject Actor05 --template-frame 156 \\
        --gender male \\
        --pin-waistband
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--smplx-npz", type=Path, required=True,
        help="Official ActorsHQ smpl_params.npz for this subject/sequence (see docstring "
             "for where it comes from). Must have betas (1,10), and per-frame "
             "global_orient/transl/body_pose/jaw_pose/left_hand_pose/right_hand_pose.")
    parser.add_argument("--gaugar-output-root", type=Path, required=True,
        help="GG output root for this subject -- must already contain "
             "<subject>/stage1/template_uv.obj (e.g. training_result/gg_outputs_actor05_pants_tf000156)")
    parser.add_argument("--output-root", type=Path, required=True,
        help="Same --output-root you pass to run_multigarment_pipeline_fixedsize.py -- "
             "garment_dict.pkl / source_body_fit.pkl are written to <output-root>/<subject>/")
    parser.add_argument("--body-model-root", type=Path, default=None,
        help="Default: <contourcraft-root>/ccraft_data/aux_data/body_models")
    parser.add_argument("--subject", required=True, help="e.g. Actor05")
    parser.add_argument("--template-frame", type=int, required=True,
        help="Frame index into --smplx-npz (and into stage2/<sequence>/meshes/, which it's "
             "aligned to) -- e.g. 156.")
    parser.add_argument("--gender", required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--checkpoint", type=Path, default=None,
        help="Default: <contourcraft-root>/ccraft_data/trained_models/contourcraft.pth "
             "(the PRETRAINED checkpoint -- this script is specifically for the "
             "skip-finetune path)")
    parser.add_argument("--pin-waistband", action="store_true",
        help="Auto-pin the topmost open boundary loop before relaxing (recommended for "
             "pants/skirts). See the docstring caveat about which space it ranks in.")
    parser.add_argument("--min-loop-vertices", type=int, default=5)
    parser.add_argument("--keep-small-components", action="store_true",
        help="Skip the largest-connected-component cleanup below and use "
             "stage1/template_uv.obj exactly as-is. Only for debugging -- "
             "add_coarse_edges() will crash if the mesh has more than one connected "
             "component (confirmed true of stage1/template_uv.obj for this subject: "
             "a 7973-vertex main body plus a stray disconnected 3-vertex/1-face island).")
    parser.add_argument("--dry-run", action="store_true",
        help="With --pin-waistband: print candidate boundary loops and exit without "
             "importing anything.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.contourcraft_root.resolve()))
    from utils.mesh_creation import GarmentCreator
    from utils.io import load_obj, save_obj
    from utils.coarse import make_graph_from_faces
    import networkx as nx

    if args.body_model_root is None:
        args.body_model_root = args.contourcraft_root / "ccraft_data" / "aux_data" / "body_models"
    if args.checkpoint is None:
        args.checkpoint = args.contourcraft_root / "ccraft_data" / "trained_models" / "contourcraft.pth"

    template_obj_path = args.gaugar_output_root / args.subject / "stage1" / "template_uv.obj"

    if not args.keep_small_components:
        # GarmentCreator.add_coarse_edges() (utils/mesh_creation.py:128) splits the mesh
        # into connected components and builds a coarse-edge hierarchy per component --
        # a stray, fully-disconnected fragment (reconstruction/segmentation noise, not a
        # real part of the garment) produces a differently-shaped edge array than the
        # main body and crashes np.concatenate(). Keep only the largest component and
        # write the cleaned mesh to a separate location -- never touches the original
        # file under --gaugar-output-root. Does NOT touch small holes/openings that stay
        # part of the main component (e.g. this subject's ~40 tiny boundary-loop holes
        # found earlier) -- those don't crash add_coarse_edges and are left alone.
        vertices, faces, uv_coords, uv_faces = load_obj(template_obj_path, tex_coords=True)
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)
        uv_coords = np.asarray(uv_coords)
        uv_faces = np.asarray(uv_faces)

        components = list(nx.connected_components(make_graph_from_faces(faces)))
        if len(components) > 1:
            sizes = sorted((len(c) for c in components), reverse=True)
            print(f"NOTE: {template_obj_path.name} has {len(components)} disconnected "
                  f"components (sizes={sizes}) -- keeping only the largest ({sizes[0]} "
                  f"verts), dropping the rest as reconstruction noise. Pass "
                  f"--keep-small-components to disable this.")
            keep_ids = np.array(sorted(max(components, key=len)))
            keep_mask = np.zeros(len(vertices), dtype=bool)
            keep_mask[keep_ids] = True
            remap = -np.ones(len(vertices), dtype=np.int64)
            remap[keep_ids] = np.arange(len(keep_ids))

            face_keep = keep_mask[faces].all(axis=1)
            vertices = vertices[keep_ids]
            faces = remap[faces[face_keep]]
            if len(uv_coords) == len(keep_mask):
                uv_coords = uv_coords[keep_ids]
                uv_faces = remap[uv_faces[face_keep]]
            else:
                # obj uses separate v/vt indexing -- filter uv_faces the same way but
                # leave uv_coords untouched rather than apply a mismatched remap.
                uv_faces = uv_faces[face_keep]
                print(f"  (uv_coords count {len(uv_coords)} != vertex count {len(keep_mask)}, "
                      f"left uv_coords as-is -- only filtered uv_faces)")

            cleaned_dir = args.output_root / "_cleaned_templates" / args.subject
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            template_obj_path = cleaned_dir / "template_uv.obj"
            save_obj(template_obj_path, vertices, faces, verts_uvs=uv_coords, faces_uvs=uv_faces)
            print(f"  Cleaned mesh ({len(vertices)} verts, {len(faces)} faces) written to "
                  f"{template_obj_path}")

    pinned_indices = None
    if args.pin_waistband or args.dry_run:
        from pin_garment_waistband import find_boundary_loops
        vertices, faces, _, _ = load_obj(template_obj_path, tex_coords=True)
        vertices = np.asarray(vertices)
        loops = find_boundary_loops(np.asarray(faces))
        if not loops:
            raise RuntimeError(f"No open boundary loop found in {template_obj_path}; "
                                f"this garment may not need pinning -- rerun without --pin-waistband.")
        loops_with_y = sorted(loops, key=lambda loop: -float(vertices[loop, 1].mean()))
        print(f"Found {len(loops_with_y)} open boundary loop(s) in {template_obj_path.name} "
              f"({len(vertices)} vertices total):")
        for i, loop in enumerate(loops_with_y):
            y = vertices[loop, 1]
            print(f"  [{i}] {len(loop):4d} verts   mean_y={y.mean():+.4f}   y_range=[{y.min():+.4f}, {y.max():+.4f}]")
        if args.dry_run:
            return
        chosen = loops_with_y[0]
        if len(chosen) < args.min_loop_vertices:
            raise RuntimeError(f"Topmost loop only has {len(chosen)} vertices "
                                f"(< --min-loop-vertices {args.min_loop_vertices}); inspect manually.")
        pinned_indices = chosen.tolist()
        print(f"Pinning {len(pinned_indices)} vertices (topmost boundary loop, loop [0]).")

    garment_out_dir = args.output_root / args.subject
    garment_out_dir.mkdir(parents=True, exist_ok=True)
    garment_dict_path = garment_out_dir / "garment_dict.pkl"
    source_fit_path = garment_out_dir / "source_body_fit.pkl"
    if garment_dict_path.exists() or source_fit_path.exists():
        raise FileExistsError(
            f"{garment_dict_path} or {source_fit_path} already exists -- "
            f"delete both (and relax_trajectory.pkl) to rebuild.")

    print(f"\n=== Loading official SMPL-X params: {args.smplx_npz} (frame {args.template_frame}) ===")
    official = np.load(args.smplx_npz)
    frame = args.template_frame
    real_betas = np.asarray(official["betas"][0], dtype=np.float32)
    # Slice with frame:frame+1 (not a bare [frame]) to keep the leading batch
    # dimension of 1 -- same convention gaugar.py::_import_garment() uses
    # (`v[template_frame:template_frame+1]`). smplx's forward() concatenates
    # betas and expression along dim=-1 internally, so both need matching,
    # consistent batch dims, and expression must be present at all.
    body_params = dict(
        betas=real_betas[None],
        global_orient=np.asarray(official["global_orient"][frame:frame+1], dtype=np.float32),
        transl=np.asarray(official["transl"][frame:frame+1], dtype=np.float32),
        body_pose=np.asarray(official["body_pose"][frame:frame+1], dtype=np.float32),
        jaw_pose=np.asarray(official["jaw_pose"][frame:frame+1], dtype=np.float32),
        expression=np.asarray(official["expression"][frame:frame+1], dtype=np.float32),
        leye_pose=np.zeros((1, 3), dtype=np.float32),   # not in the official npz; zeroed like unpack_smplx() does
        reye_pose=np.zeros((1, 3), dtype=np.float32),
        left_hand_pose=np.asarray(official["left_hand_pose"][frame:frame+1], dtype=np.float32),   # full 45-dim, no PCA
        right_hand_pose=np.asarray(official["right_hand_pose"][frame:frame+1], dtype=np.float32),  # full 45-dim, no PCA
    )

    # Same GarmentCreator construction GauGarConverter._import_garment() uses
    # (utils/gaugar.py:174) -- garment_dicts_dir points straight at this
    # garment's own cache folder so _add_posed_garment_raw() writes there directly.
    gc = GarmentCreator(garment_out_dir, args.body_model_root, "smplx", args.gender,
                         n_samples_lbs=0, verbose=True, coarse=True,
                         approximate_center=True, swap_axes=False)
    obj_dict = gc._load_from_obj(template_obj_path)

    print(f"\n=== Importing + relaxing garment (frame {frame}, PRETRAINED checkpoint) ===")
    relax_trajectory = gc._add_posed_garment_raw(
        obj_dict, args.subject, body_params, args.checkpoint,
        n_relaxation_steps=30, pinned_indices=pinned_indices, gender=args.gender)

    # _add_posed_garment_raw() writes to {garment_dicts_dir}/{garment_name}.pkl,
    # i.e. garment_out_dir/<subject>.pkl -- rename to the garment_dict.pkl
    # filename ensure_garment() actually looks for.
    raw_output = garment_out_dir / f"{args.subject}.pkl"
    raw_output.rename(garment_dict_path)

    with garment_dict_path.open("rb") as f:
        garment_dict = pickle.load(f)
    garment_dict.setdefault("recovery_metadata", {})
    garment_dict["recovery_metadata"].update(dict(
        source="official_actorshq_smplx_import", relaxed=True, pinned=bool(pinned_indices),
        template_frame=frame, gender=args.gender,
        smplx_source=str(args.smplx_npz.resolve())))
    with garment_dict_path.open("wb") as f:
        pickle.dump(garment_dict, f)

    with source_fit_path.open("wb") as f:
        pickle.dump(dict(
            betas=real_betas, gender=args.gender,
            source_trajectory=str(args.smplx_npz.resolve()),
            source_frame=frame, rmse=0.0,
            betas_initialization="official_ActorsHQ_smpl_params_npz", betas_frozen=True), f)

    with (garment_out_dir / "relax_trajectory.pkl").open("wb") as f:
        pickle.dump(relax_trajectory, f)

    print(f"\nDone.\n  {garment_dict_path}\n  {source_fit_path}\n"
          f"are now cached -- run_multigarment_pipeline_fixedsize.py --garment-names "
          f"{args.subject} ... will skip Stage 1/1b/2 and go straight to Stage 3 (untangle).")


if __name__ == "__main__":
    main()
