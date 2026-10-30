#!/usr/bin/env python3
"""Untangle a set of already-relaxed garment dicts into one non-interpenetrating
outfit, without modifying official code.

Wraps ContourCraft's utils.untanglement.Untangler (see Untanglement.ipynb): each
garment recovered + relaxed independently (recover_resize_garment_bootstrap.py /
relax_recovered_garment.py) sits in its own canonical zero-pose rest state, and
those rest states can overlap wherever garments would naturally sit close on a
body (e.g. a t-shirt under a jacket). Untangler resolves this by simulating the
garments one at a time, innermost to outermost, alternating "push the new outer
layer off the inner ones" and "relax everything together" passes, then
re-derives each garment's LBS weights from its untangled canonical shape.

Garments are addressed by name inside --garment-dicts-dir (a flat directory of
<name>.pkl files) and must be given innermost-first in --garment-names. Results
land at <garment-dicts-dir>/<outfit-name>/<name>.pkl, plus a manifest.json
recording the inputs so callers (run_multigarment_pipeline.py) can cache-check
the result instead of re-running the untangling simulation.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--garment-dicts-dir", type=Path, required=True,
        help="Flat directory containing <garment-name>.pkl for every name in --garment-names.")
    parser.add_argument("--garment-names", nargs="+", required=True,
        help="Innermost to outermost, e.g. --garment-names 00136_tshirt 00176_upper")
    parser.add_argument("--outfit-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gender", default="female",
        help="Gender of the shared canonical body the outfit is untangled on.")
    parser.add_argument("--model-type", default="smplx", choices=["smpl", "smplx"])
    parser.add_argument("--n-epochs", type=int, default=2,
        help="Untangling epochs per added outer layer (rigid-inner/cloth-outer pass, then all-cloth relax pass).")
    parser.add_argument("--steps-per-stage", type=int, default=30)
    parser.add_argument("--trajectory-output", type=Path, default=None,
        help="Optional debug trajectory .pkl of the untangling process (viewable via utils/show.py).")
    parser.add_argument("--skip-safecheck-collision-count", action="store_true",
        help="Stub out find_close_faces (BVH tri-tri proximity query, official/"
             "unmodified) for the untangling rollouts too -- Untangler's "
             "'simulate all garments as cloth' passes go through the same "
             "runner.valid_rollout(bare=True) path simulate_recovered_outfit.py "
             "does, so the same diagnostic-only 'ncoll' collision count (never "
             "read anywhere, ~25-28%% of that stage's wall time) applies here. "
             "See simulate_recovered_outfit.py's --help for the full explanation.")
    args = parser.parse_args()

    if len(args.garment_names) < 2:
        raise ValueError("--garment-names needs at least 2 garments to untangle -- for a single garment "
                          "use the recovered/relaxed garment dict directly with run_garment_pipeline.py.")

    outfit_dir = args.garment_dicts_dir / args.outfit_name
    if outfit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {outfit_dir} (pass a different --outfit-name)")

    for name in args.garment_names:
        source = args.garment_dicts_dir / f"{name}.pkl"
        if not source.exists():
            raise FileNotFoundError(f"Missing garment dict for '{name}': {source}")

    sys.path.insert(0, str(args.contourcraft_root.resolve()))
    from utils.defaults import DEFAULTS
    from utils.mesh_creation import GarmentCreator
    from utils.untanglement import Untangler
    from utils.io import pickle_dump

    DEFAULTS.project_dir = str(args.contourcraft_root.resolve())
    DEFAULTS.data_root = str(args.data_root.resolve())
    DEFAULTS.aux_data = str((args.data_root / "aux_data").resolve())
    DEFAULTS.experiment_root = str((args.data_root / "experiments").resolve())

    if args.skip_safecheck_collision_count:
        # See simulate_recovered_outfit.py's identical block for the full
        # explanation. Doesn't touch any official file.
        import torch
        import runners.utils.impulses as impulses_module

        def _skip_close_faces(vertices, faces, threshold=0):
            return torch.zeros((0, 2), dtype=torch.long, device=vertices.device)

        impulses_module.find_close_faces = _skip_close_faces
        print("--skip-safecheck-collision-count: find_close_faces stubbed out "
              "(diagnostic 'ncoll' metric will read 0; no effect on the untangled result)")

    garment_creator = GarmentCreator(
        str(args.garment_dicts_dir), str(args.body_model_root), args.model_type,
        args.gender, collect_lbs=True, n_samples_lbs=0, coarse=False, swap_axes=False)

    untangler = Untangler(
        garment_creator, str(args.checkpoint), n_steps_per_stage=args.steps_per_stage,
        n_epochs=args.n_epochs, gender=args.gender, use_uv=False,
        garment_dicts_dir=str(args.garment_dicts_dir))

    print(f"Untangling {args.garment_names} -> outfit '{args.outfit_name}'")
    _untanglement, trajectory_dict = untangler.untangle_all(list(args.garment_names), args.outfit_name)

    if args.trajectory_output is not None:
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        pickle_dump(trajectory_dict, args.trajectory_output)
        print(f"Untangling trajectory (debug): {args.trajectory_output}")

    # untangle_all() already wrote outfit_dir/<name>.pkl for every garment (via
    # Untanglement.save_new_garment_dicts) -- add our own cache-check marker on top.
    manifest = {
        "garment_names": list(args.garment_names),
        "outfit_name": args.outfit_name,
        "gender": args.gender,
        "model_type": args.model_type,
        "n_epochs": args.n_epochs,
        "steps_per_stage": args.steps_per_stage,
    }
    with (outfit_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    gnames_w_outfit = ",".join(f"{args.outfit_name}/{n}" for n in args.garment_names)
    print(f"\nUntangled outfit written to {outfit_dir}")
    print(f"garment_name for simulation: \"{gnames_w_outfit}\"")


if __name__ == "__main__":
    main()
