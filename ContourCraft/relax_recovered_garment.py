#!/usr/bin/env python3
"""Relax a recovered garment in zero pose without modifying official code."""

import argparse
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--garment-dict", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--gender", default="female")
    args = parser.parse_args()
    if args.trajectory_output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.trajectory_output}")

    sys.path.insert(0, str(args.contourcraft_root.resolve()))
    from utils.defaults import DEFAULTS
    from utils.mesh_creation import GarmentCreator

    DEFAULTS.project_dir = str(args.contourcraft_root.resolve())
    DEFAULTS.data_root = str(args.data_root.resolve())
    DEFAULTS.aux_data = str((args.data_root / "aux_data").resolve())
    DEFAULTS.experiment_root = str((args.data_root / "experiments").resolve())

    creator = GarmentCreator(
        str(args.garment_dict.parent), str(args.body_model_root), "smplx",
        args.gender, collect_lbs=True, n_samples_lbs=0, coarse=False,
        swap_axes=False)
    trajectory = creator.relax_zeropos(
        args.garment_dict.stem, str(args.checkpoint), n_steps=args.steps)
    with args.trajectory_output.open("wb") as handle:
        pickle.dump(trajectory, handle)

    with args.garment_dict.open("rb") as handle:
        garment = pickle.load(handle)
    garment.setdefault("recovery_metadata", {})["relaxed"] = True
    garment["recovery_metadata"]["relaxation_steps"] = args.steps
    garment["recovery_metadata"]["relaxation_trajectory"] = str(
        args.trajectory_output.resolve())
    with args.garment_dict.open("wb") as handle:
        pickle.dump(garment, handle)
    print(f"Relaxed garment: {args.garment_dict}")
    print(f"Relaxation trajectory: {args.trajectory_output}")


if __name__ == "__main__":
    main()
