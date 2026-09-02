#!/usr/bin/env python3
"""Prepare and simulate a fixed-size upper garment without patching ContourCraft."""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def load_betas(path: Path, index: int, key: str = "betas") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if key not in archive:
            raise KeyError(f"{path} does not contain key {key!r}; keys={archive.files}")
        value = archive[key]
    elif suffix in {".pkl", ".pickle"}:
        value = load_pickle(path)
        if isinstance(value, dict):
            if key not in value:
                raise KeyError(f"{path} does not contain key {key!r}; keys={list(value)}")
            value = value[key]
    else:
        raise ValueError(f"Unsupported beta file: {path}")

    value = np.asarray(value)
    if value.ndim == 1:
        betas = value
    elif value.ndim == 2:
        betas = value[index]
    else:
        raise ValueError(f"Expected beta array with 1 or 2 dimensions, got {value.shape}")
    if betas.shape[0] < 10:
        raise ValueError(f"Expected at least 10 beta values, got {betas.shape}")
    return np.asarray(betas[:10], dtype=np.float32)


def add_contourcraft_to_path(root: Path) -> None:
    root = root.resolve()
    if not (root / "datasets" / "ccraft.py").is_file():
        raise FileNotFoundError(f"Not a ContourCraft checkout: {root}")
    sys.path.insert(0, str(root))


def make_body_model(body_model_root: Path, model_type: str, gender: str):
    import smplx

    return smplx.create(
        str(body_model_root),
        model_type=model_type,
        gender=gender,
        use_pca=False,
    )


def _convex_hull_perimeter_xz(points_3d: "np.ndarray") -> float:
    """Perimeter of the 2D convex hull of points_3d projected onto the X-Z
    (horizontal) plane -- used as an "effective circumference" for both a
    garment's waistband loop and a body's cross-section at the same height.
    Robust to a non-planar/wavy ring (real waistband loops aren't perfectly
    circular) without needing an ordered ring traversal."""
    from scipy import spatial

    xz = np.asarray(points_3d)[:, [0, 2]]
    if len(xz) < 3:
        raise ValueError(f"Need >=3 points for a hull, got {len(xz)}")
    hull = spatial.ConvexHull(xz)
    return float(hull.area)  # scipy convention: 2D ConvexHull.area == perimeter


def apply_fit_aware_pin(output_dict: dict, target_vertices: "np.ndarray", args: argparse.Namespace) -> None:
    """Mutates output_dict in place: measures how loose this garment's
    waistband (or --pin-loop-index'd boundary loop) is against THIS run's
    target body at the same height, and only pins it (NodeType.HANDLE, same
    mechanism as pin_garment_waistband.py) if the fit is within
    --fit-ratio-threshold. Always records the measurement into
    fixed_size_metadata['waistband_fit_check'] regardless of the decision, so
    a run that ends up unpinned is still traceable.

    Deliberately independent of whatever pin state this garment dict already
    had (e.g. from --pin-waistband at Stage 1, which pins the SHARED
    garment_dict.pkl before relax purely to help that one-time relaxation
    converge -- see pin_garment_waistband.py's docstring). That decision was
    made before any target body was known and is baked into every future run
    regardless of fit; this one is made fresh per run, per target body, and
    resets node_type from scratch rather than layering onto the inherited
    state, so a garment that was pinned for relax can still come out unpinned
    here if it's a bad fit for this particular body.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pin_garment_waistband import find_boundary_loops
    from utils.common import NodeType

    template = np.asarray(output_dict["lbs"]["v"])  # this garment's locked, fixed-size rest shape
    faces = np.asarray(output_dict["faces"])
    loops = find_boundary_loops(faces)
    if not loops:
        raise RuntimeError(
            "--pin-waistband-if-snug: no open boundary loop found in this garment mesh -- "
            "it may be a closed mesh (e.g. a dress/tube) or a shirt/jacket whose collar "
            "isn't an open boundary. This flag only supports the boundary-loop case; skip "
            "it for garments where pin_garment_waistband.py itself would need --method "
            "joint-region.")

    loops_with_y = [(loop, float(template[loop, 1].mean())) for loop in loops]
    loops_with_y.sort(key=lambda item: -item[1])  # topmost (largest y) first
    chosen_index = args.pin_loop_index if args.pin_loop_index is not None else 0
    if chosen_index >= len(loops_with_y):
        raise ValueError(f"--pin-loop-index {chosen_index} out of range (found {len(loops_with_y)} loops)")
    chosen, mean_y = loops_with_y[chosen_index]
    if len(chosen) < args.pin_min_loop_vertices:
        raise RuntimeError(
            f"--pin-waistband-if-snug: chosen loop [{chosen_index}] has only {len(chosen)} "
            f"vertices (< --pin-min-loop-vertices {args.pin_min_loop_vertices}) -- doesn't "
            f"look like a real waistband.")

    garment_circumference = _convex_hull_perimeter_xz(template[chosen])

    target_np = np.asarray(target_vertices)
    band = args.pin_band_halfwidth
    for attempt in range(4):  # widen the height band if too sparse to hull
        band_mask = np.abs(target_np[:, 1] - mean_y) < band
        if band_mask.sum() >= 8:
            break
        band *= 2
    else:
        raise RuntimeError(
            f"--pin-waistband-if-snug: couldn't find enough target-body vertices near "
            f"y={mean_y:+.4f} even after widening the band to {band:.4f}m -- something's "
            f"off with the target body mesh.")
    body_circumference = _convex_hull_perimeter_xz(target_np[band_mask])

    fit_ratio = garment_circumference / body_circumference
    pinned = fit_ratio <= args.fit_ratio_threshold

    print(f"\n--pin-waistband-if-snug: loop [{chosen_index}] ({len(chosen)} verts, mean_y={mean_y:+.4f})")
    print(f"  garment waistband circumference (fixed, this size): {garment_circumference:.4f} m")
    print(f"  target body circumference at that height:           {body_circumference:.4f} m "
          f"(band ±{band:.4f}m, {int(band_mask.sum())} verts)")
    print(f"  fit ratio (garment / body) = {fit_ratio:.3f}  vs. threshold {args.fit_ratio_threshold:.3f}"
          f"  ->  {'PIN (snug enough)' if pinned else 'DO NOT PIN (too loose -- let it fall)'}")

    node_type = np.zeros_like(output_dict["rest_pos"][:, :1])
    if pinned:
        node_type[chosen] = NodeType.HANDLE
    output_dict["node_type"] = node_type.astype(np.int64)

    output_dict.setdefault("recovery_metadata", {})
    output_dict["recovery_metadata"]["pinned"] = bool(pinned)
    if pinned:
        output_dict["recovery_metadata"]["pinned_vertex_count"] = int(len(chosen))
        output_dict["recovery_metadata"]["pinned_method"] = "fit-aware-boundary-loop"
        output_dict["recovery_metadata"]["pinned_loop_mean_y"] = mean_y

    output_dict.setdefault("fixed_size_metadata", {})["waistband_fit_check"] = {
        "loop_index": chosen_index,
        "loop_vertex_count": int(len(chosen)),
        "loop_mean_y": mean_y,
        "garment_circumference_m": garment_circumference,
        "target_body_circumference_m": body_circumference,
        "fit_ratio": fit_ratio,
        "fit_ratio_threshold": args.fit_ratio_threshold,
        "pinned": bool(pinned),
    }


def apply_collision_radius_override(config, collision_radius, body_collision_radius) -> None:
    """EXPERIMENTAL, off by default. Mutates config.model.<name>.collision_radius /
    body_collision_radius in place -- the max-distance threshold (default 3e-2 m
    = 3cm in configs/contourcraft.yaml) for creating a garment-garment or
    garment-body "world edge" candidate for the GNN to reason about each step.
    This is THE parameter controlling how many contact edges get generated for
    a multi-garment outfit each simulate step, which is why multi-garment
    simulate costs far more than proportional to vertex count -- but unlike
    density/lame_mu/lame_lambda/bending_coeff (which ARE domain-randomized
    during training, see configs/contourcraft.yaml's material._min/_max
    ranges), collision_radius has no such range: it's a fixed constant the
    released checkpoint was trained with, never randomized. Overriding it here
    is mechanically safe (it's a plain float read at model construction, not
    part of the loaded state_dict, so nothing breaks), but semantically
    untested -- shrinking it to cut cost risks the network missing real
    contacts it would otherwise have caught, i.e. visible interpenetration
    that wasn't there at the trained value. Only use this if you're willing to
    visually check the render for artifacts every time you change it; it is
    NOT part of any --fast-fit/--bake-samples-style verified-safe default."""
    model_name = list(config.model.keys())[0]
    if collision_radius is not None:
        config.model[model_name].collision_radius = collision_radius
    if body_collision_radius is not None:
        config.model[model_name].body_collision_radius = body_collision_radius


def prepare(args: argparse.Namespace) -> None:
    add_contourcraft_to_path(args.contourcraft_root)
    import torch
    from scipy import spatial
    from smplx.lbs import blend_shapes

    source_dict = load_pickle(args.source_garment_dict)
    source_lbs = source_dict["lbs"]
    source_betas = load_betas(args.source_betas, args.source_beta_index, args.source_betas_key)
    target_betas = load_betas(args.target_sequence, args.target_beta_index, args.target_betas_key)

    source_gender = args.source_gender or source_dict.get("gender") or args.target_gender
    target_model = make_body_model(args.body_model_root, args.model_type, args.target_gender)

    source_v = torch.as_tensor(source_lbs["v"], dtype=torch.float32)
    source_shapedirs = torch.as_tensor(source_lbs["shapedirs"], dtype=torch.float32)
    beta_tensor = torch.as_tensor(source_betas[None], dtype=torch.float32)
    fixed_vertices = (source_v[None] + blend_shapes(beta_tensor, source_shapedirs))[0]

    target_beta_tensor = torch.as_tensor(target_betas[None], dtype=torch.float32)
    with torch.no_grad():
        target_vertices = target_model(betas=target_beta_tensor).vertices[0]

    tree = spatial.cKDTree(target_vertices.detach().cpu().numpy())
    distances, target_vertex_ids = tree.query(fixed_vertices.detach().cpu().numpy(), k=1)

    n_posedirs = target_model.posedirs.shape[0]
    target_posedirs = (
        target_model.posedirs.reshape(n_posedirs, -1, 3)[:, target_vertex_ids]
        .reshape(n_posedirs, -1)
        .detach()
        .cpu()
        .numpy()
    )
    target_weights = target_model.lbs_weights[target_vertex_ids].detach().cpu().numpy()

    output_dict = copy.deepcopy(source_dict)
    output_lbs = output_dict["lbs"]
    output_lbs["v"] = fixed_vertices.detach().cpu().numpy()
    output_lbs["shapedirs"] = np.zeros_like(source_lbs["shapedirs"], dtype=np.float32)
    output_lbs["posedirs"] = target_posedirs.astype(np.float32)
    output_lbs["lbs_weights"] = target_weights.astype(np.float32)
    output_lbs["f"] = np.asarray(source_lbs.get("f", source_dict["faces"]))
    output_dict["gender"] = args.target_gender
    output_dict["fixed_size_metadata"] = {
        "source_garment_dict": str(args.source_garment_dict.resolve()),
        "source_betas": source_betas,
        "target_betas": target_betas,
        "source_gender": source_gender,
        "target_gender": args.target_gender,
        "model_type": args.model_type,
        "binding": "nearest target shaped-body vertex",
        "mean_binding_distance": float(np.mean(distances)),
        "max_binding_distance": float(np.max(distances)),
    }

    if args.pin_waistband_if_snug:
        apply_fit_aware_pin(output_dict, target_vertices.detach().cpu().numpy(), args)

    dump_pickle(output_dict, args.output_garment_dict)
    print(f"Wrote fixed-size garment dict: {args.output_garment_dict}")
    print(
        "Binding distance: "
        f"mean={np.mean(distances):.6f} m, max={np.max(distances):.6f} m"
    )


def configure_defaults(contourcraft_root: Path, data_root: Path) -> None:
    from utils.defaults import DEFAULTS

    DEFAULTS.project_dir = str(contourcraft_root.resolve())
    DEFAULTS.data_root = str(data_root.resolve())
    DEFAULTS.aux_data = str((data_root / "aux_data").resolve())
    DEFAULTS.experiment_root = str((data_root / "experiments").resolve())


def simulate(args: argparse.Namespace) -> None:
    add_contourcraft_to_path(args.contourcraft_root)
    configure_defaults(args.contourcraft_root, args.ccraft_data_root)

    import torch
    from utils.arguments import load_params
    from utils.common import move2device
    from utils.validation import (
        apply_material_params,
        create_postcvpr_one_sequence_dataloader,
        _load_runner_from_state_dict,
    )

    garment_dict = args.garment_dict.resolve()
    if not garment_dict.is_file():
        raise FileNotFoundError(garment_dict)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.output_trajectory.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output_trajectory}")

    modules, config = load_params(
        "contourcraft", config_dir=str(args.contourcraft_root / "configs")
    )
    material = {
        "density": args.density,
        "lame_mu": args.lame_mu,
        "lame_lambda": args.lame_lambda,
        "bending_coeff": args.bending_coeff,
    }
    config = apply_material_params(config, material)
    if args.collision_radius is not None or args.body_collision_radius is not None:
        apply_collision_radius_override(config, args.collision_radius, args.body_collision_radius)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, runner = _load_runner_from_state_dict(modules, config, checkpoint_state)
    runner = runner.to(args.device)

    dataloader = create_postcvpr_one_sequence_dataloader(
        str(args.target_sequence.resolve()),
        garment_dict.stem,
        config="contourcraft",
        sequence_loader=args.sequence_loader,
        obstacle_dict_file=str(args.obstacle_dict.resolve()) if args.obstacle_dict else None,
        gender=args.target_gender,
        garment_dicts_dir=str(garment_dict.parent),
        body_model_root=str(args.body_model_root.resolve()),
        separate_arms=args.separate_arms,
    )
    sequence = move2device(next(iter(dataloader)), args.device)
    trajectory = dict(
        runner.valid_rollout(
            sequence,
            bare=True,
            n_steps=args.n_steps,
            safecheck=not args.disable_safecheck,
        )
    )

    trajectory["garment_names"] = [args.gaussian_garment_name]
    trajectory["fixed_size_metadata"] = load_pickle(garment_dict).get("fixed_size_metadata", {})
    dump_pickle(trajectory, args.output_trajectory)
    print(f"Wrote trajectory: {args.output_trajectory}")


def path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Create a new fixed-size garment dictionary")
    prep.add_argument("--contourcraft-root", type=path, required=True)
    prep.add_argument("--body-model-root", type=path, required=True)
    prep.add_argument("--source-garment-dict", type=path, required=True)
    prep.add_argument("--output-garment-dict", type=path, required=True)
    prep.add_argument("--source-betas", type=path, required=True)
    prep.add_argument("--source-betas-key", default="betas")
    prep.add_argument("--source-beta-index", type=int, default=0)
    prep.add_argument("--target-sequence", type=path, required=True)
    prep.add_argument("--target-betas-key", default="betas")
    prep.add_argument("--target-beta-index", type=int, default=0)
    prep.add_argument("--model-type", choices=("smpl", "smplx"), default="smpl")
    prep.add_argument("--source-gender", choices=("male", "female", "neutral"))
    prep.add_argument("--target-gender", choices=("male", "female", "neutral"), required=True)
    prep.add_argument("--pin-waistband-if-snug", action="store_true",
        help="After rebinding, measure this garment's waistband circumference against "
             "--target-gender's body circumference at the same height, and only pin the "
             "waistband (NodeType.HANDLE, same mechanism as pin_garment_waistband.py) if "
             "garment/body circumference <= --fit-ratio-threshold. Independent of and "
             "overrides whatever pin state this garment dict already had -- see "
             "apply_fit_aware_pin()'s docstring for why. Only supports the boundary-loop "
             "case (pants/skirts); no joint-region equivalent yet.")
    prep.add_argument("--fit-ratio-threshold", type=float, default=1.2,
        help="garment_circumference / body_circumference cutoff for --pin-waistband-if-snug "
             "-- at or under this, pin it (compensates for the rollout model's imperfect "
             "learned friction); above it, leave unpinned and let it fall. No principled "
             "'correct' value (this simulator has no real friction coefficient to derive it "
             "from) -- 1.2 is a starting point, tune by eye against rendered results.")
    prep.add_argument("--pin-loop-index", type=int, default=None,
        help="[--pin-waistband-if-snug] Override auto-selection: check the Nth boundary "
             "loop instead of the topmost one (0-based). Use --dry-run on "
             "pin_garment_waistband.py against the same garment dict to see candidates.")
    prep.add_argument("--pin-min-loop-vertices", type=int, default=5,
        help="[--pin-waistband-if-snug] Refuse to consider a candidate loop smaller than this.")
    prep.add_argument("--pin-band-halfwidth", type=float, default=0.02,
        help="[--pin-waistband-if-snug] Half-width (meters) of the height band sliced from "
             "the target body to estimate its circumference at the waistband's height. "
             "Widened automatically (doubled, up to 4 times) if the initial band catches "
             "fewer than 8 body vertices.")
    prep.set_defaults(func=prepare)

    run = subparsers.add_parser("simulate", help="Run official ContourCraft checkpoint")
    run.add_argument("--contourcraft-root", type=path, required=True)
    run.add_argument("--ccraft-data-root", type=path, required=True)
    run.add_argument("--body-model-root", type=path, required=True)
    run.add_argument("--garment-dict", type=path, required=True)
    run.add_argument("--target-sequence", type=path, required=True)
    run.add_argument("--checkpoint", type=path, required=True)
    run.add_argument("--output-trajectory", type=path, required=True)
    run.add_argument("--gaussian-garment-name", required=True)
    run.add_argument("--sequence-loader", default="cmu_npz_smpl")
    run.add_argument("--target-gender", choices=("male", "female", "neutral"), required=True)
    run.add_argument("--obstacle-dict", type=path)
    run.add_argument("--n-steps", type=int, default=200)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--separate-arms", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--disable-safecheck", action="store_true")
    run.add_argument("--density", type=float, default=0.20022)
    run.add_argument("--lame-mu", type=float, default=23600.0)
    run.add_argument("--lame-lambda", type=float, default=44400.0)
    run.add_argument("--bending-coeff", type=float, default=3.962e-5)
    run.add_argument("--collision-radius", type=float, default=None,
        help="EXPERIMENTAL, off by default (checkpoint's trained value, 3e-2m, is used "
             "unless set). Unlike the material params above, this is NOT domain-randomized "
             "during training -- see apply_collision_radius_override()'s docstring. Shrinking "
             "it cuts simulate cost but risks the network missing real contacts (visible "
             "interpenetration) since it's off the trained distribution. Check the render "
             "for artifacts every time you change this.")
    run.add_argument("--body-collision-radius", type=float, default=None,
        help="EXPERIMENTAL, same caveats as --collision-radius, for garment-body contacts "
             "instead of garment-garment/garment-self.")
    run.set_defaults(func=simulate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
