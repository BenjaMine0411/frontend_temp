#!/usr/bin/env python3
"""Recover an SMPL-X garment dictionary from a Gaussian-Garments resize rollout.

This is recover_resize_upper_garment.py with one change: --initial-fit is now
OPTIONAL. If you don't pass it, a neutral seed is bootstrapped automatically:
zero betas (average body shape), zero pose (T-pose), and a translation guess
taken from the centroid of the target body mesh for that frame. That's enough
for the Adam+LBFGS optimizer below to converge on its own -- the seed doesn't
need to be accurate, it just needs to exist with the right shapes.

Once you've run this once for a subject, its own --output-fit is a better seed
than the zero bootstrap for the *next* related trajectory (faster convergence,
less risk of hitting the RMSE cutoff) -- pass it back in via --initial-fit.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def load(path):
    with path.open("rb") as f:
        return pickle.load(f)


def save(value, path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)


def body_model(args, device):
    import smplx
    return smplx.create(str(args.body_model_root), model_type="smplx",
                        gender=args.gender, use_pca=False).to(device)


def infer_garment_name_from_filename(path):
    """Same convention inference.py uses for GG trajectories with garment_names == []:
    join the first two underscore-separated tokens, e.g. '00176_upper_Take1.pkl' -> '00176_upper'."""
    return "_".join(path.stem.split("_")[:2])


def garment_component(trajectory, name):
    names = trajectory["garment_names"]
    faces = np.asarray(trajectory["cloth_faces"])
    n = trajectory["pred"].shape[1]
    graph = [[] for _ in range(n)]
    for a, b, c in faces:
        graph[a] += [b, c]; graph[b] += [a, c]; graph[c] += [a, b]
    seen = np.zeros(n, bool)
    components = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, component = [start], []
        while stack:
            vertex = stack.pop(); component.append(vertex)
            for neighbor in graph[vertex]:
                if not seen[neighbor]:
                    seen[neighbor] = True; stack.append(neighbor)
        components.append(np.asarray(sorted(component), np.int64))
    components.sort(key=lambda ids: ids.min())
    if len(components) != len(names) or name not in names:
        raise ValueError(f"Cannot map components to garment names: {names}")
    ids = components[names.index(name)]
    if not np.array_equal(ids, np.arange(ids.min(), ids.max() + 1)):
        raise ValueError("Garment vertex IDs are not contiguous")
    local_faces = faces[np.all(np.isin(faces, ids), axis=1)] - ids.min()
    return trajectory["pred"][:, ids], local_faces


def bootstrap_seed(target):
    """Neutral SMPL-X seed: zero betas/pose, translation = target mesh centroid."""
    return dict(
        betas=np.zeros(10, dtype=np.float32),
        global_orient=np.zeros(3, dtype=np.float32),
        body_pose=np.zeros(63, dtype=np.float32),
        transl=target.mean(axis=0).astype(np.float32),
    )


def fit_body(args, trajectory, device):
    model = body_model(args, device)
    target_np = np.asarray(trajectory["obstacle"][args.frame], dtype=np.float32)
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)

    if args.initial_fit is not None:
        initial = load(args.initial_fit)
        seed_source = str(args.initial_fit.resolve())
    else:
        print("No --initial-fit given: bootstrapping a neutral zero-shape/zero-pose "
              "seed (translation = target mesh centroid) instead of requiring one.")
        initial = bootstrap_seed(target_np)
        seed_source = "bootstrapped_zero_seed"

    betas = torch.nn.Parameter(torch.as_tensor(initial["betas"], dtype=torch.float32,
                            device=device).reshape(1, 10))
    orient = torch.nn.Parameter(torch.as_tensor(initial["global_orient"],
                                dtype=torch.float32, device=device).reshape(1, 3))
    pose = torch.nn.Parameter(torch.as_tensor(initial["body_pose"],
                              dtype=torch.float32, device=device).reshape(1, 63))
    transl = torch.nn.Parameter(torch.as_tensor(initial["transl"],
                                dtype=torch.float32, device=device).reshape(1, 3))
    params = [betas, orient, pose, transl]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    for step in range(args.adam_iterations):
        optimizer.zero_grad()
        vertices = model(betas=betas, global_orient=orient,
                         body_pose=pose, transl=transl).vertices[0]
        data_loss = torch.mean((vertices - target) ** 2)
        (data_loss + args.pose_prior * torch.mean(pose ** 2)).backward()
        optimizer.step()
        if step % 200 == 0 or step == args.adam_iterations - 1:
            print(f"Adam {step:04d}: RMSE={data_loss.sqrt().item():.6f} m")
    optimizer = torch.optim.LBFGS(params, lr=0.5,
        max_iter=args.lbfgs_iterations, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        vertices = model(betas=betas, global_orient=orient,
                         body_pose=pose, transl=transl).vertices[0]
        loss = torch.mean((vertices - target) ** 2)
        (loss + args.pose_prior * torch.mean(pose ** 2)).backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        vertices = model(betas=betas, global_orient=orient,
                         body_pose=pose, transl=transl).vertices[0]
        rmse = torch.mean((vertices - target) ** 2).sqrt().item()
    fit = dict(betas=betas[0].detach().cpu().numpy(),
               global_orient=orient[0].detach().cpu().numpy(),
               body_pose=pose[0].detach().cpu().numpy(),
               transl=transl[0].detach().cpu().numpy(), gender=args.gender,
               source_trajectory=str(args.trajectory.resolve()),
               source_frame=args.frame, rmse=rmse,
               betas_initialization=seed_source,
               betas_frozen=False)
    save(fit, args.output_fit)
    print(f"Wrote fit: {args.output_fit}; RMSE={rmse:.6f} m")
    return fit


def recover(args, trajectory, fit):
    sys.path.insert(0, str(args.contourcraft_root.resolve()))
    from utils.mesh_creation import GarmentCreator, add_coarse_edges
    frames, faces = garment_component(trajectory, args.garment_name)
    posed = np.asarray(frames[args.frame], np.float32)
    params = {"betas": fit["betas"][None], "transl": fit["transl"][None],
              "global_orient": fit["global_orient"][None],
              "body_pose": fit["body_pose"][None]}
    for key, width in (("jaw_pose", 3), ("left_hand_pose", 45),
                       ("right_hand_pose", 45), ("leye_pose", 3), ("reye_pose", 3)):
        params[key] = np.zeros((1, width), np.float32)
    creator = GarmentCreator(str(args.output_garment_dict.parent),
        str(args.body_model_root), "smplx", args.gender, collect_lbs=True,
        n_samples_lbs=0, coarse=False, swap_axes=False)
    unposed = creator.unpose_garment(posed, params)
    result = creator._make_garment_dict_from_verts(
        {"vertices": unposed, "faces": faces}, vertices_canonical=posed)
    result["recovery_metadata"] = dict(
        source_trajectory=str(args.trajectory.resolve()), source_frame=args.frame,
        source_garment_name=args.garment_name,
        body_fit=str(args.output_fit.resolve()), body_fit_rmse=fit["rmse"],
        swap_axes=False, relaxed=False)
    if "center" not in result or "coarse_edges" not in result:
        result = add_coarse_edges(result, n_coarse_levels=4, approximate_center=True)
        result.setdefault("recovery_metadata", {})["coarse_edges_added_after_recovery"] = True
    save(result, args.output_garment_dict)
    print(f"Wrote garment dict: {args.output_garment_dict} "
          f"({len(posed)} vertices, {len(faces)} faces)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--garment-name", required=True)
    parser.add_argument("--initial-fit", type=Path, default=None,
        help="Optional seed body fit .pkl (betas/global_orient/body_pose/transl). "
             "If omitted, a neutral zero-shape/zero-pose seed is bootstrapped "
             "automatically from the target trajectory frame.")
    parser.add_argument("--output-fit", type=Path, required=True)
    parser.add_argument("--output-garment-dict", type=Path, required=True)
    parser.add_argument("--gender", default="female")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--adam-iterations", type=int, default=1200)
    parser.add_argument("--lbfgs-iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--pose-prior", type=float, default=1e-7)
    args = parser.parse_args()
    if args.output_fit.exists() or args.output_garment_dict.exists():
        raise FileExistsError("Refusing to overwrite an existing output")
    trajectory = load(args.trajectory)
    if not trajectory.get("garment_names"):
        trajectory["garment_names"] = [infer_garment_name_from_filename(args.trajectory)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = fit_body(args, trajectory, device)
    if fit["rmse"] > 0.005:
        raise RuntimeError(f"Body fit RMSE is too high: {fit['rmse']:.6f} m")
    recover(args, trajectory, fit)


if __name__ == "__main__":
    main()
