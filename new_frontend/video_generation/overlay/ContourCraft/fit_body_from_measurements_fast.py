#!/usr/bin/env python3
"""Same as fit_body_from_measurements.py, plus early stopping on the betas fit.

Ablation counterpart to fit_body_from_measurements.py: identical math and
identical converged result, but the Adam loop stops once the loss stops
improving instead of always running the full --iterations count. Run this
in isolation (against the original script) to measure the early-stop win by
itself, before layering it with the process-merge optimization in
fit_and_simulate.py.

Height and weight only constrain 2 of the 10 SMPL-X shape parameters, so this
finds *a* plausible body matching those two measurements, not a unique or
medically precise reconstruction. Weight is estimated from closed-mesh volume
times an assumed average body density.

Usage (fit + patch a motion sequence in one go):
    python fit_body_from_measurements_fast.py \
      --body-model-root ccraft_data/aux_data/body_models \
      --gender female --height 1.68 --weight 60 \
      --output-betas ccraft_data/outputs/fixed_size_upper/my_betas.pkl \
      --input-sequence CMU/01/01_01_stageii.npz \
      --output-sequence ccraft_data/outputs/fixed_size_upper/01_01_stageii_mybody.npz

The resulting --output-sequence is a drop-in replacement for --sequence in
simulate_recovered_garment.py -- same motion, new body shape.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def body_model(body_model_root, gender, device):
    import smplx
    return smplx.create(str(body_model_root), model_type="smplx",
                        gender=gender, use_pca=False).to(device)


def mesh_volume(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    signed = torch.sum(v0 * torch.cross(v1, v2, dim=-1), dim=-1).sum() / 6.0
    return signed.abs()


def fit_betas(args, device):
    model = body_model(args.body_model_root, args.gender, device)
    faces = torch.as_tensor(model.faces.astype(np.int64), device=device)

    betas = torch.nn.Parameter(torch.zeros(1, 10, device=device))
    zero_pose = torch.zeros(1, 63, device=device)
    zero_orient = torch.zeros(1, 3, device=device)
    zero_transl = torch.zeros(1, 3, device=device)

    optimizer = torch.optim.Adam([betas], lr=args.learning_rate)
    best_loss = float("inf")
    steps_since_improvement = 0
    for step in range(args.iterations):
        optimizer.zero_grad()
        vertices = model(betas=betas, global_orient=zero_orient,
                         body_pose=zero_pose, transl=zero_transl).vertices[0]
        height = vertices[:, args.up_axis].max() - vertices[:, args.up_axis].min()
        weight = mesh_volume(vertices, faces) * args.body_density

        loss = (height - args.height) ** 2
        loss = loss + args.weight_loss_scale * (weight - args.weight) ** 2
        loss = loss + args.betas_prior * torch.mean(betas ** 2)
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == args.iterations - 1:
            print(f"step {step:04d}: height={height.item():.3f} m (target {args.height:.3f}), "
                  f"weight={weight.item():.1f} kg (target {args.weight:.1f})")

        loss_value = loss.item()
        if best_loss - loss_value > args.tol:
            best_loss = loss_value
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if step + 1 >= args.min_iterations and steps_since_improvement >= args.patience:
            print(f"step {step:04d}: converged early (no improvement > {args.tol} "
                  f"for {args.patience} steps), stopping "
                  f"({args.iterations - step - 1} iterations skipped)")
            break

    with torch.no_grad():
        vertices = model(betas=betas, global_orient=zero_orient,
                         body_pose=zero_pose, transl=zero_transl).vertices[0]
        final_height = (vertices[:, args.up_axis].max() - vertices[:, args.up_axis].min()).item()
        final_weight = (mesh_volume(vertices, faces) * args.body_density).item()

    return betas[0].detach().cpu().numpy(), final_height, final_weight


def patch_sequence_betas(input_sequence, output_sequence, betas):
    seq = dict(np.load(input_sequence, allow_pickle=True))
    original = np.asarray(seq["betas"])
    padded = np.zeros_like(original) if original.ndim == 1 else np.zeros(original.shape[-1])
    padded[:10] = betas
    seq["betas"] = padded
    np.savez(output_sequence, **seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"])
    parser.add_argument("--height", type=float, required=True, help="Target height in meters, e.g. 1.75")
    parser.add_argument("--weight", type=float, required=True, help="Target weight in kg, e.g. 70")
    parser.add_argument("--body-density", type=float, default=985.0,
        help="Assumed average human body density in kg/m^3.")
    parser.add_argument("--up-axis", type=int, default=1, choices=[0, 1, 2],
        help="Vertical coordinate axis of the SMPL-X rest pose (0=x, 1=y, 2=z). "
             "Default 1 (y) matches this repo's body models.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-loss-scale", type=float, default=1e-4,
        help="Relative weighting of the weight term (kg) vs. the height term (m).")
    parser.add_argument("--betas-prior", type=float, default=1e-3,
        help="L2 penalty pulling betas toward 0, keeps the shape plausible.")
    parser.add_argument("--patience", type=int, default=100,
        help="Stop once loss hasn't improved by more than --tol for this many "
             "consecutive steps. Set >= --iterations to disable early stopping.")
    parser.add_argument("--tol", type=float, default=1e-6,
        help="Minimum loss improvement to reset the patience counter.")
    parser.add_argument("--min-iterations", type=int, default=100,
        help="Never stop early before this many steps, so the fit has time to leave betas=0.")
    parser.add_argument("--output-betas", type=Path, required=True)

    parser.add_argument("--input-sequence", type=Path, default=None,
        help="Optional CMU/AMASS SMPL-X motion .npz whose own betas should be replaced.")
    parser.add_argument("--output-sequence", type=Path, default=None,
        help="Where to save the motion .npz with betas overwritten (required if --input-sequence is set).")
    args = parser.parse_args()

    if args.output_betas.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_betas}")
    if args.input_sequence is not None and args.output_sequence is None:
        raise ValueError("--output-sequence is required when --input-sequence is given")
    if args.output_sequence is not None and args.output_sequence.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_sequence}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas, achieved_height, achieved_weight = fit_betas(args, device)

    args.output_betas.parent.mkdir(parents=True, exist_ok=True)
    with args.output_betas.open("wb") as f:
        pickle.dump(dict(betas=betas, gender=args.gender,
                         target_height=args.height, target_weight=args.weight,
                         achieved_height=achieved_height, achieved_weight=achieved_weight), f)
    print(f"Wrote betas: {args.output_betas} "
          f"(height={achieved_height:.3f} m, weight={achieved_weight:.1f} kg)")

    if args.input_sequence is not None:
        args.output_sequence.parent.mkdir(parents=True, exist_ok=True)
        patch_sequence_betas(args.input_sequence, args.output_sequence, betas)
        print(f"Wrote motion sequence with overridden betas: {args.output_sequence}")


if __name__ == "__main__":
    main()
