#!/usr/bin/env python3
"""Convert ActorsHQ's raw per-frame SMPL-X capture (EasyMocap-style .json:
Rh/Th/poses/shapes/expression) into the per-frame .pkl format
utils.gaugar.GauGarConverter._convert_smplx_dir() expects (separate
betas/transl/global_orient/body_pose/jaw_pose/leye_pose/reye_pose/
left_hand_pose/right_hand_pose keys).

Run this ONCE per (subject, sequence) before import_official_gg_garment.py --
point that script's --gaugar-data-root at this script's --staging-root
instead of the raw ActorsHQ mount, e.g.:

    python convert_actorshq_smplx_to_gaugar.py \\
        --actorshq-root /mnt/d/actorshq_4x \\
        --subject Actor05 --sequence Sequence1 \\
        --staging-root /path/to/GG_projects/gaugar_data_staging

    python import_official_gg_garment.py \\
        --gaugar-data-root /path/to/GG_projects/gaugar_data_staging \\
        ... (same --subject/--gaugar-output-root/--output-root as before)

Layout assumption, confirmed by inspecting
/mnt/d/actorshq_4x/Actor05/Sequence1/smplx/000000.json:
    [{"id": 0, "Rh": [[rx,ry,rz]], "Th": [[tx,ty,tz]],
      "poses": [[0,0,0, <63 body_pose>, <3 jaw>, <3 leye>, <3 reye>,
                 <6 left_hand_pca>, <6 right_hand_pca>]],   # 87 total
      "expression": [[<10>]], "shapes": [[<10 betas>]]}]
The leading 3 zeros in "poses" are a root-rotation placeholder -- the real
root rotation is "Rh", used separately as global_orient. Only single-person
frames (one dict per file) are handled; multi-person files raise an error
rather than silently picking one.

Only converts ONE sequence at a time (not GauGarConverter._convert_smplx_subject's
"every subfolder under gaugar_data_root/<subject>/" loop) -- this avoids
having to also convert every OTHER sequence under Actor05/ (e.g. Sequence2)
just because it happens to sit next to the one you actually registered in
GG Stage 2.
"""

import argparse
import json
import pickle
from pathlib import Path

POSE_LAYOUT = {
    "body_pose": (3, 66),
    "jaw_pose": (66, 69),
    "leye_pose": (69, 72),
    "reye_pose": (72, 75),
    "left_hand_pose": (75, 81),
    "right_hand_pose": (81, 87),
}


def convert_frame(json_path: Path) -> dict:
    with json_path.open() as f:
        people = json.load(f)
    if len(people) != 1:
        raise ValueError(f"{json_path} has {len(people)} people; expected exactly 1. "
                          f"Multi-person frames aren't handled -- pick one explicitly if needed.")
    person = people[0]
    poses = person["poses"][0]
    if len(poses) != 87:
        raise ValueError(f"{json_path}: 'poses' has {len(poses)} values, expected 87 "
                          f"(3 zero root placeholder + 63 body + 3 jaw + 3 leye + 3 reye + "
                          f"6 left-hand-pca + 6 right-hand-pca). Layout may differ for this "
                          f"dataset export -- check manually before trusting this converter.")
    if any(abs(v) > 1e-6 for v in poses[:3]):
        raise ValueError(f"{json_path}: 'poses[0:3]' expected to be a zero root placeholder "
                          f"(root orientation comes from 'Rh' instead) but got {poses[:3]}. "
                          f"This dataset export may not match the assumed layout.")

    frame = dict(
        betas=person["shapes"][0],
        expression=person["expression"][0],
        transl=person["Th"][0],
        global_orient=person["Rh"][0],
    )
    for key, (start, end) in POSE_LAYOUT.items():
        frame[key] = poses[start:end]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actorshq-root", type=Path, required=True,
        help="e.g. /mnt/d/actorshq_4x")
    parser.add_argument("--subject", required=True, help="e.g. Actor05")
    parser.add_argument("--sequence", required=True, help="e.g. Sequence1")
    parser.add_argument("--staging-root", type=Path, required=True,
        help="Output acts as a synthetic gaugar_data_root: "
             "<staging-root>/<subject>/<sequence>/smplx/<frame>.pkl")
    args = parser.parse_args()

    src_dir = args.actorshq_root / args.subject / args.sequence / "smplx"
    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {src_dir}")

    out_dir = args.staging_root / args.subject / args.sequence / "smplx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(json_files)} frames from {src_dir}\n  -> {out_dir}")
    for json_path in json_files:
        frame = convert_frame(json_path)
        out_path = out_dir / f"{json_path.stem}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(frame, f)
        if json_files.index(json_path) % 500 == 0:
            print(f"  {json_path.name} -> {out_path.name}")

    print(f"Done. Point --gaugar-data-root at {args.staging_root} "
          f"(NOT {args.actorshq_root}) when running import_official_gg_garment.py / "
          f"GauGarConverter for {args.subject}.")


if __name__ == "__main__":
    main()
