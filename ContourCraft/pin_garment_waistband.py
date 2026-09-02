#!/usr/bin/env python3
"""Auto-detect and pin a garment's anchor vertices (waistband for pants/skirts,
shoulder/collar region for shirts/jackets) so it doesn't slide down or drift
away from the body during relax/untangle/simulate.

Does not modify any official ContourCraft code -- it only calls the official,
unmodified utils.mesh_creation.add_pinned_verts() to write the pin, after
figuring out *which* vertices to pin itself.

Background: relax_recovered_garment.py, combine_garments_untangle.py, and
simulate_recovered_garment(_outfit).py all drive the garment with a physics
model that has no notion of friction/elastic waistbands -- only gravity,
collision, and cloth stretch/bend energy. A loose lower-body garment with no
pinned vertices has nothing to keep its waist near the body's hips, so it
free-falls for as long as the simulation runs (confirmed: e.g. a garment's
relax_trajectory.pkl showing its y-centroid dropping continuously across all
30 relax steps, never plateauing). Official ContourCraft garment dicts for
pants/skirts (e.g. ccraft_data/aux_data/garment_dicts/*/*.pkl for bottom_pants
etc.) all have ~40-60 vertices marked node_type==NodeType.HANDLE at the
waistband for exactly this reason (see GarmentImport.ipynb's "Add pinned
vertices" section: "fix the top ring of a skirt or pants to prevent it from
falling off the body").

Two selection methods, pick whichever fits the garment's shape:

--method boundary-loop (default): finds every open boundary loop in the
garment mesh (edges belonging to only one face -- a pants mesh typically has
one at the waist and one at each ankle; a skirt has one at the waist and one
at the hem), ranks them by mean height (using the canonical LBS template, the
space relax/untangle/simulate actually operate in), and pins the topmost one.
Works well for pants/skirts, where the waistband is a genuine open hole in
the mesh. Does NOT work for shirts/jackets: their anchor region (shoulder/
collar) usually isn't an open boundary at all (the collar is closed, sewn
fabric, not a hole), so "topmost open loop" ends up picking a sleeve cuff
instead -- confirmed on 00152_outer, where loop [0] (the auto-picked
"topmost") was a 26-vertex cuff, not the collar.

--method joint-region: for shirts/jackets. Each garment vertex's lbs_weights
(already in the garment dict, populated from the ORIGINAL recover step's
real-capture-based correspondence -- trustworthy at this pre-relax point,
before relax/untangle's own lbs re-derivation can corrupt it) says which
body joint it's mainly skinned to. This selects vertices whose dominant
joint is in --anchor-joints (default: spine3/neck/collars/shoulders -- the
upper-torso/shoulder joints), then narrows to only the top
--height-percentile of THOSE by height, so it isn't "pin the whole
upper torso" (which was ~30-58% of the mesh on the garments tested -- way
too much, would freeze most of the garment rigid) but "pin a thin band right
at the collar/shoulder line" (~3-5% of the mesh, same order of magnitude as
official ContourCraft's own pinned coat, which has 92/8033 pinned vertices).

Run this ONCE per garment, right after recover_resize_garment_bootstrap.py
and BEFORE relax_recovered_garment.py -- pinning only prevents drift for
simulation that happens after the pin is set; it cannot retroactively fix a
garment that has already relaxed/untangled into a bad shape.

Note: this pins the SHARED garment_dict.pkl before any target body is known,
purely to help THIS ONE-TIME relax converge -- the same pin decision then
gets reused by every future run regardless of which target body eventually
wears the garment. For the fixed-size pipeline specifically
(contourcraft_fixed_size_upper.py), there's now a second, independent
mechanism for the actual simulate-time question -- "does THIS run's target
body actually keep this garment's (locked) waistband up, or is it loose
enough that it should realistically fall?" -- driven by comparing the
garment's fixed waistband circumference against that target body's
circumference at the same height. See run_garment_pipeline_fixedsize.py's
--pin-waistband-if-snug / --fit-ratio-threshold (implemented in
contourcraft_fixed_size_upper.py's apply_fit_aware_pin()), and
run_garment_pipeline_fixedsize_README.md's "褲頭要不要 pin：兩套獨立機制"
section for the full writeup. The two are meant to be used together: this
script for relax stability, that one for per-target-body simulate realism.

Usage:
    python pin_garment_waistband.py \
        --contourcraft-root . \
        --garment-dict gg_outputs/00176_lower/garment_dict.pkl

    python pin_garment_waistband.py \
        --contourcraft-root . \
        --garment-dict gg_outputs/00152_outer/garment_dict.pkl \
        --method joint-region
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Standard SMPL(-X) joint order (utils/lbs.py, smplx.joint_names.JOINT_NAMES).
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
DEFAULT_ANCHOR_JOINTS = ["spine3", "neck", "left_collar", "right_collar",
                          "left_shoulder", "right_shoulder"]


def resolve_joint_names(names_or_indices):
    resolved = []
    for tok in names_or_indices:
        if tok.isdigit():
            idx = int(tok)
            if not (0 <= idx < len(JOINT_NAMES)):
                raise ValueError(f"Joint index {idx} out of range [0, {len(JOINT_NAMES)})")
            resolved.append(idx)
        else:
            if tok not in JOINT_INDEX:
                raise ValueError(f"Unknown joint name '{tok}'. Known: {JOINT_NAMES}")
            resolved.append(JOINT_INDEX[tok])
    return resolved


def find_joint_region(garment_dict, anchor_joint_indices, height_percentile):
    """Vertices whose dominant lbs joint is in anchor_joint_indices, narrowed to
    the top (100 - height_percentile)% by height -- computed separately per
    front/back x left/right QUADRANT, not as one global (or even one
    per-joint) threshold.

    A single global threshold is biased: e.g. front-collar vertices sit
    systematically higher than back-collar vertices in canonical pose, so one
    cutoff lets the higher side crowd out the lower side almost entirely
    (confirmed on 00152_outer: a global threshold picked 211 vertices from one
    side and only 55 from the other -- the under-selected side was the one
    still visibly flying loose in the render). Splitting per anchor JOINT
    alone doesn't fully fix it either -- within a single joint's own vertex
    set (e.g. spine3, which covers both the front chest and the back upper-
    torso), the same front/back height asymmetry still shows up (confirmed:
    spine3's top-10%-by-height split 49/39 front/back even though its whole
    vertex population was a balanced 414/465). Splitting by quadrant (using
    the whole garment's x/z medians as the left/right and front/back
    boundaries) keeps every quadrant proportionally represented regardless of
    which side happens to sit higher in canonical pose.

    Returns (selected_indices, joint_diagnostics, quadrant_diagnostics)."""
    weights = np.asarray(garment_dict["lbs"]["lbs_weights"])
    template = np.asarray(garment_dict["lbs"]["v"])
    dominant = weights.argmax(axis=1)

    joint_diagnostics = []
    for j in anchor_joint_indices:
        idx = np.where(dominant == j)[0]
        if len(idx) == 0:
            continue
        ys = template[idx, 1]
        joint_diagnostics.append((JOINT_NAMES[j] if j < len(JOINT_NAMES) else str(j),
                                   len(idx), float(ys.min()), float(ys.max())))

    region_idx = np.where(np.isin(dominant, anchor_joint_indices))[0]
    if len(region_idx) == 0:
        return region_idx, joint_diagnostics, []

    x_med, z_med = np.median(template[:, 0]), np.median(template[:, 2])
    region_pos = template[region_idx]
    quadrants = {
        "front-left":  (region_pos[:, 2] > z_med) & (region_pos[:, 0] <= x_med),
        "front-right": (region_pos[:, 2] > z_med) & (region_pos[:, 0] > x_med),
        "back-left":   (region_pos[:, 2] <= z_med) & (region_pos[:, 0] <= x_med),
        "back-right":  (region_pos[:, 2] <= z_med) & (region_pos[:, 0] > x_med),
    }

    selected_parts = []
    quadrant_diagnostics = []
    for name, qmask in quadrants.items():
        idx_in_quad = region_idx[qmask]
        if len(idx_in_quad) == 0:
            quadrant_diagnostics.append((name, 0, 0))
            continue
        ys = template[idx_in_quad, 1]
        y_thresh = np.percentile(ys, height_percentile)
        kept = idx_in_quad[ys >= y_thresh]
        selected_parts.append(kept)
        quadrant_diagnostics.append((name, len(idx_in_quad), len(kept)))

    selected = np.unique(np.concatenate(selected_parts)) if selected_parts else np.array([], dtype=np.int64)
    return selected, joint_diagnostics, quadrant_diagnostics


def find_boundary_loops(faces: np.ndarray) -> list:
    """Connected components of the mesh's open-boundary-edge graph.

    An edge shared by exactly one triangle is a boundary edge (the mesh has a
    hole/opening there). Grouping boundary edges by shared vertices gives one
    component per opening -- for a well-formed garment mesh each component is
    a simple ring around an opening (waist, hem, ankle cuff, sleeve cuff, ...).
    """
    edge_count = defaultdict(int)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[(min(u, v), max(u, v))] += 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]

    adjacency = defaultdict(set)
    for u, v in boundary_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    visited = set()
    loops = []
    for start in adjacency:
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        loops.append(np.array(sorted(component), dtype=np.int64))

    return loops


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contourcraft-root", type=Path, required=True)
    parser.add_argument("--garment-dict", type=Path, required=True,
        help="Garment dict .pkl to pin in place. Must be post-recover, pre-relax "
             "(recovery_metadata.relaxed must be False/absent).")
    parser.add_argument("--method", choices=["boundary-loop", "joint-region"], default="boundary-loop",
        help="boundary-loop (default): topmost open boundary loop, for pants/skirts. "
             "joint-region: top band of the shoulder/collar-dominant vertices, for "
             "shirts/jackets whose collar isn't an open mesh boundary. See docstring.")
    parser.add_argument("--loop-index", type=int, default=None,
        help="[boundary-loop] Override auto-selection: pin the Nth boundary loop instead "
             "of the topmost one (0-based, loops printed sorted topmost-first). Use "
             "--dry-run first to see the candidates.")
    parser.add_argument("--min-loop-vertices", type=int, default=5,
        help="[boundary-loop] Refuse to pin a candidate loop smaller than this -- likely "
             "not a real waistband/hem (e.g. a stray unwelded vertex pair).")
    parser.add_argument("--anchor-joints", nargs="+", default=DEFAULT_ANCHOR_JOINTS,
        help=f"[joint-region] Joint names or indices whose dominant-weighted vertices are "
             f"candidates (default: {' '.join(DEFAULT_ANCHOR_JOINTS)}).")
    parser.add_argument("--height-percentile", type=float, default=90.0,
        help="[joint-region] Keep only the top (100 - this)%% by height within the "
             "anchor-joint candidates (default 90 -> top 10%%).")
    parser.add_argument("--dry-run", action="store_true",
        help="Print detected boundary loops and exit without writing anything.")
    parser.add_argument("--force", action="store_true",
        help="Allow pinning even if the garment dict looks already relaxed or already "
             "pinned. Pinning after relax does NOT undo a fall that already happened "
             "during relaxation -- only use this if you know what you're doing.")
    args = parser.parse_args()

    with args.garment_dict.open("rb") as f:
        garment_dict = pickle.load(f)

    metadata = garment_dict.get("recovery_metadata", {})
    if metadata.get("relaxed") and not args.force:
        raise RuntimeError(
            f"{args.garment_dict} is already relaxed (recovery_metadata.relaxed=True). "
            f"Pinning now can't undo a fall that already happened during relaxation -- "
            f"delete this file (and its relax_trajectory.pkl) and rerun recovery so "
            f"pinning happens before relax, or pass --force if you specifically want to "
            f"pin an already-relaxed dict anyway (e.g. before re-relaxing it separately).")
    if metadata.get("pinned") and not args.force:
        raise RuntimeError(
            f"{args.garment_dict} is already marked pinned (recovery_metadata.pinned=True). "
            f"Pass --force to re-pin anyway.")

    template = np.asarray(garment_dict["lbs"]["v"])  # canonical space relax/untangle/simulate use

    if args.method == "boundary-loop":
        faces = np.asarray(garment_dict["faces"])
        loops = find_boundary_loops(faces)
        if not loops:
            raise RuntimeError(
                f"No open boundary found in {args.garment_dict} -- mesh appears closed "
                f"(e.g. a full tube/dress with no waist opening, or a shirt/jacket whose "
                f"collar is closed fabric rather than a hole). Try --method joint-region "
                f"instead, or pin vertices manually (see GarmentImport.ipynb).")

        loops_with_y = [(loop, float(template[loop, 1].mean())) for loop in loops]
        loops_with_y.sort(key=lambda item: -item[1])  # topmost (largest y) first

        print(f"Found {len(loops_with_y)} open boundary loop(s) in {args.garment_dict.name} "
              f"({len(template)} vertices total):")
        for i, (loop, mean_y) in enumerate(loops_with_y):
            y_min, y_max = template[loop, 1].min(), template[loop, 1].max()
            print(f"  [{i}] {len(loop):4d} verts   mean_y={mean_y:+.4f}   y_range=[{y_min:+.4f}, {y_max:+.4f}]")

        chosen_index = args.loop_index if args.loop_index is not None else 0
        if chosen_index >= len(loops_with_y):
            raise ValueError(f"--loop-index {chosen_index} out of range (found {len(loops_with_y)} loops)")
        chosen, chosen_mean_y = loops_with_y[chosen_index]

        if len(chosen) < args.min_loop_vertices:
            raise RuntimeError(
                f"Chosen loop [{chosen_index}] has only {len(chosen)} vertices "
                f"(< --min-loop-vertices {args.min_loop_vertices}) -- doesn't look like a real "
                f"waistband/hem. Inspect the printed candidates above and pass --loop-index "
                f"explicitly, or --min-loop-vertices 0 to override this check.")

        label = f"loop [{chosen_index}]"
        extra_metadata = {"pinned_method": "boundary-loop", "pinned_loop_mean_y": chosen_mean_y}

    else:  # joint-region
        anchor_joint_indices = resolve_joint_names(args.anchor_joints)
        chosen, joint_diag, quadrant_diag = find_joint_region(
            garment_dict, anchor_joint_indices, args.height_percentile)

        print(f"Anchor-joint candidates in {args.garment_dict.name} ({len(template)} vertices total, "
              f"anchor joints: {', '.join(args.anchor_joints)}):")
        for name, count, y_min, y_max in joint_diag:
            print(f"  {name:14s}: {count:5d} verts   y_range=[{y_min:+.4f}, {y_max:+.4f}]")
        print(f"  Per-quadrant top {100-args.height_percentile:.0f}% selection (front/back x left/right, "
              f"prevents one side crowding out the other):")
        for name, count, kept in quadrant_diag:
            print(f"    {name:12s}: {count:5d} candidates -> {kept} kept")

        if len(chosen) == 0:
            raise RuntimeError(
                f"No vertices found with dominant weight on any of {args.anchor_joints}. "
                f"Check --anchor-joints, or this garment may not cover that body region.")

        if len(chosen) < args.min_loop_vertices:
            raise RuntimeError(
                f"Selected region has only {len(chosen)} vertices "
                f"(< --min-loop-vertices {args.min_loop_vertices}) -- try a higher "
                f"--height-percentile denominator (lower value) to select more.")

        ys = template[chosen, 1]
        label = f"top {100 - args.height_percentile:.0f}% of anchor-joint region"
        print(f"\n  -> {len(chosen)} verts selected, y_range=[{ys.min():+.4f}, {ys.max():+.4f}]")
        extra_metadata = {"pinned_method": "joint-region", "pinned_anchor_joints": list(args.anchor_joints),
                           "pinned_height_percentile": args.height_percentile}

    print(f"\nPinning {label} ({len(chosen)} vertices)." +
          (" [--dry-run: not writing]" if args.dry_run else ""))

    if args.dry_run:
        return

    sys.path.insert(0, str(args.contourcraft_root.resolve()))
    from utils.mesh_creation import add_pinned_verts  # official, unmodified

    chosen_list = np.asarray(chosen).tolist()
    add_pinned_verts(args.garment_dict, chosen_list)

    # add_pinned_verts() only touches node_type -- reload/re-save to also stamp our
    # own metadata marker so callers (run_garment_pipeline.py, run_multigarment_
    # pipeline.py) can cache-check whether pinning has been applied.
    with args.garment_dict.open("rb") as f:
        garment_dict = pickle.load(f)
    garment_dict.setdefault("recovery_metadata", {})
    garment_dict["recovery_metadata"]["pinned"] = True
    garment_dict["recovery_metadata"]["pinned_vertex_count"] = int(len(chosen_list))
    garment_dict["recovery_metadata"].update(extra_metadata)
    with args.garment_dict.open("wb") as f:
        pickle.dump(garment_dict, f)

    print(f"Pinned {len(chosen_list)} vertices in {args.garment_dict}")


if __name__ == "__main__":
    main()
