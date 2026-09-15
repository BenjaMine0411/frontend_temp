#!/usr/bin/env python3
"""Quick, plain-grey-mannequin preview of a fitted body -- no garments, no
ContourCraft simulation, no Blender/Cycles. For preview_server.py's two-stage
flow: page 1 fits a body (fit_body_from_measurements_fast.py) and wants to
show the visitor "here's your body" in a couple seconds, well before any
garment or motion has been picked.

Deliberately NOT reusing render_lean.py/render_turntable.py -- both are built
around inference.py's Doll/AvatarNet machinery, which assumes at least one
garment (its Gaussian appearance model is per-garment) and always pays for a
Blender Cycles texture bake. The body mesh itself is plain SMPL-X geometry
(confirmed by reading inference.py: the body is a plain trimesh/open3d mesh,
never touched by AvatarNet), so in principle a lightweight standalone
renderer should suffice.

Tried Open3D's GPU offscreen renderer (`visualization.rendering.
OffscreenRenderer`) first, since open3d is already installed in this env --
confirmed by direct testing that it SEGFAULTS on `render_to_image()` on this
machine even with zero geometry, i.e. an environment/driver limitation
(EGL/DRI3 headless GL under WSL2), not a bug in how it was called. First
fallback was matplotlib's `mplot3d` (flat per-face shading) -- worked, but
visibly ugly: faceted/moire-looking skin (per-triangle flat color, no
anti-aliasing, and mplot3d's painter's-algorithm polygon sorting isn't a
real z-buffer). Replaced with `open3d.t.geometry.RaycastingScene` instead --
Open3D's CPU-only (Embree-based) ray-triangle intersection engine, no GL/EGL
context at all, so it can't hit the same environment failure. Casts one ray
per output pixel (2x supersampled, downsampled with Lanczos for anti-
aliasing), shades each hit with a per-vertex-normal-interpolated (Phong-
style, not flat-per-face) two-light diffuse term. Result: smooth, anti-
aliased, no visible triangulation -- much closer to a real studio product
shot -- while still ~30ms for the actual ray cast (mesh build dominates at
~0.7s, model loading). This is the version actually shipped; the matplotlib
path was fully replaced, not kept as a fallback.

Must run under the ccraft python env (needs smplx), same as
fit_body_from_measurements_fast.py.

Pose construction duplicates run_outfit_preview_pipeline.py's
make_display_pose_sequence() logic (not imported -- same "duplicate small
helpers" convention this project's scripts all follow) with one difference:
that function corrects an EXISTING motion clip's own frame-0 root_orient to
face the camera. Here there's no motion clip at all yet (page 1 runs before
any is picked), so a reference motion file's frame-0 root_orient is borrowed
purely as a known-good geometric anchor -- see make_display_pose_sequence's
own docstring for why a bare zero root_orient renders lying on its back
instead of standing (SMPL-X's raw canonical frame isn't upright in this
pipeline's convention; every real CMU/AMASS file carries a large near-
constant correction for this, which is what's being borrowed here).

Usage:
    python render_body_preview.py \\
      --betas-path gg_outputs_fixedsize/web_current_body/betas.pkl \\
      --output-image gg_outputs_fixedsize/web_current_body/body_preview.png
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

SCRIPT_DIR = Path(__file__).resolve().parent

# Deliberately NOT the 78-degree swing make_display_pose_sequence uses for
# the CLOTHED preview -- that value was tuned to keep a fitted sleeve from
# clipping into the torso during simulation (see that function's own
# docstring), which doesn't apply here since there's no garment at all.
# Measured directly (forward kinematics against a real body): 78 degrees
# leaves the upper arm only ~5.6 degrees off vertical -- reads as pressed
# flush against the torso on bare skin, which is exactly what a user
# reported ("手臂不要緊緊貼著身體") once there was no sleeve bulk to create
# the illusion of a gap. 55 degrees leaves it ~24 degrees off vertical --
# a natural "hands resting at the sides with a visible gap" look, still
# clearly "arms down", not a different pose. SMPL-X body_pose is 21 joints x
# 3 (axis-angle), flattened to 63; shoulders are joints 15/16 -> flat
# indices 45:48 / 48:51.
SHOULDER_SWING_DEG = 55.0
# Same fixed permute ccraft's dataloader (datasets/ccraft.py's swap_axes)
# applies to every vertex after root_orient -- reproduced here so this
# standalone renderer's output lands in the same "Y-up, facing +Z toward the
# camera" convention as the rest of this project's renders.
SWAP_AXES_ROTATION = Rotation.from_euler("x", -90, degrees=True)


def build_body_mesh(betas_path: Path, body_model_root: Path, reference_motion: Path):
    with betas_path.open("rb") as f:
        betas_data = pickle.load(f)
    betas = np.asarray(betas_data["betas"], dtype=np.float32)
    gender = betas_data.get("gender", "neutral")

    import smplx
    model = smplx.create(str(body_model_root), model_type="smplx", gender=gender, use_pca=False)

    # --- Pose: same construction as make_display_pose_sequence, minus the
    # need for a real motion sequence (see module docstring). ---
    body_pose = torch.zeros(1, 63)
    swing = np.deg2rad(SHOULDER_SWING_DEG)
    body_pose[0, 45:48] = torch.tensor([0.0, 0.0, -swing])  # left shoulder
    body_pose[0, 48:51] = torch.tensor([0.0, 0.0, swing])   # right shoulder, mirrored

    ref = dict(np.load(reference_motion, allow_pickle=True))
    ref_root_orient = ref["root_orient"][0]
    ref_trans = ref["trans"][0] if "trans" in ref and ref["trans"].ndim >= 1 else np.zeros(3)

    # Yaw-correct so the body faces the camera -- identical math to
    # make_display_pose_sequence (see its docstring for the full derivation);
    # duplicated here rather than imported.
    root_matrix = Rotation.from_rotvec(ref_root_orient).as_matrix()
    forward_final = SWAP_AXES_ROTATION.as_matrix() @ (root_matrix @ np.array([0.0, 0.0, 1.0]))
    yaw = -np.arctan2(forward_final[0], forward_final[2])
    yaw_world = Rotation.from_euler("y", yaw).as_matrix()
    yaw_raw = SWAP_AXES_ROTATION.as_matrix().T @ yaw_world @ SWAP_AXES_ROTATION.as_matrix()
    new_root_matrix = yaw_raw @ root_matrix
    global_orient = torch.tensor(Rotation.from_matrix(new_root_matrix).as_rotvec(),
                                  dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(betas=torch.tensor(betas).unsqueeze(0), global_orient=global_orient,
                        body_pose=body_pose, transl=torch.zeros(1, 3))
    vertices = output.vertices[0].cpu().numpy()
    faces = model.faces.astype(np.int32)

    # Apply the same post-root_orient permute as ccraft's dataloader, plus
    # re-center on the reference motion's own root translation so the body
    # sits at a plausible ground height instead of at SMPL-X's own pelvis-at-
    # origin convention.
    vertices = SWAP_AXES_ROTATION.apply(vertices + ref_trans)
    return vertices, faces


def _vertex_normals(vertices, faces):
    """Per-vertex normals = average of adjacent face normals, unnormalized
    sum first then normalize -- standard cheap approximation (no area
    weighting), good enough at this mesh density for smooth-looking shading."""
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    vn = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vn, faces[:, i], face_normals)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)


def render(vertices, faces, output_image: Path, width=480, height=640, supersample=2):
    """CPU ray-traced render via open3d.t.geometry.RaycastingScene -- see
    module docstring for why (Open3D's GPU offscreen renderer segfaults here;
    a first matplotlib-based attempt worked but looked faceted/aliased).

    Casts one ray per (supersampled) output pixel, shades each hit with a
    per-vertex-normal-interpolated normal (via the hit triangle's barycentric
    coordinates) rather than the triangle's own flat face normal -- this is
    what actually removes the faceted look, not just anti-aliasing.

    Camera framing + checkered floor deliberately match the real outfit-
    preview turntable's look (far-away full-body shot with headroom, floor
    grid) rather than a tight head-to-toe studio crop -- the user compared
    this page's body directly against a real generated preview and the tight
    crop read as "different pose" even though the joint angles are the same
    (confirmed numerically: the 78-degree shoulder swing below moves the
    elbow/wrist by the same distance regardless of framing) -- a small
    figure with visible floor and headroom is just a very different-feeling
    composition than a cropped close-up, which is what actually needed to
    match, not the underlying skeleton.
    """
    vertex_normals = _vertex_normals(vertices, faces)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh(o3c.Tensor(vertices.astype(np.float32)),
                                          o3c.Tensor(faces.astype(np.uint32)))
    scene.add_triangles(mesh_t)

    render_w, render_h = width * supersample, height * supersample
    center = (vertices.min(0) + vertices.max(0)) / 2
    floor_y = vertices[:, 1].min()
    body_height = vertices[:, 1].max() - floor_y  # Y is up
    # Aimed slightly above floor level (not straight at chest height) --
    # this, not the joint angles, is what makes it read as the same style of
    # shot as the real outfit-preview turntable. Distance/FOV tuned so the
    # figure fills most of the frame with a little headroom and floor space
    # left over -- an earlier farther-back version (2.9x/38deg) was
    # confirmed too small/background-heavy by the user; this is noticeably
    # tighter while still keeping the full body and some floor in frame.
    eye = np.array([center[0], floor_y + body_height * 0.95, center[2] + body_height * 1.8])
    look_at = np.array([center[0], floor_y + body_height * 0.55, center[2]])
    world_up = np.array([0.0, 1.0, 0.0])
    forward = look_at - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    fov_y = np.deg2rad(34.0)
    half_h = np.tan(fov_y / 2)
    half_w = half_h * (render_w / render_h)
    xs = np.linspace(-half_w, half_w, render_w)
    ys = np.linspace(half_h, -half_h, render_h)
    grid_x, grid_y = np.meshgrid(xs, ys)
    directions = grid_x[..., None] * right + grid_y[..., None] * cam_up + forward
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(eye, directions.shape)
    rays = np.concatenate([origins, directions], axis=-1).astype(np.float32)

    result = scene.cast_rays(o3c.Tensor(rays))
    hit = result["t_hit"].numpy() < np.inf
    tri_idx = result["primitive_ids"].numpy()[hit]
    bary_uv = result["primitive_uvs"].numpy()[hit]  # (u, v); third weight is 1-u-v

    hit_verts = faces[tri_idx]  # (N, 3) vertex indices of each hit triangle
    w0, w1, w2 = 1.0 - bary_uv[:, 0] - bary_uv[:, 1], bary_uv[:, 0], bary_uv[:, 1]
    interp_normal = (w0[:, None] * vertex_normals[hit_verts[:, 0]]
                      + w1[:, None] * vertex_normals[hit_verts[:, 1]]
                      + w2[:, None] * vertex_normals[hit_verts[:, 2]])
    interp_normal /= np.linalg.norm(interp_normal, axis=1, keepdims=True) + 1e-8

    # Two lights (key + softer fill) plus a flat ambient term -- avoids the
    # single-hard-light look, keeps the far side of the body legibly grey
    # instead of near-black.
    key_dir = np.array([0.35, 0.55, 0.75])
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.5, 0.2, 0.6])
    fill_dir /= np.linalg.norm(fill_dir)
    diffuse = (np.clip(interp_normal @ key_dir, 0, 1) * 0.7
               + np.clip(interp_normal @ fill_dir, 0, 1) * 0.25)
    shade = np.clip(0.28 + diffuse, 0, 1.15)  # 0.28 ambient floor
    base_color = np.array([0.74, 0.74, 0.78])  # same flat-grey mannequin look used elsewhere

    image = np.ones((render_h, render_w, 3), dtype=np.float32)  # white background/sky
    _draw_floor(image, hit, origins, directions, eye, floor_y, body_height)
    image[hit] = np.clip(base_color[None, :] * shade[:, None], 0, 1)

    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    if supersample != 1:
        pil_image = pil_image.resize((width, height), Image.LANCZOS)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_image)


def _draw_floor(image, body_hit, origins, directions, eye, floor_y, body_height,
                 cell_size=0.18):
    """Fills every pixel that missed the body but hits the y=floor_y ground
    plane with a light checker pattern (matching the checkered floor used
    throughout this project's real garment renders), fading to white/sky
    toward the horizon instead of a hard cutoff line. Mutates `image` in
    place for pixels it applies to; leaves body/sky pixels untouched."""
    t_plane = (floor_y - eye[1]) / directions[..., 1]
    floor_hit = (~body_hit) & (t_plane > 0)
    points = origins + directions * t_plane[..., None]
    checker = (np.floor(points[..., 0] / cell_size) + np.floor(points[..., 2] / cell_size)) % 2
    floor_color = np.where(checker[..., None] == 0, np.array([0.92, 0.92, 0.92]),
                            np.array([0.82, 0.82, 0.82]))
    distance = np.linalg.norm(points - eye, axis=-1)
    fade = np.clip(1.0 - (distance - body_height * 2.0) / (body_height * 3.0), 0.0, 1.0)
    floor_color = floor_color * fade[..., None] + 1.0 * (1 - fade[..., None])
    image[floor_hit] = floor_color[floor_hit]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--betas-path", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path,
        default=SCRIPT_DIR / "ccraft_data" / "aux_data" / "body_models")
    parser.add_argument("--reference-motion", type=Path,
        default=SCRIPT_DIR / "CMU" / "01" / "01_01_stageii.npz",
        help="Any real CMU/AMASS motion file -- only its frame-0 root_orient/trans are used, "
             "purely as a known-good geometric anchor (see module docstring).")
    args = parser.parse_args()

    vertices, faces = build_body_mesh(args.betas_path, args.body_model_root, args.reference_motion)
    render(vertices, faces, args.output_image)
    print(f"Wrote body preview: {args.output_image}")


if __name__ == "__main__":
    main()
