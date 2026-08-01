"""M5 interactive artifact: ALOHA end-effector traces in a Viser 3D scene.

Design: NOTES_VISUALIZATION.md §5 (stretch) / DESIGN.md §8. Joint-space arrays from M4
(outputs/mock_deploy/act_aloha/ep*.npz) are converted to 3D end-effector positions with
MuJoCo forward kinematics (gym-aloha's bimanual ViperX model), then served as an
interactive scene: ground-truth EE traces as lines, ACT's predicted 100-step chunks
branching off at anchor states, and a timestep slider driving a marker along the truth.

Usage:
  python scripts/05_viser_aloha.py [--episode 47] [--port 8080]
  -> open http://localhost:8080
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

MOCK_DIR = Path("outputs/mock_deploy/act_aloha")


def load_model() -> mujoco.MjModel:
    import gym_aloha

    xml = Path(gym_aloha.__file__).parent / "assets" / "bimanual_viperx_insertion.xml"
    return mujoco.MjModel.from_xml_path(str(xml))


def fk_ee(model: mujoco.MjModel, actions: np.ndarray) -> np.ndarray:
    """[T, 14] joint actions -> [T, 2, 3] left/right gripper positions (meters).

    Action layout: [left 6 joints, left grip, right 6 joints, right grip];
    qpos layout: left 0-5 (+fingers 6,7), right 8-13 (+fingers 14,15).
    Gripper opening does not move the gripper base, so finger qpos stays 0.
    """
    data = mujoco.MjData(model)
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vx300s_left/gripper_link")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vx300s_right/gripper_link")
    out = np.empty((len(actions), 2, 3))
    for t, a in enumerate(actions):
        data.qpos[0:6] = a[0:6]
        data.qpos[8:14] = a[7:13]
        mujoco.mj_forward(model, data)
        out[t, 0] = data.xpos[lid]
        out[t, 1] = data.xpos[rid]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=47)
    ap.add_argument("--n-anchors", type=int, default=5)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import matplotlib
    import viser

    d = dict(np.load(MOCK_DIR / f"ep{args.episode:03d}.npz"))
    model = load_model()

    gt_joints = d["gt"][:, 0, :]  # depth-0 action at every state = the GT trajectory
    gt_ee = fk_ee(model, gt_joints)  # [T, 2, 3]
    anchors = np.linspace(0, len(gt_joints) - 110, args.n_anchors).astype(int)
    chunks_ee = {int(a): fk_ee(model, d["pred"][a]) for a in anchors}  # each [100, 2, 3]

    # FK sanity (V6) before serving: finite, plausible workspace, smooth.
    assert np.isfinite(gt_ee).all(), "FK produced non-finite positions"
    step = np.linalg.norm(np.diff(gt_ee, axis=0), axis=-1)
    print(f"GT EE z range [{gt_ee[..., 2].min():.3f}, {gt_ee[..., 2].max():.3f}] m, "
          f"max per-frame step {step.max() * 100:.1f} cm")

    server = viser.ViserServer(port=args.port, label=f"ACT ALOHA ep{args.episode}")
    server.scene.add_grid("/table", width=1.2, height=0.8, position=(0.0, 0.0, 0.0))

    side_names = ("left", "right")
    side_colors = ((60, 180, 90), (40, 120, 220))  # GT: green-ish left, blue-ish right
    for s, (name, color) in enumerate(zip(side_names, side_colors)):
        server.scene.add_spline_catmull_rom(
            f"/gt/{name}", gt_ee[:, s], color=color, line_width=3.0)

    cmap = matplotlib.colormaps["autumn"]  # cm.get_cmap was removed in matplotlib 3.9
    for i, (a, ee) in enumerate(chunks_ee.items()):
        rgb = tuple(int(255 * c) for c in cmap(i / max(len(chunks_ee) - 1, 1))[:3])
        for s, name in enumerate(side_names):
            server.scene.add_spline_catmull_rom(
                f"/pred/anchor_{a}/{name}", ee[:, s], color=rgb, line_width=4.0)

    markers = [
        server.scene.add_icosphere(f"/marker/{name}", radius=0.012, color=color,
                                   position=tuple(gt_ee[0, s]))
        for s, (name, color) in enumerate(zip(side_names, side_colors))
    ]
    slider = server.gui.add_slider("timestep", min=0, max=len(gt_ee) - 1, step=1, initial_value=0)

    @slider.on_update
    def _(_) -> None:
        for s, marker in enumerate(markers):
            marker.position = tuple(gt_ee[slider.value, s])

    with server.gui.add_folder("about"):
        server.gui.add_markdown(
            f"**ACT / ALOHA insertion, held-out episode {args.episode}.**\n\n"
            "Green/blue lines: demonstrator's end-effector paths (MuJoCo FK of the "
            "14 GT joint actions). Warm-colored lines: ACT's predicted 100-step chunks "
            f"branching off at {args.n_anchors} anchor states (yellow = later anchor). "
            "Slide *timestep* to move the markers along the ground truth."
        )

    # Default camera fits the 1.2 m grid, which dwarfs the ~30 cm traces — frame them.
    center = gt_ee.reshape(-1, 3).mean(axis=0)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = tuple(center + np.array([0.45, -0.45, 0.35]))
        client.camera.look_at = tuple(center)

    # Viser silently falls back to the next free port if args.port is taken.
    print(f"Viser scene ready: http://localhost:{server.get_port()}")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
