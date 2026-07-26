"""Mock deployment: open-loop replay on held-out episodes (DESIGN.md stage 3a).

Usage:
  python scripts/03_mock_deploy.py --checkpoint outputs/train/diffusion_pusht \
      --repo-id lerobot/pusht --test-episodes 184-205

TODO(M4):
  - load trained policy from checkpoint
  - for each held-out episode: step through observations, query policy,
    record predicted action sequence vs ground-truth demonstrator actions
  - metrics: per-step action MSE, EE position error over prediction horizon
  - dump results (npz/json) for 04_visualize.py
  - separately: closed-loop rollout via `lerobot-eval` (see DESIGN.md 3b)
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
