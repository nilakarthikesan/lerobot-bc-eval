"""Dataset EDA (DESIGN.md stage 1).

Usage: python scripts/01_explore_dataset.py --repo-id lerobot/pusht

TODO(M1):
  - load LeRobotDataset(repo_id)
  - print: n episodes, n frames, fps, camera keys, state/action dims,
    episode length distribution
  - save a few decoded frames + an action-trajectory plot to reports/
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/pusht")
    args = parser.parse_args()

    # TODO(M1): from lerobot.datasets.lerobot_dataset import LeRobotDataset
    raise NotImplementedError(f"EDA for {args.repo_id}")


if __name__ == "__main__":
    main()
