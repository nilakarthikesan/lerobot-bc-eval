# LeRobot Behavior Cloning Evaluation

Training and evaluation scripts for ACT on simulated ALOHA insertion and Diffusion Policy on PushT, using [LeRobot](https://github.com/huggingface/lerobot). The project compares held-out action prediction with closed-loop simulation performance and visualizes policy errors across checkpoints.

My work in this repository covers dataset inspection, training configuration, checkpoint evaluation, prediction visualization, and occlusion-sensitivity analysis. The policy algorithms, datasets, and simulators come from the cited upstream projects.

## Documented results

The [report](reports/REPORT.md) records 50-episode simulation confirmations of 48% success for Diffusion Policy at 200K training steps and 20% for ACT at 20K steps. ACT's final 100K checkpoint scored 0% in a separate 10-episode screen. The ACT 20% result belongs to the early checkpoint.

Open-loop evaluation uses held-out demonstrations: PushT episodes 185–205 and ALOHA episodes 45–49. These observations come from the demonstrator's state distribution, while closed-loop simulation exposes each policy to states produced by its own actions. The two measurements answer different questions.

Occlusion maps describe how image perturbations affect predicted action chunks. Concentrated maps co-occur with declining ACT success in these runs; they do not establish that the encoder improved or that failures originate in the action decoder. See [NOTES_SALIENCY.md](NOTES_SALIENCY.md).

The repository includes figures and run documentation. Model checkpoints are hosted on Hugging Face; local `outputs/` evaluation arrays and logs are not committed. Retain those artifacts when reproducing a result. Results apply to these simulation tasks and checkpoints; physical robot performance was not evaluated.

## Setup and dataset exploration

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_explore_dataset.py --repo-id lerobot/pusht
```

For a local training smoke run on Apple Silicon:

```bash
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --steps=2000 --batch_size=32 --policy.device=mps \
  --output_dir=outputs/smoke_diffusion
```

Full training commands are in [scripts/02_train.sh](scripts/02_train.sh). Hugging Face Jobs modes require account access and paid compute. [Training notes](training/NOTES_TRAINING.md) document configurations and compatibility issues.

## Artifacts and entry points

- [Diffusion / PushT checkpoints](https://huggingface.co/nilakarthikesan/diffusion_pusht): 200K-step training run.
- [ACT / ALOHA checkpoints](https://huggingface.co/nilakarthikesan/act_aloha_insertion): 100K-step training run.
- [scripts/03_mock_deploy.py](scripts/03_mock_deploy.py): held-out open-loop replay.
- [scripts/03b_screen_checkpoints.sh](scripts/03b_screen_checkpoints.sh): closed-loop checkpoint screening.
- [scripts/04_visualize.py](scripts/04_visualize.py): saved prediction plots and overlays.
- [scripts/05_viser_aloha.py](scripts/05_viser_aloha.py): interactive end-effector visualization.
- [scripts/08_saliency.py](scripts/08_saliency.py): image occlusion probe.
- [reports/REPORT.md](reports/REPORT.md): results, limitations, references, and issues encountered.

The two policies use different tasks, observations, action spaces, and training schedules. Their success rates are not a controlled comparison of ACT against Diffusion Policy.
