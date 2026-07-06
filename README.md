<div align="center">

# VLA on a Budget

**ACT vs Diffusion Policy vs SmolVLA on one task, one GPU, one honest table.**

[![LeRobot](https://img.shields.io/badge/LeRobot-0.5.1-FF9D00)](https://github.com/huggingface/lerobot)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10_cu126-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX_4060_Laptop_8GB-76B900?logo=nvidia&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<img src="docs/results.png" width="95%" alt="Training curves and final benchmark"/>

</div>

## The question

Robot imitation learning has three generations of policy heads: deterministic regression
(**ACT**), generative diffusion (**Diffusion Policy**), and foundation-model VLAs with
flow-matching action experts (**SmolVLA** — the same recipe as π0, Gemini Robotics, GR00T).
What do they actually deliver on a consumer 8 GB GPU, trained locally, evaluated identically?

## The answer

| Policy | Params | Budget | Success (20 eps) | Max reward |
|---|---|---|---|---|
| ACT | ~52 M | 50 k × bs 32 | **0 %** | 0.47 |
| **Diffusion Policy** | ~263 M | 50 k × bs 32 | **35 %** | 0.67 |
| SmolVLA | ~450 M | 20 k × bs 8 | **0 %** | 0.26 |
| Diffusion (upstream pretrained, larger budget) | ~263 M | — | 70 % | 0.90 |

Task: **PushT** — push a T-shaped block onto a target; success = 95 % coverage.
Training data: the standard 206-episode human demonstration dataset (`lerobot/pusht`).

<div align="center">
<img src="docs/rollouts.png" width="95%" alt="ACT stalls halfway; diffusion completes"/>
</div>

## Three findings

**1 — Generative beats deterministic on multimodal demonstrations, 35-to-0.**
The demos contain many valid paths around the block. ACT regresses their mean and stalls
halfway (top strip) — its coverage plateaus at ~0.4 and never crosses the threshold.
Diffusion samples one coherent path and finishes (bottom strip). Identical data, identical
budget; only the action head differs.

**2 — A 450M VLA is not automatically better on a narrow task.**
SmolVLA underperformed both baselines here — and the *why* matters more than the number:
- batch 8 (VRAM-bound) means it saw **~10× fewer samples** than the baselines
- its frozen vision encoder is pretrained on real-world images; PushT's synthetic 96×96
  renders are far out of distribution
- its learning curve was still rising at cutoff (0.13 → 0.25)

Sample-matching the baselines would need ~200 k steps ≈ 40 GPU-hours — which *is* the
finding: **on consumer hardware, specialized policies are compute-optimal for narrow tasks.
VLA pretraining pays off in the multi-task, language-diverse settings it was built for** —
consistent with how the SmolVLA and π0 papers themselves position these models.

**3 — The tooling is younger than the models.**
Found along the way (Windows, LeRobot 0.5.1): installing `lerobot` silently replaces CUDA
torch with a CPU build; `--policy.path=<hub_id>` mangles repo ids into backslash paths;
pre-0.5 checkpoints need a bundled migration script; and `--rename_map` passes config
validation but is never applied to training batches (blocks cross-embodiment fine-tuning
from `smolvla_base`; upstream-reportable). Full list in [docs/NOTES.md](docs/NOTES.md).

## Reproduce

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install "lerobot[pusht,smolvla]"
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# B1 - ACT (~3.5 h)
lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
  --dataset.repo_id=lerobot/pusht --env.type=pusht --output_dir=runs/act `
  --steps=50000 --batch_size=32 --eval_freq=10000 --eval.n_episodes=10 --wandb.enable=false

# B2 - Diffusion (~6 h): same command with --policy.type=diffusion
# B3 - SmolVLA (~4.5 h): --policy.type=smolvla --steps=20000 --batch_size=8

# final benchmark, any run:
lerobot-eval --policy.path=runs/act/checkpoints/last/pretrained_model `
  --env.type=pusht --eval.n_episodes=20 --eval.batch_size=10 --policy.device=cuda

# figures
python make_figures.py
```

Interrupted? Every run resumes:
`lerobot-train --config_path=<out>/checkpoints/last/pretrained_model/train_config.json --resume=true`

## Limitations & next steps

- Single task, single seed, 20-episode evals — directional evidence, not a paper.
- SmolVLA deserves a sample-matched run (~200 k steps) and an unfrozen vision encoder ablation.
- The natural extension: LIBERO multi-task suites, where language conditioning should finally
  earn its parameters. (Blocked on Windows; planned on Linux/WSL.)

## Companion project

Perception-side sibling: [OpenVocab-4D](https://github.com/muhammadmahadazher/openvocab-4D) —
open-vocabulary 4D scene understanding (VGGT + SAM 3) on the same 8 GB laptop.

<div align="center">
<sub>Every number in this README was measured on one RTX 4060 Laptop GPU · MIT License</sub>
</div>
