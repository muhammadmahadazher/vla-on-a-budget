# Engineering notes (Windows + LeRobot 0.5.1)

Everything that broke, in order, with fixes. Kept because the next person will hit the same walls.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `torch.cuda.is_available() == False` after installing lerobot | dependency resolver replaces `torch+cu126` with `torch+cpu` | `pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall` |
| 2 | `HFValidationError: 'lerobot\diffusion_pusht'` | `--policy.path` converts hub repo ids through `pathlib.Path` → backslashes on Windows | `snapshot_download()` the checkpoint, pass the local directory |
| 3 | `ProcessorMigrationError` loading `lerobot/diffusion_pusht` | pre-0.5 checkpoint format | run the bundled `lerobot/processor/migrate_policy_normalization.py` on a writable copy |
| 4 | `ValueError: 'policy.repo_id' argument missing` at train start | config validation wants a hub target even for local runs | `--policy.push_to_hub=false` |
| 5 | `ModuleNotFoundError: transformers` for SmolVLA | SmolVLA deps live in an extra | `pip install "lerobot[smolvla]"` (recheck torch afterwards — see #1) |
| 6 | `Feature mismatch ... camera1/2/3` fine-tuning `smolvla_base` on PushT | base checkpoint expects a 3-camera embodiment | `--rename_map` should fix it, but see #7 |
| 7 | `ValueError: All image features are missing from the batch` despite `--rename_map` | rename map is applied during config validation but **not** to training batches — upstream bug | train with `--policy.type=smolvla` instead (config derives from the dataset; VLM weights still load pretrained) |
| 8 | `No CUDA GPUs are available` mid-run, GPU gone from Device Manager | laptop OEM power management powers the dGPU off on battery | plug in the charger; the GPU re-enumerates |

## Eval protocol

- In-training: 10 episodes every 5–10 k steps (fast, noisy — ±15 % swings are normal)
- Final: 20 episodes, fixed protocol across all policies, `lerobot-eval`, seed-controlled env
- Success on PushT = 95 % target coverage; `max_reward` = best coverage reached in an episode

## Run inventory (local)

```
C:\Users\mahad\trackB_runs\
  train_act_pusht\            ACT 50k   (checkpoints 10k..50k + eval videos per checkpoint)
  train_diffusion_pusht\      Diffusion 50k
  train_smolvla_pusht_v2\     SmolVLA 20k
  eval_act_final\             20-episode benchmarks + videos
  eval_diffusion_final\
  eval_smolvla_final\
  eval_diffusion_pretrained\  70% reference run
```
