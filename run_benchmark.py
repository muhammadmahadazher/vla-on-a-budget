"""Reproduce the full VLA-on-a-Budget study with one command.

Trains ACT, Diffusion Policy, and SmolVLA on PushT with the exact budgets used
in the README, runs the identical 20-episode final benchmark on each, and
collects everything into results/summary.json.

    python run_benchmark.py                     # full study (~14 GPU-hours)
    python run_benchmark.py --policies act      # one policy
    python run_benchmark.py --eval-only         # re-run benchmarks on finished runs
    python run_benchmark.py --device mps        # macOS

Every run checkpoints; re-running the script resumes interrupted training
automatically. Works on Windows / Linux / macOS (device: cuda / mps / cpu).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# the exact budgets from the study
BUDGETS = {
    "act":       {"steps": 50_000, "batch_size": 32},
    "diffusion": {"steps": 50_000, "batch_size": 32},
    "smolvla":   {"steps": 20_000, "batch_size": 8},
}


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train(policy, device):
    out = RUNS / policy
    resume_cfg = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if resume_cfg.exists():
        print(f"[{policy}] resuming from existing checkpoint")
        run(["lerobot-train", f"--config_path={resume_cfg}", "--resume=true"])
        return
    b = BUDGETS[policy]
    run(["lerobot-train",
         f"--policy.type={policy}", f"--policy.device={device}",
         "--policy.push_to_hub=false",
         "--dataset.repo_id=lerobot/pusht", "--env.type=pusht",
         f"--output_dir={out}",
         f"--steps={b['steps']}", f"--batch_size={b['batch_size']}",
         "--log_freq=500", "--save_freq=10000",
         "--eval_freq=10000", "--eval.n_episodes=10", "--eval.batch_size=5",
         "--wandb.enable=false"])


def benchmark(policy, device):
    ckpt = RUNS / policy / "checkpoints" / "last" / "pretrained_model"
    if not ckpt.exists():
        print(f"[{policy}] no checkpoint at {ckpt} - train first")
        return None
    out = RESULTS / f"eval_{policy}"
    if out.exists():
        shutil.rmtree(out)
    run(["lerobot-eval",
         f"--policy.path={ckpt}", "--env.type=pusht",
         "--eval.n_episodes=20", "--eval.batch_size=5",
         f"--policy.device={device}", f"--output_dir={out}"])
    info = json.loads((out / "eval_info.json").read_text())
    agg = info.get("overall", info.get("aggregated", {}))
    return {"pc_success": agg.get("pc_success"),
            "avg_max_reward": agg.get("avg_max_reward"),
            "n_episodes": agg.get("n_episodes")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", nargs="+", default=["act", "diffusion", "smolvla"],
                   choices=list(BUDGETS))
    p.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    summary = {}
    for policy in args.policies:
        if not args.eval_only:
            train(policy, args.device)
        summary[policy] = benchmark(policy, args.device)
        print(f"[{policy}] {summary[policy]}")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Final benchmark (20 episodes each) ===")
    for policy, r in summary.items():
        if r:
            print(f"  {policy:10s} success {r['pc_success']:5.1f}%   "
                  f"max reward {r['avg_max_reward']:.3f}")
    print(f"\nsummary -> {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
