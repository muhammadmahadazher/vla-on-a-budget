"""Generate the Track B result figures: training curves + final bars, and rollout strips."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path(__file__).resolve().parent / "docs"
DOCS.mkdir(exist_ok=True)
RUNS = Path(r"C:\Users\mahad\trackB_runs")

# in-training eval trajectories (10-episode evals)
ACT = ([10, 20, 30, 40, 50], [0.271, 0.460, 0.280, 0.365, 0.463])
DIF = ([10, 30, 40, 50], [0.416, 0.623, 0.556, 0.600])
SMO = ([2.5, 5, 7.5, 10], [0.132, 0.154, 0.249, 0.203])  # 5k..20k rescaled to sample-equivalent
SMO_RAW = ([5, 10, 15, 20], [0.132, 0.154, 0.249, 0.203])

# final 20-episode benchmark
FINAL = [("ACT\n50k", 0.0, 0.472, "#d95f5f"),
         ("Diffusion\n50k", 35.0, 0.668, "#5a8fd9"),
         ("SmolVLA\n20k", 0.0, 0.258, "#8f6fc4"),
         ("Diffusion\npretrained", 70.0, 0.904, "#9db5cf")]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axes[0]
ax.plot(*ACT, "o-", color="#d95f5f", label="ACT (deterministic)")
ax.plot(*DIF, "s-", color="#5a8fd9", label="Diffusion (generative)")
ax.plot(*SMO_RAW, "^-", color="#8f6fc4", label="SmolVLA (flow matching)")
ax.set_xlabel("training steps (thousands)")
ax.set_ylabel("max reward (coverage), 10-episode evals")
ax.set_title("In-training evaluation")
ax.set_ylim(0, 1)
ax.axhline(0.95, color="gray", lw=0.8, ls="--")
ax.text(11, 0.965, "success threshold (95% coverage)", fontsize=8, color="gray")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1]
xs = np.arange(len(FINAL))
ax.bar(xs - 0.18, [f[1] for f in FINAL], width=0.36, color=[f[3] for f in FINAL],
       label="success rate (%)")
ax.bar(xs + 0.18, [f[2] * 100 for f in FINAL], width=0.36,
       color=[f[3] for f in FINAL], alpha=0.45, label="max reward (×100)")
ax.set_xticks(xs, [f[0] for f in FINAL])
ax.set_title("Final benchmark (20 episodes, identical protocol)")
ax.set_ylim(0, 100)
for i, f in enumerate(FINAL):
    ax.text(i - 0.18, f[1] + 1.5, f"{f[1]:.0f}%", ha="center", fontsize=10)
ax.legend()
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(DOCS / "results.png", dpi=140)
print("saved", DOCS / "results.png")

# rollout strips: ACT stalls halfway, diffusion completes
CLIPS = [("ACT — pushes halfway, never completes",
          RUNS / "eval_act_final/videos/pusht_0/eval_episode_0.mp4"),
         ("Diffusion — completes the task",
          RUNS / "eval_diffusion_final/videos/pusht_0/eval_episode_3.mp4")]
N = 6
fig, axes = plt.subplots(2, N, figsize=(14, 5))
for r, (title, path) in enumerate(CLIPS):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for c, fi in enumerate(np.linspace(0, total - 1, N).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        ax = axes[r][c]
        if ok:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        if c == 0:
            ax.set_title(title, loc="left", fontsize=11)
    cap.release()
fig.tight_layout()
fig.savefig(DOCS / "rollouts.png", dpi=140)
print("saved", DOCS / "rollouts.png")
