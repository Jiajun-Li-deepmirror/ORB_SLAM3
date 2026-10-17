"""Keyframe z(t) small multiples: baseline vs Kalibr extrinsic vs Kalibr + corrected ground.
Reads the maps under results/validation/kalibr_extrinsic_20261015 (not in the repo)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from splg_slam.map.io import load_map

RESULTS = Path("/home/dm/slam_ws/results/validation/kalibr_extrinsic_20261015")
OUT = Path(__file__).resolve().parents[1] / "z_comparison.png"
NAMES = ["line_for", "line_for_2", "line_lat", "cali_line_for", "turn_for",
         "loop_for", "loop_for_2", "loop_lat", "cali_loop_for"]
LEVEL = {"line_for_2", "loop_for_2"}
SERIES = [("baseline", "基线（出厂外参）", "#2a78d6"),
          ("kalibr", "Kalibr 外参", "#eb6834"),
          ("kalibr_ground_n05_bias_tol015", "Kalibr + 地面约束（法向校正，重力容差 1.5%）", "#1baf7a")]

from matplotlib import font_manager
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # fc-list ":lang=zh" file
font_manager.fontManager.addfont(_cjk)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=_cjk).get_name(),
                     "axes.unicode_minus": False, "font.size": 9})
fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
fig.patch.set_facecolor("#fcfcfb")
for ax, name in zip(axes.flat, NAMES):
    ax.set_facecolor("#fcfcfb")
    for variant, label, color in SERIES:
        kfs = sorted(load_map(RESULTS / variant / name / "map.pkl").keyframes.values(), key=lambda k: k.timestamp_ns)
        t = np.array([k.timestamp_ns for k in kfs]) * 1e-9
        z = np.array([k.pose_wc()[2, 3] for k in kfs])
        ax.plot(t - t[0], z - z[0], color=color, lw=1.6, label=label)
    ax.set_title(name + ("（水平地面）" if name in LEVEL else ""), color="#0b0b0b", loc="left")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.axhline(0, color="#b5b4ad", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e")
for ax in axes[-1]:
    ax.set_xlabel("时间 (s)", color="#52514e")
for ax in axes[:, 0]:
    ax.set_ylabel("相对 z (m)", color="#52514e")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("关键帧高度随时间变化（相对第一帧，无真值）", y=0.955, color="#0b0b0b", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=110)
print(OUT)
