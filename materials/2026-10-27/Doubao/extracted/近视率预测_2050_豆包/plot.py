# -*- coding: utf-8 -*-
"""省份级近视率历史（1995-2022）与预测（2023-2050）可视化，300 dpi。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA_DIR = r"C:\Users\echo_\OneDrive\Desktop\06_myopia"
OUT_DIR = r"C:\Users\echo_\OneDrive\Desktop\近视率预测_2050"

# 中文字体
font_path = r"C:\Windows\Fonts\msyh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
plt.rcParams["axes.unicode_minus"] = False

# 历史：为保证跨年可比，只保留覆盖全部 8 个年级x城乡人群且处于全国波次（该年>=20个省份）的点
obs = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"))
obs = obs[obs["y"].notna()]
cells = obs.groupby(["province", "year"]).apply(
    lambda g: g.groupby(["grade_group", "residence"]).ngroups,
    include_groups=False).reset_index(name="n_cells")
wave_years = cells[cells["n_cells"] == 8].groupby("year").size()
wave_years = set(wave_years[wave_years >= 20].index)

hist = pd.read_csv(os.path.join(OUT_DIR, "province_history.csv")).merge(
    cells, on=["province", "year"], how="left")
hist = hist[(hist["n_cells"] == 8) & (hist["year"].isin(wave_years))].sort_values(
    ["province", "year"])
fut = pd.read_csv(os.path.join(OUT_DIR, "province_future.csv")).sort_values(["province", "year"])

provinces = sorted(set(hist["province"]) | set(fut["province"]))
# 31 色
cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
colors = [cmaps[i // 20](i % 20) for i in range(31)]
color_map = dict(zip(provinces, colors))

fig, ax = plt.subplots(figsize=(16, 9))

# 预测期浅灰底
ax.axvspan(2022.5, 2050.5, color="0.95", zorder=0)

for prov in provinces:
    h = hist[hist["province"] == prov]
    f = fut[fut["province"] == prov]
    c = color_map[prov]
    ax.plot(h["year"], h["cell"], color=c, lw=1.1, alpha=0.8, zorder=2)
    ax.plot(h["year"], h["cell"], "o", color=c, ms=2.8, alpha=0.8, zorder=2)
    ax.plot(f["year"], f["cell"], color=c, lw=1.2, ls="--", alpha=0.9, zorder=2)

# 全国省际等权均值，加粗黑线
hn = hist.groupby("year")["cell"].mean()
fn = fut.groupby("year")["cell"].mean()
ax.plot(hn.index, hn.values, color="black", lw=2.4, zorder=4,
        label="全国均值（历史） National mean (observed)")
ax.plot(fn.index, fn.values, color="black", lw=2.4, ls="--", zorder=4,
        label="全国均值（预测） National mean (forecast)")

ax.axvline(2022.5, color="grey", lw=1.2, ls=":", zorder=1)
ax.text(2022.8, 0.02, "预测区间 2023–2050", color="grey", fontsize=11, va="bottom")

# 图例：省份 + 线型说明
handles, labels = ax.get_legend_handles_labels()
for prov in provinces:
    handles.append(plt.Line2D([0], [0], color=color_map[prov], lw=1.6, label=prov))
    labels.append(prov)
ax.legend(handles, labels, bbox_to_anchor=(1.01, 1.0), loc="upper left",
          fontsize=8, ncol=2, frameon=False, handlelength=2.5)

ax.set_title("中国各省青少年近视患病率：历史值（1995–2022）与预测值（2023–2050）\n"
             "Provincial adolescent myopia prevalence in China — observed and forecast",
             fontsize=15, pad=12)
ax.set_xlabel("年份 Year", fontsize=12)
ax.set_ylabel("近视患病率 Prevalence", fontsize=12)
ax.set_xlim(1994, 2051)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_xticks(range(1995, 2051, 5))
ax.grid(alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)

# 线型说明文字
ax.text(0.01, 0.98, "实线 = 历史观测值（六次全国调查波次：1995/2000/2005/2010/2014/2019；按样本量加权合并检测方法，8个年级×城乡人群等权）\n"
                   "虚线 = 模型预测值（logit空间时空Ridge，十折时序交叉验证）；2020–2022年仅有零散小样本，图中不展示",
        transform=ax.transAxes, va="top", ha="left", fontsize=9, color="0.25")

fig.tight_layout()
out_path = os.path.join(OUT_DIR, "province_myopia_forecast.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("saved:", out_path)
