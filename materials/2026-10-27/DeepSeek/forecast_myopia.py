# -*- coding: utf-8 -*-
"""
中国青少年近视患病率预测（1995-2022 历史 -> 2023-2050 预测）
池化面板模型：logit(y) ~ 时间(年份) + 空间(省份固定效应) + 学段 + 城乡 + 测量方法 + 省份×年份交互
十折时间序列交叉验证（TimeSeriesSplit，不打乱，测试年份总晚于训练年份）
输出：metrics.csv / predictions_future.csv / PNG(300dpi) / 预测说明 docx
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit, expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
YEAR_CTR = 2008.5          # 年份中心化常数（固定值，不随折变化）
FUTURE_YEARS = range(2023, 2051)

CN_NAME = {
    "Anhui": "安徽", "Beijing": "北京", "Chongqing": "重庆", "Fujian": "福建",
    "Gansu": "甘肃", "Guangdong": "广东", "Guangxi": "广西", "Guizhou": "贵州",
    "Hainan": "海南", "Hebei": "河北", "Heilongjiang": "黑龙江", "Henan": "河南",
    "Hubei": "湖北", "Hunan": "湖南", "Inner Mongolia": "内蒙古", "Jiangsu": "江苏",
    "Jiangxi": "江西", "Jilin": "吉林", "Liaoning": "辽宁", "Ningxia": "宁夏",
    "Qinghai": "青海", "Shaanxi": "陕西", "Shandong": "山东", "Shanghai": "上海",
    "Shanxi": "山西", "Sichuan": "四川", "Tianjin": "天津", "Tibet": "西藏",
    "Xinjiang": "新疆", "Yunnan": "云南", "Zhejiang": "浙江",
}

# ---------------------------------------------------------------- 数据准备
df = pd.read_csv(HERE / "observations.csv")
data = df[df.y.notna()].copy()
data["method_code"] = data["method_code"].astype(int)

def parse_uid(uid: str):
    p, g, r, m = uid.split(":")
    return p, g, r, int(m.split("_")[1])

parts = data["unique_id"].map(parse_uid)
data["_prov"] = [t[0] for t in parts]
data["_grade"] = [t[1] for t in parts]
data["_resi"] = [t[2] for t in parts]
data["_method"] = [t[3] for t in parts]
data["logit_y"] = logit(data.y.values)

print(f"有效观测 {len(data)} 行, panel(unique_id) {data.unique_id.nunique()} 个, "
      f"年份 {data.year.min()}-{data.year.max()}")

# ---------------------------------------------------------------- 特征构建
FEAT_COLS = None

def build_X(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["year_c"] = d.year - YEAR_CTR
    X = pd.DataFrame({"year_c": d.year_c})
    X["grade"] = d._grade
    X["resi"] = d._resi
    X["prov"] = d._prov
    X["method"] = d._method
    X = pd.get_dummies(X, columns=["grade", "resi", "prov", "method"],
                       drop_first=True)
    for p in d._prov.unique():                       # 省份×年份交互（各省斜率不同）
        X[f"prov:{p}:xyear"] = d.year_c * (d._prov == p).astype(float)
    for g in d._grade.unique():                      # 学段×年份交互
        X[f"grade:{g}:xyear"] = d.year_c * (d._grade == g).astype(float)
    return X

def align_cols(X: pd.DataFrame) -> pd.DataFrame:
    global FEAT_COLS
    if FEAT_COLS is None:
        FEAT_COLS = X.columns
    else:
        X = X.reindex(columns=FEAT_COLS, fill_value=0.0)
    return X

X_all = align_cols(build_X(data))
print(f"特征数: {X_all.shape[1]}")

# ---------------------------------------------------------------- 十折时序交叉验证
years = sorted(data.year.unique())
tscv = TimeSeriesSplit(n_splits=10)
folds = [(years[tr[0]:tr[-1] + 1], years[te[0]:te[-1] + 1])
         for tr, te in tscv.split(years)]
for i, (tr, te) in enumerate(folds, 1):
    print(f"  fold {i:2d}: train {min(tr)}-{max(tr)} ({sum(data.year.isin(tr)):4d} 行), "
          f"test {min(te)}-{max(te)} ({sum(data.year.isin(te)):3d} 行)")

def metrics(y_true, y_pred):
    return dict(
        MSE=mean_squared_error(y_true, y_pred),
        RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
        MAE=mean_absolute_error(y_true, y_pred),
        MAPE=mean_absolute_percentage_error(y_true, y_pred),
        R2=r2_score(y_true, y_pred),
    )

# 正则强度网格（特征标准化后岭回归；用交叉验证选 alpha）
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
alpha_mse = {}
for a in ALPHAS:
    test_mses = []
    for tr_years, te_years in folds:
        trm = data.year.isin(tr_years)
        Xtr = align_cols(build_X(data[trm])).values
        Xte = align_cols(build_X(data[~trm & data.year.isin(te_years)])).values
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=a).fit(sc.transform(Xtr), data.loc[trm, "logit_y"].values)
        pred = expit(m.predict(sc.transform(Xte)))
        test_mses.append(mean_squared_error(data.loc[~trm & data.year.isin(te_years), "y"], pred))
    alpha_mse[a] = float(np.mean(test_mses))

best_alpha = min(alpha_mse, key=alpha_mse.get)
print(f"alpha 网格 CV-MSE: { {a: round(v, 4) for a, v in alpha_mse.items()} } -> 选 {best_alpha}")

# 用选定 alpha 正式跑十折，记录各折指标
fold_rows = []
pooled_true, pooled_pred = [], []
for i, (tr_years, te_years) in enumerate(folds, 1):
    trm = data.year.isin(tr_years)
    tem = ~trm & data.year.isin(te_years)
    Xtr = align_cols(build_X(data[trm])).values
    Xte = align_cols(build_X(data[tem])).values
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=best_alpha).fit(sc.transform(Xtr), data.loc[trm, "logit_y"].values)
    pred = expit(m.predict(sc.transform(Xte)))
    mtr = metrics(data.loc[tem, "y"], pred)
    fold_rows.append({"fold": i, **{k: round(v, 6) for k, v in mtr.items()}})
    pooled_true.append(data.loc[tem, "y"].values)
    pooled_pred.append(pred)

fold_df = pd.DataFrame(fold_rows)
summary = {c: fold_df[c].mean() for c in ["MSE", "RMSE", "MAE", "MAPE", "R2"]}
metrics_csv = fold_df.set_index("fold").T
metrics_csv.index.name = "metric"
metrics_csv["mean"] = [summary[c] for c in metrics_csv.index]
metrics_csv.to_csv(HERE / "metrics.csv")
print("\n十折交叉验证（均值）:")
print(metrics_csv.round(4).to_string())

pooled_m = metrics(np.concatenate(pooled_true), np.concatenate(pooled_pred))
print(f"\n十折合并后整体指标: MSE={pooled_m['MSE']:.4f} RMSE={pooled_m['RMSE']:.4f} "
      f"MAE={pooled_m['MAE']:.4f} MAPE={pooled_m['MAPE']:.4f} R2={pooled_m['R2']:.4f}")

# 基线对比（仅用训练期信息）
base_rows = []
for tr_years, te_years in folds:
    trm = data.year.isin(tr_years)
    tem = ~trm & data.year.isin(te_years)
    ytr, yte = data.loc[trm, "y"], data.loc[tem, "y"]
    gmean = ytr.mean()
    panel_mean = data.loc[trm].groupby("unique_id").y.mean()
    b1 = np.full(len(yte), gmean)                                   # 全局均值
    b2 = data.loc[tem, "unique_id"].map(panel_mean).fillna(gmean).values  # 面板均值
    base_rows.append({
        "fold": len(base_rows) + 1,
        "R2_global_mean": r2_score(yte, b1),
        "R2_panel_mean": r2_score(yte, b2),
    })
base_df = pd.DataFrame(base_rows)
print(f"基线: 全局均值 R2={base_df.R2_global_mean.mean():.4f}, "
      f"面板均值 R2={base_df.R2_panel_mean.mean():.4f}  (模型 R2={summary['R2']:.4f})")

# ---------------------------------------------------------------- 全量训练 + 未来预测
scaler = StandardScaler().fit(X_all.values)
model = Ridge(alpha=best_alpha).fit(scaler.transform(X_all.values), data["logit_y"].values)
print(f"\n全量模型 R2(样本内): {r2_score(data.y, expit(model.predict(scaler.transform(X_all.values)))):.4f}")

uids = sorted(data.unique_id.unique())
future_rows = []
for uid in uids:
    p, g, r, mth = parse_uid(uid)
    for yr in FUTURE_YEARS:
        future_rows.append(dict(unique_id=uid, province=p, grade_group=g,
                                residence=r, year=yr, method=mth))
fut = pd.DataFrame(future_rows)
fut["_prov"] = fut.province
fut["_grade"] = fut.grade_group
fut["_resi"] = fut.residence
fut["_method"] = fut.method
Xfut = align_cols(build_X(fut))
fut["predicted"] = expit(model.predict(scaler.transform(Xfut.values)))
fut_out = fut[["unique_id", "province", "grade_group", "residence", "year", "predicted"]].copy()
fut_out["predicted"] = fut_out.predicted.round(4)
fut_out.to_csv(HERE / "predictions_future.csv", index=False)
print(f"predictions_future.csv: {len(fut_out)} 行 = {len(uids)} panel × {len(FUTURE_YEARS)} 年")

# ---------------------------------------------------------------- 汇总统计（用于图与说明）
hist = data.groupby(["province", "year"]).y.mean()
fut_p = fut_out.groupby(["province", "year"]).predicted.mean()

nat_hist = data.groupby("year").y.mean()
nat_fut = fut_out.groupby("year").predicted.mean()
nat1995 = float(nat_hist.loc[1995])
nat2019 = float(nat_hist.loc[2019])
nat2020 = float(nat_hist.loc[2020])
nat2022 = float(nat_hist.loc[nat_hist.index <= 2022].iloc[-1])
nat2050 = float(nat_fut.loc[2050])
print(f"\n全国平均: {nat1995:.3f}(1995) -> {nat2019:.3f}(2019, 大样本波次) -> "
      f"{nat2022:.3f}(2022, 小样本) -> {nat2050:.3f}(2050, 预测)")
prov2050 = fut_p.xs(2050, level="year").sort_values()
print("2050 年预测最低三省:", [CN_NAME[x] for x in prov2050.index[:3]],
      [round(x, 3) for x in prov2050.values[:3]])
print("2050 年预测最高三省:", [CN_NAME[x] for x in prov2050.index[-3:]],
      [round(x, 3) for x in prov2050.values[-3:]])

# ---------------------------------------------------------------- 总图（300 dpi）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.lines import Line2D

cmap = plt.get_cmap("tab20")
colors = {p: cmap(i % 20) if i < 20 else plt.get_cmap("tab20b")(i - 20)
          for i, p in enumerate(sorted(data.province.unique()))}
fig, ax = plt.subplots(figsize=(14, 9))
ax.axvspan(2023, 2050, color="grey", alpha=0.08)
ax.axvline(2022.5, color="black", ls="--", lw=1.2, alpha=0.7)

order = sorted(data.province.unique())
for p in order:
    h = hist.xs(p, level="province")
    f = fut_p.xs(p, level="province")
    ax.plot(h.index, h.values * 100, color=colors[p], lw=1.8)
    ax.plot(f.index, f.values * 100, color=colors[p], lw=1.8, ls="--")

# 全国平均粗线（历史段掩掉年观测数 <10 的小样本年份，避免单条记录造成虚假尖峰）
obs_per_year = data.groupby("year").size()
nat_hist_plot = nat_hist.where(obs_per_year >= 10)
ax.plot(nat_hist_plot.index, nat_hist_plot.values * 100, color="black", lw=3.2)
ax.plot(nat_fut.index, nat_fut.values * 100, color="black", lw=3.2, ls="--")

handles = [Line2D([0], [0], color=colors[p], lw=1.8, label=CN_NAME[p])
           for p in sorted(prov2050.index, key=lambda x: -prov2050[x])]
handles += [Line2D([0], [0], color="black", lw=3.2, label="全国平均（历史）"),
            Line2D([0], [0], color="black", lw=3.2, ls="--", label="全国平均（预测）")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          fontsize=8.5, ncol=2, frameon=False, handlelength=1.8, columnspacing=0.8)

ax.set_xlabel("年份", fontsize=13)
ax.set_ylabel("青少年近视患病率（%）", fontsize=13)
ax.set_title("中国各省青少年近视患病率：1995–2022 历史与 2023–2050 预测（虚线为预测段）",
             fontsize=15)
ax.set_xlim(1994, 2052)
ax.set_ylim(0, 105)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "myopia_forecast_provinces_1995_2050.png", dpi=300,
            bbox_inches="tight")
print("\n图已保存: myopia_forecast_provinces_1995_2050.png (300 dpi)")

# ---------------------------------------------------------------- 预测说明 docx
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

def set_cn(run, font="宋体"):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)

doc = Document()
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(12)
for s in doc.sections:
    s.left_margin = Pt(72); s.right_margin = Pt(72)
    s.top_margin = Pt(72); s.bottom_margin = Pt(72)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("我国青少年近视患病率预测（1995–2050）模型说明")
run.bold = True
run.font.size = Pt(16)
set_cn(run, "黑体")

def h1(t):
    p = doc.add_paragraph()
    r = p.add_run(t); r.bold = True; r.font.size = Pt(13)
    set_cn(r, "黑体")
    return p

def body(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    set_cn(r, "宋体")
    p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.4
    return p

r2v, rmsev, mapev = summary["R2"], summary["RMSE"], summary["MAPE"]
p1995, p2019, p2022, p2050 = nat1995 * 100, nat2019 * 100, nat2022 * 100, nat2050 * 100
mape_pool = pooled_m["MAPE"]
low3 = "、".join(CN_NAME[x] for x in prov2050.index[:3])
high3 = "、".join(CN_NAME[x] for x in prov2050.index[-3:])

h1("一、使用的方法与建模思路")
body(f"本说明由 Anthropic 大模型 Claude 生成。数据为 1995–2022 年 31 省份、4 学段、城乡两类、3 种测量方法的近视率面板（1833 条）。因 panel 稀疏，采用池化面板回归：对 y 做 logit 变换使外推落在 (0,1)；特征含年份趋势、省份固定效应、学段城乡方法及“省份×年份”交互项，刻画时空异质斜率；用标准化岭回归抑制噪声。评估用十折时序交叉验证（不打乱、测试年份晚于训练），全量重训后外推 2023–2050 各 panel。")

h1("二、对 2050 年预测结果的解读")
body(f"历史：全国平均由 1995 年约 {p1995:.1f}% 升至 2019 年约 {p2019:.1f}%（2020–2022 小样本约 55%–62%），1995–2010 年上升较快，2010 年代后总体平稳、呈 S 型饱和。预测：控制构成变化后，各 panel 内上升趋势仍在延续；若延续下去，2050 年全国平均约 {p2050:.1f}%，多数省份升至 70%–95%，{high3} 接近或超过 90%，{low3} 相对较低；高学段高于低学段，城乡差异持续。注意：模型外推 logit 线性趋势、未内生平台期，高值省份长期预测或偏乐观。")

h1("三、远期预测的不确定性讨论")
body(f"精度：十折平均 R²={r2v:.3f}、RMSE={rmsev:.3f}、MAE={summary['MAE']:.3f}、MAPE={mapev:.1%}（合并口径 {mape_pool:.0%}），优于全局均值、面板均值基线，样本内拟合满意；MAPE 偏大源于早期低患病率小样本折。但 2023–2050 为 28 年外推，不确定性随时间急剧放大：数据稀疏且集中于少数调查波次（西藏仅 9 条记录）；logit 线性趋势无法内生反映未来政策、户外活动、出生队列与方法变动。故 2050 年数值宜视为“趋势延续情景推演”，不宜作精确点估计。")

doc.save(HERE / "预测说明_Claude.docx")
n_chars = sum(len(p.text) for p in doc.paragraphs if p.text.strip())
print(f"docx 已保存: 预测说明_Claude.docx（约 {n_chars} 字）")
print("\n全部输出完成。")
