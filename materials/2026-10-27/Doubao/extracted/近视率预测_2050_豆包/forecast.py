# -*- coding: utf-8 -*-
"""
中国各省青少年近视患病率面板预测（1995-2022 训练，2023-2050 预测）
- 10 折滚动起点（expanding window）时间序列交叉验证，按年份严格切分，不随机打乱
- 模型：在 logit 空间建模，含面板OLS基线 / Ridge 线性趋势 / GBDT / Ridge+GBDT 残差混合
- 特征：时间趋势 + 面板/省份/人群历史统计 + 空间距离矩阵（MDS 坐标、反距离加权空间滞后）+ 日照排名
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------------
DATA_DIR = r"C:\Users\echo_\OneDrive\Desktop\06_myopia"
OUT_DIR = r"C:\Users\echo_\OneDrive\Desktop\近视率预测_2050"
os.makedirs(OUT_DIR, exist_ok=True)

CN2EN = {
    "北京": "Beijing", "天津": "Tianjin", "河北": "Hebei", "山西": "Shanxi",
    "内蒙古": "Inner Mongolia", "辽宁": "Liaoning", "吉林": "Jilin",
    "黑龙江": "Heilongjiang", "上海": "Shanghai", "江苏": "Jiangsu",
    "浙江": "Zhejiang", "安徽": "Anhui", "福建": "Fujian", "江西": "Jiangxi",
    "山东": "Shandong", "河南": "Henan", "湖北": "Hubei", "湖南": "Hunan",
    "广东": "Guangdong", "广西": "Guangxi", "海南": "Hainan", "重庆": "Chongqing",
    "四川": "Sichuan", "贵州": "Guizhou", "云南": "Yunnan", "西藏": "Tibet",
    "陕西": "Shaanxi", "甘肃": "Gansu", "青海": "Qinghai", "宁夏": "Ningxia",
    "新疆": "Xinjiang",
}
GRADE_ORD = {"1-3": 1, "4-6": 2, "7-9": 3, "10-12": 4}
EPS = 1e-4
RANDOM_STATE = 42
CV_YEARS = list(range(2013, 2023))   # 10 折，测试年依次为 2013..2022
FUTURE_YEARS = list(range(2023, 2051))

NUM_FEATS = [
    "t", "grade_ord", "urban", "sunshine_rank", "mds1", "mds2",
    "spatial_lag", "nn5_lag",
    "panel_n", "panel_age", "panel_last_z", "panel_mean_z", "panel_slope_z",
    "prov_mean_z", "prov_slope_z", "prov_last_z",
    "pd_mean_z", "pd_slope_z",
    "demo_mean_z", "demo_slope_z",
    "method_mean_z", "glob_mean_z", "glob_slope_z",
]

# 最终 Ridge 模型特征：时间趋势 + 层级历史统计 + 显式空间特征（经CV网格筛选）
RIDGE_FEATS = [
    "t", "grade_ord", "urban",
    "panel_mean_z", "panel_slope_z",
    "prov_mean_z", "prov_slope_z",
    "pd_mean_z", "pd_slope_z",
    "demo_mean_z", "demo_slope_z",
    "glob_mean_z", "glob_slope_z",
    "sunshine_rank", "mds1", "mds2", "spatial_lag", "nn5_lag",
]
RIDGE_ALPHA = 200.0


def logit(y):
    y = np.clip(np.asarray(y, float), EPS, 1 - EPS)
    return np.log(y / (1 - y))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------
# 1. 读取数据与空间信息
# ----------------------------------------------------------------------------
def load_data():
    obs = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"))
    d = obs[obs["y"].notna()].copy()
    d["t"] = d["year"] - 1995
    d["grade_ord"] = d["grade_group"].map(GRADE_ORD).astype(int)
    d["urban"] = (d["residence"] == "urban").astype(int)
    d["z"] = logit(d["y"].values)

    # 日照排名（中文省名 -> 英文）
    sun = pd.read_csv(os.path.join(DATA_DIR, "spatial", "sunshine-hours-rank.csv"))
    sun["province"] = sun["province"].map(CN2EN)
    sun = sun.rename(columns={"rank": "sunshine_rank"})

    # 距离矩阵（中文表头 -> 英文）
    dist = pd.read_csv(os.path.join(DATA_DIR, "spatial", "province_distance_matrix.csv"))
    dist = dist.rename(columns={"省份": "province"})
    dist["province"] = dist["province"].map(CN2EN)
    dist = dist.set_index("province")
    dist.columns = [CN2EN[c] for c in dist.columns]
    dist = dist.sort_index(axis=0).sort_index(axis=1)

    # 由距离矩阵做 MDS，得到 2 维空间坐标（固定地理信息，无数据泄露）
    mds = MDS(n_components=2, dissimilarity="precomputed", normalized_stress="auto",
              random_state=RANDOM_STATE)
    coords = mds.fit_transform(dist.values)
    coords = pd.DataFrame(coords, index=dist.index, columns=["mds1", "mds2"]).reset_index()

    static = sun.merge(coords, on="province", how="outer")
    return d, dist, static


# ----------------------------------------------------------------------------
# 2. 快照统计（仅用 cutoff 年之前的训练数据，杜绝未来信息泄露）
# ----------------------------------------------------------------------------
def ols_by_group(df, keys, col_t="t", col_z="z"):
    """按组分批做 z 对 t 的 OLS，返回 mean_z / slope_z / intercept_z / n。"""
    tmp = df[keys + [col_t, col_z]].copy()
    tmp["tt"] = tmp[col_t] ** 2
    tmp["tz"] = tmp[col_t] * tmp[col_z]
    g = tmp.groupby(keys, observed=True)
    agg = g.agg(mean_t=(col_t, "mean"), mean_z=(col_z, "mean"),
                st=(col_t, "sum"), sz=(col_z, "sum"),
                stt=("tt", "sum"), stz=("tz", "sum"), n=(col_z, "size")).reset_index()
    sxx = agg["stt"] - agg["st"] ** 2 / agg["n"]
    sxy = agg["stz"] - agg["st"] * agg["sz"] / agg["n"]
    agg["slope_z"] = np.where(sxx > 1e-9, sxy / sxx.replace(0, np.nan), np.nan)
    agg["intercept_z"] = agg["mean_z"] - agg["slope_z"] * agg["mean_t"]
    return agg[keys + ["n", "mean_z", "slope_z", "intercept_z"]]


def build_snapshot(train, dist, static):
    snap = {}
    # 全局
    t, z = train["t"].values, train["z"].values
    A = np.vstack([np.ones_like(t), t]).T
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    snap["glob"] = {"mean_z": z.mean(), "slope_z": coef[1], "intercept_z": coef[0]}

    # 各层级
    snap["panel"] = ols_by_group(train, ["unique_id"])
    last = train.sort_values("year").groupby("unique_id", observed=True).agg(
        last_year=("year", "max"), last_z=("z", "last")).reset_index()
    snap["panel"] = snap["panel"].merge(last, on="unique_id", how="left")

    snap["prov"] = ols_by_group(train, ["province"])
    prov_last = train.sort_values("year").groupby(["province", "year"], observed=True)["z"].mean().reset_index()
    prov_last = prov_last.sort_values("year").groupby("province", observed=True).tail(1)[["province", "z"]]
    prov_last = prov_last.rename(columns={"z": "last_z"})
    snap["prov"] = snap["prov"].merge(prov_last, on="province", how="left")

    snap["pd"] = ols_by_group(train, ["province", "grade_group", "residence"])
    snap["demo"] = ols_by_group(train, ["grade_group", "residence"])
    snap["method"] = ols_by_group(train, ["method_code"])

    # 空间滞后：反距离加权的外省均值 + 最近5省均值
    pv = snap["prov"].set_index("province")
    provs = dist.index
    mean_z = pv["mean_z"].reindex(provs)
    D = dist.loc[provs, provs].values
    W = 1.0 / (D + 50.0) ** 2
    np.fill_diagonal(W, 0.0)
    # 缺失省份均值用全局均值补
    mz = mean_z.fillna(snap["glob"]["mean_z"]).values
    lag = (W * mz[None, :]).sum(axis=1) / W.sum(axis=1)
    nn5 = np.vstack([mz[np.argsort(D[i])[1:6]] for i in range(len(provs))]).mean(axis=1)
    spatial = pd.DataFrame({"province": provs, "spatial_lag": lag, "nn5_lag": nn5})
    static2 = static.merge(spatial, on="province", how="left")

    # 静态变量全局缺失填均值
    for c in ["sunshine_rank", "mds1", "mds2", "spatial_lag", "nn5_lag"]:
        static2[c] = static2[c].fillna(static2[c].mean())
    return snap, static2


def build_features(frame, snap, static2):
    """frame: 含 province/grade_group/residence/method_code/unique_id/year 的批次。"""
    B = frame.copy()
    B["t"] = B["year"] - 1995
    B["grade_ord"] = B["grade_group"].map(GRADE_ORD).astype(int)
    B["urban"] = (B["residence"] == "urban").astype(int)
    B = B.merge(static2, on="province", how="left")

    # 面板统计
    pa = snap["panel"].rename(columns={"n": "panel_n", "mean_z": "panel_mean_z",
                                       "slope_z": "panel_slope_z",
                                       "intercept_z": "panel_int_z",
                                       "last_year": "panel_last_year",
                                       "last_z": "panel_last_z"})
    B = B.merge(pa, on="unique_id", how="left")
    B["panel_n"] = B["panel_n"].fillna(0)
    B["panel_age"] = np.where(B["panel_last_year"].notna(),
                              B["year"] - B["panel_last_year"], 30)

    # 省份
    pr = snap["prov"].rename(columns={"n": "prov_n", "mean_z": "prov_mean_z",
                                      "slope_z": "prov_slope_z",
                                      "last_z": "prov_last_z"})[
        ["province", "prov_mean_z", "prov_slope_z", "prov_last_z"]]
    B = B.merge(pr, on="province", how="left")

    # 省份 x 人群（年级x城乡）
    pd_ = snap["pd"].rename(columns={"n": "pd_n", "mean_z": "pd_mean_z",
                                     "slope_z": "pd_slope_z"})[
        ["province", "grade_group", "residence", "pd_mean_z", "pd_slope_z"]]
    B = B.merge(pd_, on=["province", "grade_group", "residence"], how="left")

    # 人群
    dm = snap["demo"].rename(columns={"mean_z": "demo_mean_z",
                                      "slope_z": "demo_slope_z"})[
        ["grade_group", "residence", "demo_mean_z", "demo_slope_z"]]
    B = B.merge(dm, on=["grade_group", "residence"], how="left")

    # 方法
    mm = snap["method"].rename(columns={"mean_z": "method_mean_z"})[
        ["method_code", "method_mean_z"]]
    B = B.merge(mm, on="method_code", how="left")

    g = snap["glob"]
    B["glob_mean_z"] = g["mean_z"]
    B["glob_slope_z"] = g["slope_z"]

    # 层级回退（冷启动面板）
    B["pd_mean_z"] = B["pd_mean_z"].fillna(B["prov_mean_z"])
    B["pd_slope_z"] = B["pd_slope_z"].fillna(B["prov_slope_z"])
    B["panel_mean_z"] = B["panel_mean_z"].fillna(B["pd_mean_z"])
    B["panel_last_z"] = B["panel_last_z"].fillna(B["panel_mean_z"])
    B["panel_slope_z"] = B["panel_slope_z"].fillna(B["pd_slope_z"])
    for c in ["prov_mean_z", "prov_slope_z", "prov_last_z", "pd_mean_z",
              "pd_slope_z", "demo_mean_z", "demo_slope_z", "method_mean_z",
              "panel_mean_z", "panel_last_z", "panel_slope_z"]:
        B[c] = B[c].fillna(g["mean_z"] if "mean" in c else g["slope_z"])
    return B


# ----------------------------------------------------------------------------
# 3. 模型
# ----------------------------------------------------------------------------
def fit_ridge(Xtr, ztr):
    """Ridge：数值特征标准化 + 类别哑变量 + 省份x时间趋势（各省独立 logit 趋势）。"""
    num = Xtr[RIDGE_FEATS]
    scaler = StandardScaler().fit(num.values)
    dummies = pd.get_dummies(
        Xtr[["province", "grade_group", "residence", "method_code"]].astype("category"),
        drop_first=False)
    pt = pd.get_dummies(Xtr["province"].astype("category")).multiply(Xtr["t"].values, axis=0)
    X = np.hstack([scaler.transform(num.values),
                   dummies.values.astype(float), pt.values.astype(float)])
    model = Ridge(alpha=RIDGE_ALPHA).fit(X, ztr)
    return ("ridge", scaler, dummies.columns, pt.columns, model)


def pred_ridge(fit_, X):
    _, scaler, dcols, pcols, model = fit_
    num = X[RIDGE_FEATS]
    dummies = pd.get_dummies(
        X[["province", "grade_group", "residence", "method_code"]].astype("category"),
        columns=["province", "grade_group", "residence", "method_code"])
    dummies = dummies.reindex(columns=dcols, fill_value=0)
    pt = pd.get_dummies(X["province"].astype("category")).reindex(columns=pcols, fill_value=0)
    pt = pt.multiply(X["t"].values, axis=0)
    Xm = np.hstack([scaler.transform(num.values),
                    dummies.values.astype(float), pt.values.astype(float)])
    return model.predict(Xm)


def fit_gbdt(Xtr, target):
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=15, l2_regularization=1.0,
        random_state=RANDOM_STATE)
    model.fit(Xtr[NUM_FEATS].values, target)
    return model


def pred_gbdt(model, X):
    return model.predict(X[NUM_FEATS].values)


def baseline_pred(B, snap):
    """面板层级 OLS 基线（logit 空间）。"""
    g = snap["glob"]
    slope = B["panel_slope_z"].fillna(B["pd_slope_z"]).fillna(B["prov_slope_z"])
    slope = slope.fillna(B["demo_slope_z"]).fillna(g["slope_z"])
    # 有面板截距优先，其次省份x人群，再省份，再全局
    a = B["panel_int_z"]
    a = a.fillna(B["pd_mean_z"] - slope * B["t"])  # 以均值点近似锚定
    a = a.fillna(B["prov_mean_z"] - slope * B["t"])
    a = a.fillna(g["intercept_z"])
    return a + slope * B["t"]


def fit_all(train, dist, static):
    snap, static2 = build_snapshot(train, dist, static)
    Xtr = build_features(train, snap, static2)
    ztr = Xtr["z"].values
    ridge = fit_ridge(Xtr, ztr)
    ridge_z = pred_ridge(ridge, Xtr)
    gbdt = fit_gbdt(Xtr, ztr - ridge_z)
    return snap, static2, ridge, gbdt


def predict_all(fit_, batch):
    snap, static2, ridge, gbdt = fit_
    X = build_features(batch, snap, static2)
    z_base = baseline_pred(X, snap)
    z_ridge = pred_ridge(ridge, X)
    z_gbdt = pred_gbdt(gbdt, X)
    out = pd.DataFrame({
        "panel_ols": sigmoid(z_base),
        "ridge": sigmoid(z_ridge),
        "gbdt": sigmoid(z_gbdt),
        "hybrid": sigmoid(z_ridge + z_gbdt),
    })
    out = out.clip(0.0, 1.0)
    return X, out


# ----------------------------------------------------------------------------
# 4. 指标
# ----------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    mse = mean_squared_error(y, p)
    mae = mean_absolute_error(y, p)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mae,
        "MAPE(%)": 100.0 * np.mean(np.abs((y - p) / y)),
        "R2": r2_score(y, p),
    }


MODELS = ["panel_ols", "ridge", "gbdt", "hybrid"]


def main():
    d, dist, static = load_data()
    print(f"有效样本 {len(d)} 条，面板 {d.unique_id.nunique()} 个，年份 {d.year.min()}-{d.year.max()}")

    # ---------------- 10 折时序交叉验证 ----------------
    fold_rows = {m: [] for m in MODELS}
    pooled_y, pooled_p = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    for k, yr in enumerate(CV_YEARS, 1):
        tr = d[d["year"] < yr]
        te = d[d["year"] == yr]
        fit_ = fit_all(tr, dist, static)
        Xte, pred = predict_all(fit_, te)
        for m in MODELS:
            mt = metrics(te["y"].values, pred[m].values)
            mt.update(fold=k, test_year=yr, n_test=len(te))
            fold_rows[m].append(mt)
            pooled_y[m].append(te["y"].values)
            pooled_p[m].append(pred[m].values)
        print(f"fold {k:2d} test_year={yr} n={len(te):3d} "
              + " ".join(f"{m} MSE={metrics(te['y'].values, pred[m].values)['MSE']:.4f}"
                         for m in MODELS))

    # 模型选择：10 折宏平均 MSE
    cv_summary = {}
    for m in MODELS:
        fdf = pd.DataFrame(fold_rows[m])
        macro = {c: fdf[c].mean() for c in ["MSE", "RMSE", "MAE", "MAPE(%)", "R2"]}
        pool = metrics(np.concatenate(pooled_y[m]), np.concatenate(pooled_p[m]))
        cv_summary[m] = {"macro": macro, "pooled": pool}
    best = min(MODELS, key=lambda m: cv_summary[m]["macro"]["MSE"])
    print("\n=== 10 折宏平均指标 ===")
    for m in MODELS:
        mm = cv_summary[m]["macro"]
        print(f"{m:10s} MSE={mm['MSE']:.5f} RMSE={mm['RMSE']:.5f} MAE={mm['MAE']:.5f} "
              f"MAPE={mm['MAPE(%)']:.2f}% R2={mm['R2']:.4f}")
    print("selected model:", best)

    # metrics.csv：最优模型的逐折 + 均值 + 池化
    fdf = pd.DataFrame(fold_rows[best])[["fold", "test_year", "n_test",
                                          "MSE", "RMSE", "MAE", "MAPE(%)", "R2"]]
    mean_row = {"fold": "mean", "test_year": "2013-2022",
                "n_test": int(fdf["n_test"].sum())}
    mean_row.update({c: fdf[c].mean() for c in ["MSE", "RMSE", "MAE", "MAPE(%)", "R2"]})
    pool = cv_summary[best]["pooled"]
    pool_row = {"fold": "pooled", "test_year": "2013-2022",
                "n_test": int(fdf["n_test"].sum())}
    pool_row.update(pool)
    metrics_df = pd.concat([fdf, pd.DataFrame([mean_row, pool_row])], ignore_index=True)
    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig",
                      float_format="%.6f")
    print("saved:", metrics_path)

    # 顺便保存四模型对比表，供报告参考
    cmp_df = pd.DataFrame({
        m: {**{f"cv_macro_{k2}": v2 for k2, v2 in cv_summary[m]["macro"].items()},
            **{f"pooled_{k2}": v2 for k2, v2 in cv_summary[m]["pooled"].items()}}
        for m in MODELS}).T.reset_index().rename(columns={"index": "model"})
    cmp_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False,
                  encoding="utf-8-sig", float_format="%.6f")

    # ---------------- 全量训练 + 未来预测 ----------------
    fit_full = fit_all(d, dist, static)
    panels = d[["unique_id", "province", "grade_group", "residence",
                "method_code"]].drop_duplicates("unique_id")
    fut = panels.assign(key=1).merge(
        pd.DataFrame({"year": FUTURE_YEARS, "key": 1}), on="key").drop(columns="key")
    _, fut_pred = predict_all(fit_full, fut)
    fut_out = fut[["unique_id", "province", "grade_group", "residence", "year"]].copy()
    fut_out["predicted"] = fut_pred[best].values
    fut_out = fut_out.sort_values(["unique_id", "year"]).reset_index(drop=True)
    fut_path = os.path.join(OUT_DIR, "predictions_future.csv")
    fut_out.to_csv(fut_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print("saved:", fut_path, fut_out.shape)

    # ---------------- 省份级聚合（供绘图与解读） ----------------
    # 历史：同一(省份,年,年级,城乡)内按 sample_size 加权合并方法，再对 8 个人群格等权
    def cell_agg(frame, value_col, weighted=False):
        if weighted:
            w = frame["sample_size"].clip(lower=1)
            tmp = frame.assign(_w=w, _v=frame[value_col] * w)
            g = tmp.groupby(["province", "year", "grade_group", "residence"],
                            observed=True).agg(_v=("_v", "sum"), _w=("_w", "sum")).reset_index()
            g["cell"] = g["_v"] / g["_w"]
        else:
            g = frame.groupby(["province", "year", "grade_group", "residence"],
                              observed=True)[value_col].mean().reset_index().rename(
                columns={value_col: "cell"})
        return g.groupby(["province", "year"], observed=True)["cell"].mean().reset_index()

    hist_prov = cell_agg(d, "y", weighted=True)
    fut_prov = cell_agg(fut_out.assign(sample_size=1.0), "predicted", weighted=False)
    hist_prov.to_csv(os.path.join(OUT_DIR, "province_history.csv"), index=False,
                     encoding="utf-8-sig", float_format="%.6f")
    fut_prov.to_csv(os.path.join(OUT_DIR, "province_future.csv"), index=False,
                    encoding="utf-8-sig", float_format="%.6f")

    # 全国等省均值（报告用）
    nat = pd.concat([hist_prov.assign(kind="hist"),
                     fut_prov.assign(kind="future")]).groupby(["year", "kind"])["cell"].mean()
    print("\n全国（省际等权）关键年份：")
    for yr in [1995, 2000, 2005, 2010, 2014, 2019, 2022, 2030, 2040, 2050]:
        sub = pd.concat([hist_prov, fut_prov])
        v = sub[sub.year == yr]["cell"].mean()
        print(f"  {yr}: {v:.3f}")
    return best, cv_summary, metrics_df, hist_prov, fut_prov, d


if __name__ == "__main__":
    main()
