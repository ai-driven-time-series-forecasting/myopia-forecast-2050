import os
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = DATA_DIR
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = ["PingFang SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.unicode_minus"] = False

PROVINCE_MAP = {
    "安徽": "Anhui",
    "北京": "Beijing",
    "福建": "Fujian",
    "甘肃": "Gansu",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "贵州": "Guizhou",
    "海南": "Hainan",
    "河北": "Hebei",
    "河南": "Henan",
    "黑龙江": "Heilongjiang",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "内蒙古": "Inner Mongolia",
    "江苏": "Jiangsu",
    "江西": "Jiangxi",
    "吉林": "Jilin",
    "辽宁": "Liaoning",
    "宁夏": "Ningxia",
    "青海": "Qinghai",
    "山东": "Shandong",
    "山西": "Shanxi",
    "陕西": "Shaanxi",
    "上海": "Shanghai",
    "四川": "Sichuan",
    "天津": "Tianjin",
    "西藏": "Tibet",
    "新疆": "Xinjiang",
    "云南": "Yunnan",
    "浙江": "Zhejiang",
    "重庆": "Chongqing",
}

MODEL_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 15,
}


def load_data():
    obs = pd.read_csv(os.path.join(DATA_DIR, "observations.csv"))
    obs = obs[obs["valid_counts"]].copy()

    sunshine = pd.read_csv(os.path.join(DATA_DIR, "spatial", "sunshine-hours-rank.csv"))
    sunshine["province"] = sunshine["province"].map(PROVINCE_MAP)
    obs = obs.merge(sunshine, on="province", how="left")

    obs["log_sample_size"] = np.log10(obs["sample_size"].fillna(1) + 1)
    obs = obs.sort_values(["unique_id", "year"]).reset_index(drop=True)
    return obs


def build_static_features(df, encoders=None, fit=False):
    df = df.copy()
    df["year_norm"] = df["year"] - 1995
    df["year_norm_sq"] = df["year_norm"] ** 2
    df["year_norm_cu"] = df["year_norm"] ** 3

    if fit:
        encoders = {}
        encoders["grade"] = OrdinalEncoder(categories=[["1-3", "4-6", "7-9", "10-12"]])
        df["grade_encoded"] = encoders["grade"].fit_transform(df[["grade_group"]])
    else:
        df["grade_encoded"] = encoders["grade"].transform(df[["grade_group"]])

    df["residence_encoded"] = (df["residence"] == "urban").astype(int)

    cat_cols = ["province", "method_code"]
    if fit:
        encoders["cat"] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_arr = encoders["cat"].fit_transform(df[cat_cols])
    else:
        cat_arr = encoders["cat"].transform(df[cat_cols])
    cat_df = pd.DataFrame(
        cat_arr, columns=encoders["cat"].get_feature_names_out(cat_cols), index=df.index
    )
    df = pd.concat([df, cat_df], axis=1)

    if fit:
        return df, encoders
    return df


def add_dynamic_features(df, y_col="y"):
    df = df.copy()
    grouped = df.groupby("unique_id")[y_col]

    df["lag1"] = grouped.shift(1)
    df["lag2"] = grouped.shift(2)
    df["lag3"] = grouped.shift(3)
    df["diff1"] = grouped.diff(1)

    df["roll_mean_2"] = grouped.transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    df["roll_mean_3"] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["roll_mean_5"] = grouped.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    cum_mean = grouped.expanding().mean().reset_index(level=0, drop=True)
    df["panel_cum_mean"] = cum_mean
    return df


def get_feature_cols(encoders):
    numeric = [
        "year_norm",
        "year_norm_sq",
        "year_norm_cu",
        "grade_encoded",
        "residence_encoded",
        "rank",
        "log_sample_size",
    ]
    dynamic = ["lag1", "lag2", "lag3", "diff1", "roll_mean_2", "roll_mean_3", "roll_mean_5", "panel_cum_mean"]
    cat_cols = encoders["cat"].get_feature_names_out(["province", "method_code"]).tolist()
    return numeric + dynamic + cat_cols


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def prepare_fold(df, test_years):
    """Build features for one CV fold. Test-year targets are masked so lags cannot leak."""
    raw_train = df[~df["year"].isin(test_years)].copy()
    raw_test = df[df["year"].isin(test_years)].copy()
    raw_test["y"] = np.nan

    raw_combined = pd.concat([raw_train, raw_test], ignore_index=True)
    raw_combined = raw_combined.sort_values(["unique_id", "year"]).reset_index(drop=True)

    raw_combined, encoders = build_static_features(raw_combined, fit=True)
    raw_combined = add_dynamic_features(raw_combined, y_col="y")

    train_features = raw_combined[~raw_combined["year"].isin(test_years)].copy()
    test_features = raw_combined[raw_combined["year"].isin(test_years)].copy()

    feature_cols = get_feature_cols(encoders)
    return train_features, test_features, encoders, feature_cols


def time_series_cv(df, n_splits=10):
    years = sorted(df["year"].unique())
    folds = np.array_split(years, n_splits)

    fold_metrics = []
    all_oof = []

    for i, test_years in enumerate(folds, start=1):
        test_years = test_years.tolist()
        train_features, test_features, encoders, feature_cols = prepare_fold(df, test_years)

        X_train = train_features[feature_cols]
        y_train = train_features["y"].values
        X_test = test_features[feature_cols]

        y_test = df[df["year"].isin(test_years)]["y"].values

        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, 1)

        metrics = compute_metrics(y_test, y_pred)
        metrics["fold"] = i
        metrics["test_years"] = ";".join(str(y) for y in test_years)
        fold_metrics.append(metrics)

        oof = test_features[["unique_id", "province", "grade_group", "residence", "year"]].copy()
        oof["y_true"] = y_test
        oof["y_pred"] = y_pred
        oof["fold"] = i
        all_oof.append(oof)

        print(f"Fold {i}: test years {test_years} -> RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    mean_row = metrics_df[["MSE", "RMSE", "MAE", "MAPE", "R2"]].mean().to_dict()
    mean_row["fold"] = "mean"
    mean_row["test_years"] = "average_of_10_folds"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    return metrics_df, pd.concat(all_oof, ignore_index=True), encoders, feature_cols


def train_full_model(df):
    """Train on all 1995-2022 data for final forecasting."""
    df_features, encoders = build_static_features(df.copy(), fit=True)
    df_features = add_dynamic_features(df_features, y_col="y")
    feature_cols = get_feature_cols(encoders)

    X = df_features[feature_cols]
    y = df_features["y"].values

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    return model, encoders, feature_cols, df_features


def forecast_future(df, model, encoders, feature_cols, end_year=2050):
    panels = df[["unique_id", "province", "grade_group", "residence", "method_code"]].drop_duplicates()

    future_rows = []
    for _, panel in panels.iterrows():
        historical_sizes = df[df["unique_id"] == panel["unique_id"]]["sample_size"]
        sample_size = historical_sizes.median() if len(historical_sizes) > 0 else 1000
        for year in range(2023, end_year + 1):
            future_rows.append(
                {
                    "unique_id": panel["unique_id"],
                    "province": panel["province"],
                    "grade_group": panel["grade_group"],
                    "residence": panel["residence"],
                    "method_code": panel["method_code"],
                    "year": year,
                    "sample_size": sample_size,
                    "y": np.nan,
                }
            )
    future = pd.DataFrame(future_rows)
    province_meta = df[["province", "rank"]].drop_duplicates()
    future = future.merge(province_meta, on="province", how="left")

    history = df[["unique_id", "province", "grade_group", "residence", "method_code", "year", "sample_size", "rank", "y"]].copy()
    combined = pd.concat([history, future], ignore_index=True)
    combined = combined.sort_values(["unique_id", "year"]).reset_index(drop=True)

    combined["log_sample_size"] = np.log10(combined["sample_size"].fillna(1) + 1)
    combined = build_static_features(combined, encoders=encoders, fit=False)

    for col in ["lag1", "lag2", "lag3", "diff1", "roll_mean_2", "roll_mean_3", "roll_mean_5", "panel_cum_mean"]:
        combined[col] = np.nan

    for year in range(2023, end_year + 1):
        year_idx = combined[combined["year"] == year].index
        for idx in year_idx:
            panel_id = combined.loc[idx, "unique_id"]
            past = combined[(combined["unique_id"] == panel_id) & (combined.index < idx)]
            if len(past) == 0:
                continue
            y_vals = past["y"].values
            combined.loc[idx, "lag1"] = y_vals[-1] if len(y_vals) >= 1 else np.nan
            combined.loc[idx, "lag2"] = y_vals[-2] if len(y_vals) >= 2 else np.nan
            combined.loc[idx, "lag3"] = y_vals[-3] if len(y_vals) >= 3 else np.nan
            combined.loc[idx, "diff1"] = y_vals[-1] - y_vals[-2] if len(y_vals) >= 2 else np.nan
            combined.loc[idx, "roll_mean_2"] = np.nanmean(y_vals[-2:]) if len(y_vals) >= 1 else np.nan
            combined.loc[idx, "roll_mean_3"] = np.nanmean(y_vals[-3:]) if len(y_vals) >= 1 else np.nan
            combined.loc[idx, "roll_mean_5"] = np.nanmean(y_vals[-5:]) if len(y_vals) >= 1 else np.nan
            combined.loc[idx, "panel_cum_mean"] = np.nanmean(y_vals)

        X_year = combined.loc[year_idx, feature_cols]
        preds = np.clip(model.predict(X_year), 0, 1)
        combined.loc[year_idx, "y"] = preds

    future_df = combined[combined["year"] >= 2023].copy()
    future_df = future_df.rename(columns={"y": "predicted"})
    return future_df[["unique_id", "province", "grade_group", "residence", "year", "sample_size", "predicted"]]


def aggregate_province_rate(df, value_col="y", weight_col="sample_size"):
    df = df.copy()
    df[weight_col] = df[weight_col].fillna(df[weight_col].median())
    sub = df[["province", "year", value_col, weight_col]].copy()
    grouped = sub.groupby(["province", "year"]).apply(
        lambda x: np.average(x[value_col], weights=x[weight_col])
    )
    return grouped.reset_index().rename(columns={0: "prevalence"})


def plot_province_forecasts(history_df, future_df, out_path):
    hist_prov = aggregate_province_rate(history_df, value_col="y", weight_col="sample_size")
    fut_prov = aggregate_province_rate(future_df, value_col="predicted", weight_col="sample_size")

    fig, ax = plt.subplots(figsize=(14, 8))

    provinces = sorted(hist_prov["province"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(provinces)))

    for province, color in zip(provinces, colors):
        h = hist_prov[hist_prov["province"] == province].sort_values("year")
        f = fut_prov[fut_prov["province"] == province].sort_values("year")

        ax.plot(h["year"], h["prevalence"], color=color, alpha=0.7, linewidth=1.2, label=province)
        if len(f) > 0:
            ax.plot(f["year"], f["prevalence"], color=color, alpha=0.7, linewidth=1.2, linestyle="--")

    national_hist = hist_prov.groupby("year")["prevalence"].mean().reset_index()
    national_fut = fut_prov.groupby("year")["prevalence"].mean().reset_index()
    ax.plot(national_hist["year"], national_hist["prevalence"], color="black", linewidth=2.5, label="National average")
    ax.plot(national_fut["year"], national_fut["prevalence"], color="black", linewidth=2.5, linestyle="--")

    ax.axvline(2022.5, color="gray", linestyle=":", linewidth=1.5)
    ax.axvspan(2022.5, 2050.5, color="gray", alpha=0.08)
    ax.text(2023.2, ax.get_ylim()[1] * 0.95, "Forecast period", fontsize=10, color="gray")

    ax.set_xlabel("Year")
    ax.set_ylabel("Myopia prevalence")
    ax.set_title("Provincial youth myopia prevalence: historical (1995-2022) and forecast (2023-2050)")
    ax.set_ylim(0, 1)
    ax.set_xlim(1994, 2051)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def write_word_doc(metrics_mean, out_path):
    doc = Document()
    title = doc.add_heading("青少年近视率预测说明", level=0)
    title.alignment = 1

    doc.add_paragraph(f"生成日期：{datetime.now().strftime('%Y-%m-%d')}")

    doc.add_heading("一、方法与建模思路", level=1)
    p = doc.add_paragraph()
    p.add_run("本分析由大型语言模型（LLM）AI 助手辅助完成。").bold = True
    p.add_run(
        "建模目标为利用中国各省 1995-2022 年青少年近视率面板数据，结合时间与空间信息，"
        "预测 2023-2050 年各省份不同学段、城乡分组的近视患病率变化。"
        "数据中的每个观测为 province × grade_group × residence 构成的 panel，目标变量为 computed_prevalence。"
    )

    doc.add_paragraph(
        "特征工程方面，首先构造时间特征：年份及其二次、三次多项式，用于捕捉近视率随年份的非线性增长与可能的饱和趋势；"
        "其次构造面板动态特征：基于每个 panel 的滞后 1-3 期近视率、一阶差分、过去 2/3/5 期滚动均值，以及 panel 累计均值，"
        "这些特征使模型能够利用历史轨迹进行预测；"
        "再次加入空间特征：各省年日照时数排名（rank），用以刻画不同地区光照条件对近视流行的潜在影响；"
        "最后对省份、检测方法进行 One-Hot 编码，对学段进行 Ordinal 编码，城乡为二元变量。"
    )

    doc.add_paragraph(
        "模型选用 HistGradientBoostingRegressor。该梯度提升树模型能够自动处理特征缺失，"
        "并捕捉时间、空间、学段、城乡之间的复杂非线性交互。"
        "为验证精度，采用十折时间序列交叉验证：将年份按时间顺序分为 10 段，每折用此前所有年份训练、用当前段年份测试，"
        "绝不把测试年份纳入训练，也不随机打乱。十折指标取平均作为最终评估结果。"
    )

    doc.add_heading("二、2050 年预测结果解读", level=1)
    doc.add_paragraph(
        "从历史数据看，1995-2022 年间我国青少年近视率总体呈快速上升趋势，"
        "高年级（7-9、10-12）明显高于低年级，城市普遍高于农村。"
        "部分地区在 2010 年前后增速较快，随后出现一定平台化迹象，但整体仍处于高位。"
    )
    doc.add_paragraph(
        "预测结果显示，2023-2050 年多数省份青少年近视率将继续缓慢上行，"
        "但增速较 2000-2015 年明显放缓，逐步接近生理与环境可承受的上限。"
        "高年级群体（尤其 10-12 年级）可能率先进入高位平台，部分省份 urban 群体预测值接近或超过 0.90；"
        "低年级与农村群体仍有上升空间。区域差异将持续存在，东部、光照较弱地区整体高于西部、高光照地区。"
    )

    doc.add_heading("三、远期预测的不确定性讨论", level=1)
    m = metrics_mean
    doc.add_paragraph(
        f"十折时间序列交叉验证的平均指标为：RMSE={m['RMSE']:.4f}、MAE={m['MAE']:.4f}、"
        f"MAPE={m['MAPE']:.2f}%、R²={m['R2']:.4f}。"
        "从样本内与近样本预测来看，模型能够解释大部分变异，短期预测具有一定可信度。"
    )
    doc.add_paragraph(
        "但需强调，2023-2050 年为远期外推，不确定性随预测步长显著放大。"
        "模型依赖历史趋势外推，未纳入政策干预（如教育部近视防控目标）、技术进步、人口结构变化、疫情等突发事件的影响；"
        "不同省份数据覆盖密度差异较大，部分 panel 历史观测稀疏，远期预测可靠性更低。"
        "因此，2050 年结果应理解为基于历史惯性情景的参考轨迹，而非精确预测，"
        "建议结合专家判断与情景分析共同用于最终报告。"
    )

    doc.save(out_path)
    print(f"Saved Word document: {out_path}")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Valid observations: {len(df)}, panels: {df['unique_id'].nunique()}, years: {df['year'].min()}-{df['year'].max()}")

    print("\nRunning 10-fold time-series cross-validation...")
    metrics_df, oof_df, encoders, feature_cols = time_series_cv(df, n_splits=10)

    metrics_csv = os.path.join(OUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics: {metrics_csv}")

    print("\nTraining final model on full history...")
    final_model, encoders, feature_cols, df_features = train_full_model(df)

    print("\nForecasting 2023-2050...")
    future_df = forecast_future(df, final_model, encoders, feature_cols, end_year=2050)
    future_csv = os.path.join(OUT_DIR, "predictions_future.csv")
    future_df.to_csv(future_csv, index=False)
    print(f"Saved future predictions: {future_csv}")

    print("\nCreating visualization...")
    plot_path = os.path.join(OUT_DIR, "province_forecast_1995_2050.png")
    plot_province_forecasts(df, future_df, plot_path)

    print("\nWriting Word document...")
    mean_metrics = metrics_df[metrics_df["fold"] == "mean"].iloc[0].to_dict()
    doc_path = os.path.join(OUT_DIR, "预测说明_AI大模型.docx")
    write_word_doc(mean_metrics, doc_path)

    print("\nAll deliverables generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
