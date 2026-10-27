"""Myopia prevalence spatiotemporal forecasting pipeline (library)."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn import metrics as skm

DATA_DIR = "C:/Users/echo_/OneDrive/Desktop/06_myopia"

EN2CN = {
 'Beijing':'北京','Tianjin':'天津','Hebei':'河北','Shanxi':'山西','Inner Mongolia':'内蒙古',
 'Liaoning':'辽宁','Jilin':'吉林','Heilongjiang':'黑龙江','Shanghai':'上海','Jiangsu':'江苏',
 'Zhejiang':'浙江','Anhui':'安徽','Fujian':'福建','Jiangxi':'江西','Shandong':'山东',
 'Henan':'河南','Hubei':'湖北','Hunan':'湖南','Guangdong':'广东','Guangxi':'广西',
 'Hainan':'海南','Chongqing':'重庆','Sichuan':'四川','Guizhou':'贵州','Yunnan':'云南',
 'Tibet':'西藏','Shaanxi':'陕西','Gansu':'甘肃','Qinghai':'青海','Ningxia':'宁夏','Xinjiang':'新疆',
}
CN2EN = {v: k for k, v in EN2CN.items()}
PROVINCES = sorted(EN2CN.keys())
GRADES = ['1-3', '4-6', '7-9', '10-12']


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_data():
    df = pd.read_csv(DATA_DIR + "/observations.csv")
    df = df[df['y'].notna()].copy()
    df['sample_size'] = df['sample_size'].fillna(0)
    return df


def build_static_spatial(df):
    sun = pd.read_csv(DATA_DIR + "/spatial/sunshine-hours-rank.csv")
    sun.columns = [c.strip().lstrip('\ufeff') for c in sun.columns]
    sun['en'] = sun['province'].map(CN2EN)
    sun_map = dict(zip(sun['en'], sun['rank']))
    ranks = np.array([sun_map.get(p, np.nan) for p in PROVINCES], dtype=float)
    ranks = (ranks - np.nanmean(ranks)) / np.nanstd(ranks)
    sun_scaled = dict(zip(PROVINCES, ranks))

    D = pd.read_csv(DATA_DIR + "/spatial/province_distance_matrix.csv")
    D.columns = [c.strip() for c in D.columns]
    D = D.rename(columns={D.columns[0]: 'prov'})
    D['prov'] = D['prov'].str.strip()
    # rename chinese column headers to english
    rename_cols = {c: CN2EN[c.strip()] for c in D.columns if c.strip() in CN2EN}
    D = D.rename(columns=rename_cols)
    D['en'] = D['prov'].map(CN2EN)
    D = D.set_index('en')
    dist = D[PROVINCES]

    prov_mean = df.groupby('province')['y'].mean()
    prov_logit = {p: logit(prov_mean.get(p, np.nan)) for p in PROVINCES}

    W = {}
    for p in PROVINCES:
        w = {}
        for q in PROVINCES:
            if p == q:
                continue
            d = float(dist.loc[p, q])
            w[q] = 1.0 / (d + 50.0)
        s = sum(w.values())
        W[p] = {q: v / s for q, v in w.items()}

    lag = {}
    for p in PROVINCES:
        num, den = 0.0, 0.0
        for q, wq in W[p].items():
            if not np.isnan(prov_logit[q]):
                num += wq * prov_logit[q]
                den += wq
        lag[p] = num / den if den > 0 else 0.0
    lag_arr = np.array([lag[p] for p in PROVINCES])
    lag_arr = (lag_arr - lag_arr.mean()) / lag_arr.std()
    lag_scaled = dict(zip(PROVINCES, lag_arr))
    return sun_scaled, lag_scaled


def make_features(df, cfg, sun_scaled, lag_scaled):
    X = pd.DataFrame(index=df.index)
    yr = df['year'].values.astype(float)
    t = (yr - 2008.0) / 10.0
    X['t'] = t
    for p in PROVINCES:
        X['prov_' + p.replace(' ', '_')] = (df['province'].values == p).astype(float)
    for g in GRADES[1:]:
        X['grade_' + g] = (df['grade_group'].values == g).astype(float)
    X['urban'] = (df['residence'].values == 'urban').astype(float)
    for m in [2, 3]:
        X['method_' + str(m)] = (df['method_code'].values == m).astype(float)
    X['sunshine'] = df['province'].map(sun_scaled).values
    X['spatial_lag'] = df['province'].map(lag_scaled).values
    if cfg.get('grade_slope', False):
        for g in GRADES[1:]:
            X['t_grade_' + g] = t * (df['grade_group'].values == g).astype(float)
    if cfg.get('res_slope', False):
        X['t_urban'] = t * X['urban'].values
    if cfg.get('prov_slope', False):
        for p in PROVINCES:
            X['t_prov_' + p.replace(' ', '_')] = t * X['prov_' + p.replace(' ', '_')].values
    if cfg.get('quad', False):
        X['t2'] = t * t
    return X


def build_model(cfg):
    alphas = np.logspace(-2, 4, 30)
    return Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=alphas))])


def fit_predict(Xtr, ytr_logit, wtr, Xte, cfg):
    model = build_model(cfg)
    model.fit(Xtr, ytr_logit, ridge__sample_weight=wtr)
    return inv_logit(model.predict(Xte)), model


def evaluate(y_true, y_pred):
    mse = skm.mean_squared_error(y_true, y_pred)
    return dict(MSE=mse, RMSE=np.sqrt(mse), MAE=skm.mean_absolute_error(y_true, y_pred),
                MAPE=np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                R2=skm.r2_score(y_true, y_pred))


def forward_chaining_cv(df, cfg, sun_scaled, lag_scaled, n_folds=10):
    years = np.sort(df['year'].unique())
    test_years = years[-n_folds:]
    X_all = make_features(df, cfg, sun_scaled, lag_scaled)
    y_all = logit(df['y'].values)
    w_all = df['sample_size'].values.astype(float)
    w_all = np.where(w_all <= 0, 1.0, w_all)
    yr_all = df['year'].values
    rows = []
    for i, ty in enumerate(test_years):
        tr = yr_all < ty
        te = yr_all == ty
        if tr.sum() < 30 or te.sum() == 0:
            continue
        pred, _ = fit_predict(X_all[tr], y_all[tr], w_all[tr], X_all[te], cfg)
        m = evaluate(df['y'].values[te], pred)
        m.update(fold=i + 1, test_year=int(ty), n_train=int(tr.sum()), n_test=int(te.sum()))
        rows.append(m)
    folds = pd.DataFrame(rows)
    mean = folds[['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']].mean()
    return folds, mean
