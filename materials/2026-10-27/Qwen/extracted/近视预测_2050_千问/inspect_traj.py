import sys
sys.path.insert(0, "C:/Users/echo_/OneDrive/Desktop/近视预测_2050")
import numpy as np, pandas as pd
from myopia_lib import *

df = load_data()
sun, lag = build_static_spatial(df)

# historical province-year mean (raw) for reference
hist = df.groupby(['province', 'year'])['y'].mean().reset_index()
nat_hist = df.groupby('year')['y'].mean()

# build future prediction grid at province level (average over grade/residence/method present)
future_years = np.arange(2023, 2051)
# use the panel structure observed historically to build future panels
panels = df[['unique_id','province','grade_group','residence','method_code']].drop_duplicates('unique_id')

def province_traj(cfg):
    X_all = make_features(df, cfg, sun, lag)
    ylog = logit(df['y'].values)
    w = np.where(df['sample_size'].values <= 0, 1.0, df['sample_size'].values.astype(float))
    model = build_model(cfg)
    model.fit(X_all, ylog, ridge__sample_weight=w)
    # future panels
    rows = []
    for _, p in panels.iterrows():
        for yr in future_years:
            rows.append(dict(unique_id=p.unique_id, province=p.province, grade_group=p.grade_group,
                             residence=p.residence, method_code=p.method_code, year=yr))
    fut = pd.DataFrame(rows)
    Xf = make_features(fut, cfg, sun, lag)
    fut['pred'] = inv_logit(model.predict(Xf))
    prov = fut.groupby(['province','year'])['pred'].mean().reset_index()
    nat = fut.groupby('year')['pred'].mean()
    return prov, nat, fut

for name, cfg in {'A_base':dict(), 'B_graderes':dict(grade_slope=True,res_slope=True),
                  'C_full':dict(grade_slope=True,res_slope=True,prov_slope=True)}.items():
    prov, nat, fut = province_traj(cfg)
    print(f"=== {name} ===")
    print("  national hist 2000/2010/2022:", round(nat_hist.get(2000,np.nan),3), round(nat_hist.get(2010,np.nan),3), round(nat_hist.get(2022,np.nan),3))
    print("  national pred 2030/2040/2050:", round(nat.get(2030),3), round(nat.get(2040),3), round(nat.get(2050),3))
    # per-province 2050 range
    p50 = prov[prov.year==2050]['pred']
    print("  province 2050 pred min/median/max:", round(p50.min(),3), round(p50.median(),3), round(p50.max(),3))
    print("  any pred<=0 or >=1:", bool((fut.pred<=0.001).any() or (fut.pred>=0.999).any()))
    # monotonic increase fraction at province level
    inc = prov.sort_values('year').groupby('province')['pred'].apply(lambda s: (np.diff(s.values)>=-1e-6).mean())
    print("  median monotonic-increase fraction:", round(inc.median(),3))
    print()
