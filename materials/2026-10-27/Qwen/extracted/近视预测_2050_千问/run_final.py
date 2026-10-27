import sys
sys.path.insert(0, "C:/Users/echo_/OneDrive/Desktop/近视预测_2050")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from myopia_lib import *

OUT = "C:/Users/echo_/OneDrive/Desktop/近视预测_2050"
CFG = dict(grade_slope=True, res_slope=True)   # selected model B

df = load_data()
sun, lag = build_static_spatial(df)

# ---------- 1) 10-fold forward-chaining CV -> metrics.csv ----------
folds, mean = forward_chaining_cv(df, CFG, sun, lag, n_folds=10)
metrics_out = folds[['fold','test_year','n_train','n_test','MSE','RMSE','MAE','MAPE','R2']].copy()
mean_row = {'fold':'mean','test_year':'2013-2022','n_train':int(folds.n_train.mean()),
            'n_test':int(folds.n_test.mean()),'MSE':mean.MSE,'RMSE':mean.RMSE,
            'MAE':mean.MAE,'MAPE':mean.MAPE,'R2':mean.R2}
metrics_out = pd.concat([metrics_out, pd.DataFrame([mean_row])], ignore_index=True)
metrics_out.to_csv(OUT + "/metrics.csv", index=False)
print("metrics.csv saved. MEAN:")
print(mean.round(5).to_string())

# ---------- 2) fit final model on all data, predict 2023-2050 ----------
X_all = make_features(df, CFG, sun, lag)
ylog = logit(df['y'].values)
w = np.where(df['sample_size'].values <= 0, 1.0, df['sample_size'].values.astype(float))
model = build_model(CFG)
model.fit(X_all, ylog, ridge__sample_weight=w)

panels = df[['unique_id','province','grade_group','residence','method_code']].drop_duplicates('unique_id')
future_years = np.arange(2023, 2051)
rows = []
for _, p in panels.iterrows():
    for yr in future_years:
        rows.append(dict(unique_id=p.unique_id, province=p.province, grade_group=p.grade_group,
                         residence=p.residence, method_code=p.method_code, year=int(yr)))
fut = pd.DataFrame(rows)
Xf = make_features(fut, CFG, sun, lag)
fut['predicted'] = inv_logit(model.predict(Xf))
pred_out = fut[['unique_id','province','grade_group','residence','year','predicted']]
pred_out.to_csv(OUT + "/predictions_future.csv", index=False)
print("predictions_future.csv saved. rows:", len(pred_out), "panels:", panels.unique_id.nunique())

# ---------- 3) province-level trajectories for the figure ----------
hist_prov = df.groupby(['province','year'])['y'].mean().reset_index()
fut_prov = fut.groupby(['province','year'])['predicted'].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 8))
cmap = plt.get_cmap('tab20', len(PROVINCES))
colors = {p: cmap(i) for i, p in enumerate(PROVINCES)}
rmse_cv = float(mean.RMSE)   # out-of-sample residual SD from 10-fold CV

for p in PROVINCES:
    h = hist_prov[hist_prov.province==p].sort_values('year')
    f = fut_prov[fut_prov.province==p].sort_values('year')
    c = colors[p]
    ax.plot(h.year, h.y, color=c, lw=1.6, alpha=0.95)              # historical solid
    anchor_y = h.y.values[-1] if len(h) else f.predicted.values[0]
    anchor_x = h.year.values[-1] if len(h) else 2022
    ax.plot([anchor_x]+list(f.year), [anchor_y]+list(f.predicted),
            color=c, lw=1.6, ls='--', alpha=0.95)                  # forecast dashed
    # 95% prediction interval for the province-year mean (dashed edges + light fill)
    npan = fut[fut.province==p].groupby('year')['predicted'].size()
    half = 1.96 * rmse_cv / np.sqrt(npan.reindex(f.year).fillna(1).values)
    up = np.clip(f.predicted.values + half, 0, 1)
    lo = np.clip(f.predicted.values - half, 0, 1)
    ax.fill_between(f.year, lo, up, color=c, alpha=0.10, lw=0)
    ax.plot(f.year, up, color=c, lw=0.6, ls='--', alpha=0.55)
    ax.plot(f.year, lo, color=c, lw=0.6, ls='--', alpha=0.55)

ax.axvline(2022.5, color='grey', ls=':', lw=1.2)
ax.text(2022.7, 0.03, 'forecast $\\rightarrow$', color='grey', fontsize=10)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Myopia prevalence (proportion)', fontsize=12)
ax.set_title('China adolescent myopia prevalence by province: historical (1995-2022, solid)\n'
             'and projected (2023-2050, dashed)', fontsize=13)
ax.set_xlim(1994, 2051); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.25)
handles = [plt.Line2D([0],[0], color=colors[p], lw=2, label=p) for p in PROVINCES]
ax.legend(handles=handles, title='Province', bbox_to_anchor=(1.01, 1), loc='upper left',
          fontsize=7, ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT + "/province_myopia_projection.png", dpi=300, bbox_inches='tight')
print("figure saved")

# ---------- numbers for the report ----------
nat_hist = df.groupby('year')['y'].mean()
nat_fut = fut.groupby('year')['predicted'].mean()
print("NAT hist 1995/2000/2010/2022:", [round(nat_hist.get(y,np.nan),3) for y in [1995,2000,2010,2022]])
print("NAT fut 2030/2040/2050:", [round(nat_fut.get(y),3) for y in [2030,2040,2050]])
p50 = fut_prov[fut_prov.year==2050].set_index('province')['predicted']
print("2050 province min/max:", round(p50.min(),3), p50.idxmin(), round(p50.max(),3), p50.idxmax())
# grade-level 2050
g = fut[fut.year==2050].groupby('grade_group')['predicted'].mean()
print("2050 by grade:", g.round(3).to_dict())
r = fut[fut.year==2050].groupby('residence')['predicted'].mean()
print("2050 by residence:", r.round(3).to_dict())
# save summary json for report
import json
summary = dict(mean_metrics=mean.round(5).to_dict(),
               nat_hist={int(k):round(float(v),4) for k,v in nat_hist.items()},
               nat_fut={int(k):round(float(v),4) for k,v in nat_fut.items()},
               p2050_min=[p50.idxmin(), round(float(p50.min()),3)],
               p2050_max=[p50.idxmax(), round(float(p50.max()),3)],
               grade2050={k:round(float(v),3) for k,v in g.items()},
               res2050={k:round(float(v),3) for k,v in r.items()})
json.dump(summary, open(OUT+"/summary.json","w"), ensure_ascii=False, indent=2)
print("summary.json saved")
