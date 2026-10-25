"""Reproducible spatial panel forecasting. See exported methodology.md.

Run: python model_forecast.py --input 06_myopia.zip --out outputs/myopia_forecast
No network calls. Original source files are never changed.
"""
import argparse
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import expit, logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROVINCES = dict(zip(
    ['Anhui','Beijing','Chongqing','Fujian','Gansu','Guangdong','Guangxi','Guizhou','Hainan','Hebei','Heilongjiang','Henan','Hubei','Hunan','Inner Mongolia','Jiangsu','Jiangxi','Jilin','Liaoning','Ningxia','Qinghai','Shaanxi','Shandong','Shanghai','Shanxi','Sichuan','Tianjin','Tibet','Xinjiang','Yunnan','Zhejiang'],
    ['安徽','北京','重庆','福建','甘肃','广东','广西','贵州','海南','河北','黑龙江','河南','湖北','湖南','内蒙古','江苏','江西','吉林','辽宁','宁夏','青海','陕西','山东','上海','山西','四川','天津','西藏','新疆','云南','浙江']))
PNAMES = list(PROVINCES)
GRADES = ['1-3','4-6','7-9','10-12']
PHI = .95  # Declared structural assumption, not tuned against outer test data.
LAMBDAS = [.3, 3., 30.]
METRICS = ['MSE','RMSE','MAE','MAPE','R2']


def read_source(path):
    with zipfile.ZipFile(path) as z:
        raw = pd.read_csv(io.BytesIO(z.read('06_myopia/observations.csv')))
        dist = pd.read_csv(io.BytesIO(z.read('06_myopia/spatial/province_distance_matrix.csv')), index_col=0)
    assert raw.observation_id.is_unique
    assert set(raw.province) == set(PNAMES)
    assert set(raw.grade_group) == set(GRADES)
    valid = (raw.valid_counts.eq(True) & raw.computed_prevalence.notna() &
             raw.computed_prevalence.between(0, 1) & raw.sample_size.gt(0) &
             raw.cases.ge(0) & raw.cases.le(raw.sample_size))
    d = raw.loc[valid].copy()
    assert np.allclose(d.computed_prevalence, d.cases / d.sample_size)
    assert pd.to_datetime(raw.ds).dt.year.eq(raw.year).all()
    # Multiple studies within one panel-year are pooled, never across years.
    keys = ['unique_id','province','grade_group','residence','method_code','year']
    panel = d.groupby(keys, as_index=False).agg(
        cases=('cases','sum'), sample_size=('sample_size','sum'), n_studies=('observation_id','size'))
    panel['target'] = panel.cases / panel.sample_size
    panel = panel.sort_values(['year','unique_id']).reset_index(drop=True)
    panels = raw[['unique_id','province','grade_group','residence','method_code']].drop_duplicates()
    assert panels.unique_id.is_unique
    dd = dist.loc[[PROVINCES[p] for p in PNAMES], [PROVINCES[p] for p in PNAMES]].to_numpy(float)
    assert np.isfinite(dd).all() and np.allclose(dd,dd.T) and np.allclose(np.diag(dd),0)
    # Static geographical graph: four closest provinces, symmetrized, 800-unit kernel.
    w = np.zeros_like(dd)
    for i in range(31):
        jj = np.argsort(dd[i])[1:5]
        w[i,jj] = np.exp(-dd[i,jj]/800.)
    w = (w+w.T)/2
    w /= w.sum(axis=1).mean()
    lap = np.diag(w.sum(axis=1))-w
    audit = dict(raw_rows=len(raw), valid_rows=len(d), excluded_rows=int((~valid).sum()),
                 zero_denominator=int(raw.zero_denominator.sum()), missing_target=int(raw.computed_prevalence.isna().sum()),
                 reported_rate_disagreement=int(raw.rate_disagreement.sum()),
                 y_different_from_target=int((abs(d.y-d.computed_prevalence)>1e-6).sum()),
                 panel_year_rows=len(panel), repeated_panel_year_rows=int(d.duplicated(['unique_id','year']).sum()),
                 all_panels=len(panels), observed_panels=d.unique_id.nunique(),
                 panels_without_valid_target=len(panels)-d.unique_id.nunique(), provinces=31,
                 years=sorted(map(int,d.year.unique())), target='computed_prevalence (pooled cases/sample_size)',
                 graph='symmetric 4-nearest-province graph; exp(-distance/800); normalized mean degree 1',
                 distance_unit='source does not document units; 800 is on the supplied distance scale',
                 sunshine_rank_used=False, original_split_used=False,
                 input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return raw, d, panel, panels, lap, audit


def design(d, cutoff, phi=PHI):
    n=len(d)
    delta=d.year.to_numpy(float)-cutoff
    effective=delta.copy()
    future=delta>0
    effective[future] = (delta[future] if phi==1 else phi*(1-phi**delta[future])/(1-phi))
    t=effective/10.
    g=np.column_stack([d.grade_group.eq(k).to_numpy(float) for k in GRADES[1:]])
    u=d.residence.eq('urban').to_numpy(float)
    m=np.column_stack([d.method_code.eq(k).to_numpy(float) for k in [1,2]])
    pr=np.column_stack([d.province.eq(k).to_numpy(float) for k in PNAMES])
    x=np.column_stack([np.ones(n),t,g,u,m,g*u[:,None],g*t[:,None],u*t,
                       pr,pr*t[:,None], np.einsum('ni,nj->nij',pr,g).reshape(n,-1),pr*u[:,None]])
    assert x.shape[1]==201
    return x


def penalty(lam,lap,spatial=True):
    diag=np.full(201,1e-5)
    diag[1]=.1
    diag[6:8]=.1
    diag[11:15]=.1
    diag[15:46]=lam
    diag[46:77]=10*lam
    diag[77:170]=3*lam
    diag[170:201]=3*lam
    r=np.diag(diag)
    if spatial:
        r[15:46,15:46] += 3*lam*lap
        r[46:77,46:77] += 5*lam*lap
    return r


def fit(d,lam,lap,spatial=True):
    cutoff=int(d.year.max())
    x=design(d,cutoff)
    # Equal study-cell weight: enormous surveys do not dominate heterogeneous studies.
    # Counts weight only studies pooled inside the same panel-year cell.
    z=logit(np.clip(d.target.to_numpy(float),1e-5,1-1e-5))
    a=x.T@x + penalty(lam,lap,spatial)
    coef=solve(a,x.T@z,assume_a='pos')
    return dict(coef=coef,cutoff=cutoff,lam=lam)


def predict(model,d,phi=PHI):
    return expit(design(d,model['cutoff'],phi)@model['coef'])


def score(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    err=y-p
    mse=float(np.mean(err**2))
    den=float(np.sum((y-y.mean())**2))
    nz=y!=0
    return dict(MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(np.mean(abs(err))),
                MAPE=float(np.mean(abs(err[nz]/y[nz]))*100) if nz.any() else np.nan,
                R2=1-float(err@err)/den if den>0 else np.nan)


def inner_select(train,lap):
    ys=sorted(train.year.unique())
    # Up to three inner two-observed-year test blocks, with all training earlier.
    start=max(3,len(ys)-6)
    blocks=[ys[j:j+2] for j in range(start,len(ys)-1,2)]
    losses=[]
    for lam in LAMBDAS:
        fold_losses=[]
        for years in blocks:
            a=train[train.year<min(years)]
            b=train[train.year.isin(years)]
            if len(a)<30 or len(b)<2: continue
            assert a.year.max()<b.year.min()
            p=predict(fit(a,lam,lap),b)
            fold_losses.append(float(np.mean((b.target.to_numpy()-p)**2)))
        losses.append(np.mean(fold_losses) if fold_losses else np.inf)
    selected=LAMBDAS[int(np.argmin(losses))] if np.isfinite(losses).any() else 3.
    return selected,[float(x) if np.isfinite(x) else None for x in losses]


def persistence(train,test):
    last=train.sort_values('year').groupby('unique_id').tail(1).set_index('unique_id').target
    group=train.groupby(['grade_group','residence']).target.mean()
    return np.array([last.get(r.unique_id,group.get((r.grade_group,r.residence),train.target.mean()))
                     for r in test.itertuples()])


def cv(panel,lap):
    ys=sorted(panel.year.unique())
    assert len(ys)>=21
    initial=len(ys)-20
    rows=[]; preds=[]; baselines=[]; selections=[]
    for f in range(10):
        testyears=ys[initial+2*f:initial+2*f+2]
        train=panel[panel.year<min(testyears)].copy()
        test=panel[panel.year.isin(testyears)].copy()
        assert train.year.max()<test.year.min()
        assert not set(train.year)&set(test.year)
        lam,losses=inner_select(train,lap)
        model=fit(train,lam,lap)
        p=predict(model,test)
        rows.append(dict(fold=f+1,train_start=int(train.year.min()),train_end=int(train.year.max()),
                         test_start=int(test.year.min()),test_end=int(test.year.max()),
                         n_train=len(train),n_test=len(test),
                         unseen_panel_test_cells=int((~test.unique_id.isin(train.unique_id)).sum()),
                         selected_lambda=lam,**score(test.target,p)))
        q=test.copy();q['fold']=f+1;q['train_end']=int(train.year.max());q['predicted']=p;preds.append(q)
        for name,pb in [('last_observation',persistence(train,test)),
                        ('same_model_without_spatial',predict(fit(train,lam,lap,False),test))]:
            baselines.append(dict(model=name,fold=f+1,**score(test.target,pb)))
        selections.append(dict(fold=f+1,outer_train_end=int(train.year.max()),
                               inner_candidate_lambdas=LAMBDAS,inner_mean_mse=losses,selected=lam))
        print('FOLD',f+1,'train<=',int(train.year.max()),'test',testyears,'lambda',lam,
              'RMSE',round(rows[-1]['RMSE'],4),flush=True)
    met=pd.DataFrame(rows)
    mean={k:float(met[k].mean()) for k in METRICS}
    met=pd.concat([met,pd.DataFrame([dict(fold='mean',**mean)])],ignore_index=True)
    base=pd.DataFrame(baselines)
    base=pd.concat([base,base.groupby('model')[METRICS].mean().reset_index().assign(fold='mean')],ignore_index=True)
    return met,pd.concat(preds,ignore_index=True),base,selections


def province_grid():
    return pd.DataFrame([dict(province=p,grade_group=g,residence=r,method_code=3,year=y)
                         for p in PNAMES for g in GRADES for r in ['rural','urban'] for y in range(1995,2051)])


def chart(prov,observed,out):
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc') if Path('C:/Windows/Fonts/msyh.ttc').exists() else FontProperties()
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(6,6,figsize=(19,19),sharex=True,sharey=True)
    navy='#214d74';gold='#c89038';gray='#8d969c'
    for ax,p in zip(axes.flat,PNAMES):
        a=prov[prov.province.eq(p)].sort_values('year')
        h=a[a.year<=2022];f=a[a.year>=2022]
        o=observed[observed.province.eq(p)].sort_values('year')
        ax.axvspan(2022.5,2050,color='#f6f1e7',zorder=0)
        ax.plot(o.year,100*o.observed,color=gray,lw=.7,marker='o',ms=2.8,alpha=.9)
        ax.plot(h.year,100*h.predicted,color=navy,lw=1.5)
        ax.fill_between(f.year,100*f.scenario_min,100*f.scenario_max,color=gold,alpha=.23,lw=0)
        ax.plot(f.year,100*f.predicted,color=navy,lw=1.8,ls='--')
        ax.axvline(2022.5,color='#b0a492',lw=.6,ls=':')
        ax.set_title(PROVINCES[p],fontproperties=font,fontsize=12,pad=7,loc='left')
        v=a[a.year.eq(2050)].iloc[0]
        ax.text(.96,.08,f"2050：{v.predicted:.1%}",transform=ax.transAxes,ha='right',fontsize=9,color=navy)
        ax.set_xlim(1995,2050);ax.set_ylim(0,100)
        ax.set_xticks([1995,2022,2050]);ax.set_yticks([0,25,50,75,100])
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=8,length=2)
        ax.grid(axis='y',color='#e2e5e7',lw=.5)
        for side in ['top','right']:ax.spines[side].set_visible(False)
        for side in ['left','bottom']:ax.spines[side].set_color('#b7bdc1')
    for ax in list(axes.flat)[31:]:ax.set_visible(False)
    handles=[Line2D([0],[0],color=gray,marker='o',ms=4,lw=.8,label='历史调查样本合并比例（构成随年变化）'),
             Line2D([0],[0],color=navy,lw=1.6,label='1995–2022 历史模型拟合（标准化）'),
             Line2D([0],[0],color=navy,lw=1.7,ls='--',label='2023–2050 预测（标准化）'),
             Patch(facecolor=gold,alpha=.23,label='趋势情景范围：φ=0.90、0.95、1.00；非置信区间')]
    fig.suptitle('中国各省青少年近视患病率：历史与至 2050 年预测',fontproperties=font,fontsize=23,x=.055,ha='left',y=.984)
    fig.text(.055,.955,'31 省份  ·  空间正则化 Logit 面板趋势模型  ·  主情景 φ=0.95',fontproperties=font,fontsize=12,color='#52616b')
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.036),ncol=2,prop=font,frameon=False)
    fig.text(.055,.018,'纵轴：患病率（%）；横轴：年份。标准化为四年级组×城乡等权、验光方式3。灰点是观测，缺年未填成实测值。',fontproperties=font,fontsize=10)
    fig.subplots_adjust(left=.055,right=.98,top=.925,bottom=.11,hspace=.52,wspace=.32)
    fig.savefig(out/'province_forecasts_300dpi.png',dpi=300,facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw,valid,panel,panels,lap,audit=read_source(args.input)
    metrics,oof,base,selections=cv(panel,lap)
    lam,losses=inner_select(panel,lap);model=fit(panel,lam,lap)
    fut=panels.merge(pd.DataFrame({'year':range(2023,2051)}),how='cross')
    fut['predicted']=predict(model,fut)
    cols=['unique_id','province','grade_group','residence','year','predicted']
    fut=fut.sort_values(['unique_id','year']).reset_index(drop=True)
    assert len(fut)==len(panels)*28 and not fut.duplicated(['unique_id','year']).any()
    assert fut.predicted.between(0,1).all() and fut.predicted.notna().all()
    pg=province_grid()
    pg['predicted']=predict(model,pg)
    pg['phi090']=predict(model,pg,.9);pg['phi100']=predict(model,pg,1.)
    prov=pg.groupby(['province','year'],as_index=False)[['predicted','phi090','phi100']].mean()
    prov['scenario_min']=prov[['predicted','phi090','phi100']].min(axis=1)
    prov['scenario_max']=prov[['predicted','phi090','phi100']].max(axis=1)
    obs=valid.groupby(['province','year'],as_index=False).agg(cases=('cases','sum'),sample_size=('sample_size','sum'))
    obs['observed']=obs.cases/obs.sample_size
    coverage=panels.merge(panel.groupby('unique_id',as_index=False).agg(
        valid_panel_years=('year','size'),first_year=('year','min'),last_year=('year','max')),on='unique_id',how='left')
    coverage['has_valid_history']=coverage.valid_panel_years.notna()
    coverage['valid_panel_years']=coverage.valid_panel_years.fillna(0).astype(int)
    observed_province_coverage=panel.groupby('province',as_index=False).agg(first_year=('year','min'),last_year=('year','max'),panel_years=('year','size'))
    for name,d in [('metrics.csv',metrics),('predictions_future.csv',fut[cols]),
                   ('cv_predictions.csv',oof),('baseline_metrics.csv',base),('cleaned_panel_year.csv',panel),
                   ('province_standardized_trends.csv',prov),('province_observed_rates.csv',obs),
                   ('panel_coverage.csv',coverage),('province_coverage.csv',observed_province_coverage)]:
        d.to_csv(out/name,index=False,encoding='utf-8-sig',float_format='%.12g')
    chart(prov,obs,out)
    coefnames=['intercept','time_decades']+[f'grade_{g}' for g in GRADES[1:]]+['urban','method_1','method_2']
    coefnames += [f'grade_{g}_urban' for g in GRADES[1:]]+[f'grade_{g}_time' for g in GRADES[1:]]+['urban_time']
    coefnames += [f'province_{p}' for p in PNAMES]+[f'province_{p}_time' for p in PNAMES]
    coefnames += [f'province_{p}_grade_{g}' for p in PNAMES for g in GRADES[1:]]+[f'province_{p}_urban' for p in PNAMES]
    assert len(coefnames)==201
    pd.DataFrame({'feature':coefnames,'coefficient':model['coef']}).to_csv(out/'model_coefficients.csv',index=False,encoding='utf-8-sig')
    byyear=valid.groupby('year',as_index=False).agg(valid_rows=('year','size'),provinces=('province','nunique'))
    byyear.to_csv(out/'year_coverage.csv',index=False,encoding='utf-8-sig')
    yy=prov.pivot(index='province',columns='year',values='predicted')
    summary=pd.DataFrame({'province':yy.index,'province_cn':[PROVINCES[x] for x in yy.index],
                          'fitted_1995':yy[1995].values,'fitted_2022':yy[2022].values,'predicted_2050':yy[2050].values})
    summary['change_1995_2022_pp']=100*(summary.fitted_2022-summary.fitted_1995)
    summary['change_2022_2050_pp']=100*(summary.predicted_2050-summary.fitted_2022)
    summary.to_csv(out/'province_2050_summary.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    mean=metrics[metrics.fold.eq('mean')].iloc[0][METRICS].astype(float).to_dict()
    oof_score=score(oof.target,oof.predicted)
    audit.update(final_lambda=lam,final_inner_mse=losses,phi=PHI,coefficient_count=201,
                 future_rows=len(fut),cv_mean=mean,cv_pooled=oof_score,
                 standardized_mean_1995=float(yy[1995].mean()),standardized_mean_2022=float(yy[2022].mean()),
                 standardized_mean_2050=float(yy[2050].mean()),
                 minimum_2050=float(yy[2050].min()),maximum_2050=float(yy[2050].max()),
                 final_fit_years=[int(panel.year.min()),int(panel.year.max())],
                 validation_horizon='two consecutive observed years per outer fold; calendar gaps vary; not 28-year validation',
                 evaluation_grain='pooled unique_id/year cells, equally weighted',
                 zero_mape_targets=int((panel.target==0).sum()),
                 province_standardization='method_code=3, equal weights across 4 grade groups x 2 residences',
                 python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
                 checks=dict(strict_temporal_separation=True,fold_count=10,all_future_panel_years=True,
                             finite_bounded_predictions=True,zero_target_not_imputed=True,
                             outer_test_not_used_for_selection=True))
    (out/'run_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'nested_cv_selection.json').write_text(json.dumps(selections,ensure_ascii=False,indent=2),encoding='utf-8')
    paired=panel.pivot(index='unique_id',columns='year',values='target')[[1995,2019]].dropna()
    paired.reset_index().to_csv(out/'balanced_1995_2019_panels.csv',index=False,encoding='utf-8-sig')
    audit['balanced_historical_comparison']=dict(matched_panels=len(paired),mean_1995=float(paired[1995].mean()),
        mean_2019=float(paired[2019].mean()),change_pp=float(100*(paired[2019]-paired[1995]).mean()))
    earlier=panel[panel.year<=2005]
    earlier_lam,_=inner_select(earlier,lap)
    earlier_model=fit(earlier,earlier_lam,lap)
    long_rows=[]
    for year in [2014,2019,2022]:
        test=panel[panel.year.eq(year)]
        for name,pr in [('spatial_damped_logit',predict(earlier_model,test)),('last_observation',persistence(earlier,test))]:
            long_rows.append(dict(model=name,train_end=2005,test_year=year,horizon_years=year-2005,
                n_test=len(test),selected_lambda=earlier_lam,**score(test.target,pr)))
    pd.DataFrame(long_rows).to_csv(out/'long_horizon_diagnostics.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    audit['long_horizon_diagnostic']='Separate train<=2005 forecasts at 2014,2019,2022; never used for model selection; not included in ten-fold mean.'
    (out/'run_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['cv_mean','cv_pooled','final_lambda','future_rows','standardized_mean_1995','standardized_mean_2022','standardized_mean_2050']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
