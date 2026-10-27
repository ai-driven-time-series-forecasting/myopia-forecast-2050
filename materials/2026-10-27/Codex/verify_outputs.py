"""Independently recalculate metrics and validate requested outputs.

python verify_outputs.py --out myopia_forecast --input input/06_myopia.zip
"""
import argparse
import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--word-reviewed',action='store_true',help='Record an actually completed separate visual review.')
    args=ap.parse_args();out=Path(args.out)
    a=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    m=pd.read_csv(out/'metrics.csv');cv=pd.read_csv(out/'cv_predictions.csv');f=pd.read_csv(out/'predictions_future.csv')
    clean=pd.read_csv(out/'cleaned_panel_year.csv')
    with zipfile.ZipFile(args.input) as z:
        raw=pd.read_csv(io.BytesIO(z.read('06_myopia/observations.csv')))
    checks={}
    checks['10_folds_and_mean_row']=len(m)==11 and m.fold.astype(str).tolist()==list(map(str,range(1,11)))+['mean']
    checks['all_cv_test_years_after_cutoff']=(cv.year>cv.train_end).all()
    for fold,g in cv.groupby('fold'):
        y=g.target.to_numpy();pr=g.predicted.to_numpy();e=y-pr
        calc={'MSE':np.mean(e**2),'RMSE':np.sqrt(np.mean(e**2)), 'MAE':np.mean(abs(e)),
              'MAPE':100*np.mean(abs(e[y!=0]/y[y!=0])),
              'R2':1-np.sum(e**2)/np.sum((y-y.mean())**2)}
        row=m[m.fold.astype(str).eq(str(fold))].iloc[0]
        checks[f'fold_{fold}_metrics_recomputed']=all(np.isclose(calc[k],row[k],rtol=1e-9,atol=1e-11) for k in calc)
    keys=['MSE','RMSE','MAE','MAPE','R2']
    checks['mean_is_unweighted_arithmetic_mean']=np.allclose(m.iloc[:10][keys].mean(),m.iloc[-1][keys].astype(float),atol=1e-10)
    checks['exact_future_columns']=list(f.columns)==['unique_id','province','grade_group','residence','year','predicted']
    checks['all_panels_28_years']=len(f)==raw.unique_id.nunique()*28 and f.groupby('unique_id').size().eq(28).all()
    checks['future_year_range']=set(f.year)==set(range(2023,2051))
    checks['unique_panel_year']=not f.duplicated(['unique_id','year']).any()
    checks['finite_bounded_no_missing_future']=np.isfinite(f.predicted).all() and f.predicted.between(0,1).all() and not f.isna().any().any()
    cols=['unique_id','province','grade_group','residence']
    expected=raw[cols].drop_duplicates().sort_values('unique_id').reset_index(drop=True)
    actual=f[cols].drop_duplicates().sort_values('unique_id').reset_index(drop=True)
    checks['future_metadata_matches_source']=actual.equals(expected)
    valid=raw[raw.valid_counts.eq(True)&raw.computed_prevalence.notna()]
    original=valid.groupby(['unique_id','year'],as_index=False).agg(cases=('cases','sum'),sample_size=('sample_size','sum'))
    observed=original.merge(clean[['unique_id','year','target']],on=['unique_id','year'],validate='1:1')
    checks['cleaned_targets_from_original_counts']=len(observed)==len(clean) and np.allclose(observed.target,observed.cases/observed.sample_size,atol=1e-11)
    checks['observed_data_and_predictions_not_mixed']=len(clean)==1791 and clean.year.max()==2022 and f.year.min()==2023
    img=Image.open(out/'province_forecasts_300dpi.png')
    checks['png_300dpi']=all(abs(v-300)<.1 for v in img.info['dpi'])
    checks['png_5700px_square']=img.size==(5700,5700)
    with zipfile.ZipFile(out/'预测说明_Codex.docx') as z:
        xml=ET.fromstring(z.read('word/document.xml'))
    text=''.join(xml.itertext())
    checks['word_contains_required_methods_metrics_uncertainty']=all(x in text for x in ['Codex','Logit','十折','MSE','RMSE','MAE','MAPE','R²','2050','不确定性'])
    checks['report_metrics_match_manifest']=all(f"{a['cv_mean'][k]:.4f}" in text for k in ['MSE','RMSE','MAE'])
    checks={k:bool(v) for k,v in checks.items()}
    assert all(checks.values()),checks
    record=dict(checks=checks,all_passed=True,word_render_review=('Microsoft Word read-only PDF export; 1 page, latest PNG visually inspected' if args.word_reviewed else 'Not performed by this verification script'),
                png_dpi=img.info['dpi'],png_dimensions=img.size)
    (out/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=True,checks_count=len(checks),future_rows=len(f),figure_dpi=img.info['dpi']),indent=2))


if __name__=='__main__': main()
