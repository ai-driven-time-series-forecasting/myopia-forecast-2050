import sys
sys.path.insert(0, "C:/Users/echo_/OneDrive/Desktop/近视预测_2050")
import pandas as pd
from myopia_lib import *

df = load_data()
sun, lag = build_static_spatial(df)

configs = {
 'A_base':      dict(),
 'B_graderes':  dict(grade_slope=True, res_slope=True),
 'C_full':      dict(grade_slope=True, res_slope=True, prov_slope=True),
 'D_quad':      dict(grade_slope=True, res_slope=True, quad=True),
}
for name, cfg in configs.items():
    folds, mean = forward_chaining_cv(df, cfg, sun, lag)
    print(f"=== {name} ===  n_folds={len(folds)}")
    print(mean.round(5).to_string())
    print()
