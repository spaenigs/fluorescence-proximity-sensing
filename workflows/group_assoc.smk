from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from more_itertools import chunked

import pandas as pd
import numpy as np
import altair as alt

rule all:
    input:
        "data/dyn_biosens/res/train/corr_Association Level 1.html"

rule correlation:
    input:
        "data/dyn_biosens/csv/kd_values_encoded_train.csv"
    output:
        "data/dyn_biosens/res/train/corr_Association Level 1.html"
    run:
        df = pd.read_csv(input[0], index_col=0)
        df["peptide"] = df.index

        from scipy.stats import pearsonr

        target = ["kon (M-1s-1)", "koff (s-1)", "KD (M)"]

        charts = []
        for t in target:
            def get_p(grp):
                res = pearsonr(
                    df.loc[df.group == grp, t],
                    df.loc[df.group == grp, "Association Level 1"]
                )[0]
                return np.round(res, 2)
            pr_iter_1 = get_p("iter_1")
            pr_iter_2 = get_p("iter_2")
            c = alt.Chart(
                df,
                title=f"Pearson R: {pr_iter_1} (iter 1), {pr_iter_2} (iter 2)"
            ).mark_point(
                filled=True,
                opacity=1.0,
                size=70
            ).encode(
                y="Association Level 1:Q",
                x=t + ":Q",
                color="group:N",
                tooltip="peptide:N"
            ).properties(
                height=250,
                width=250,
            )
            charts.append(c)

        alt.hconcat(*charts).save(output[0])
