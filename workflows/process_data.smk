from functools import reduce

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
        expand("data/dyn_biosens/res/{linker}/res.html", linker=["left", "right"]),
        expand("data/dyn_biosens/res/left_vs_right/{target}.csv",
               target=["kon (M-1s-1)", "koff (s-1)", "KD (M)"]),
        expand("data/dyn_biosens/res/{linker}/feat_imp/data.csv",
               linker=["left", "right"]),
        expand("data/dyn_biosens/res/{linker}/res.csv", linker=["left", "right"])

rule split_sequences:
    input:
        "data/dyn_biosens/kd_values_2nd_iter.csv"
    output:
        "data/dyn_biosens/temp/kd_values_splitted.csv"
    run:
        df = pd.read_csv(input[0], index_col=0)
        df = df.loc[df["KD (M)"] > 0]

        df_splits = df.Sequence.str.split(pat="(.*?)([OJK]$|[OJK]{2,})", expand=True, n=2)[[1,2]]
        df_splits.columns = ["Sequence", "Linker"]
        df_splits["Seq_old"] = df.Sequence

        assert \
            all(df_splits.apply(lambda row: row["Sequence"] + row["Linker"] == row["Seq_old"],axis=1)) == \
            True

        df["Linker"] = df_splits.Linker
        df.Sequence = df_splits.Sequence

        cols = list(df.columns)
        cols.insert(1, cols.pop(-1))
        df = df[cols]

        # merge duplicated sequences using the median
        df_grouped = df.groupby(["Sequence", "Linker", "group"]).median().reset_index()
        df_grouped.index = df_grouped[["Sequence", "Linker"]]\
            .apply(lambda row: row[0] + row[1],axis=1)\
            .apply(lambda s: df_splits.loc[df_splits["Seq_old"] == s].index[0])\
            .values

        df_grouped["KD (M)"] = np.log(df_grouped["KD (M)"])**2
        df_grouped["koff (s-1)"] = np.log(df_grouped["koff (s-1)"]) ** 2

        df_grouped.to_csv(output[0])

rule encode_linker:
    input:
        "data/dyn_biosens/temp/kd_values_splitted.csv"
    output:
        "data/dyn_biosens/temp/{linker}/kd_values_encoded_linker.csv"
    run:
        df = pd.read_csv(input[0], index_col=0)

        linkers = list(df.Linker)
        max_linker_len = sorted([len(l) for l in linkers])[-1]

        def join_linker(l, side):
            if side == "left":
                return "".join(
                    ["10" if e == "J" else "01" if e == "O" else "00"
                     for e in "".join(["-"] * (max_linker_len - len(l))) + l]
                )
            else:
                return "".join(
                    ["10" if e == "J" else "01" if e == "O" else "00"
                     for e in l + "".join(["-"] * (max_linker_len - len(l)))]
                )

        df.Linker = df.Linker.apply(lambda l: join_linker(l, wildcards.linker))
        df.to_csv(output[0])

rule encode_data:
    input:
        "data/dyn_biosens/temp/{linker}/kd_values_encoded_linker.csv"
    output:
        "data/dyn_biosens/csv/{linker}/kd_values_encoded_train.csv",
        "data/dyn_biosens/csv/{linker}/kd_values_encoded_test.csv"
    run:
        df = pd.read_csv(input[0], index_col=0, dtype={"Linker": str})

        from iFeature import CKSAAP
        from iFeature import AAC

        fastas = [[n, s] for s, n in zip(df.Sequence, df.index)]

        # res = np.array(CKSAAP.CKSAAP(fastas,gap=2, **{"order": None}))
        res = np.array(AAC.AAC(fastas,**{"order": None}))
        df_encoded = pd.DataFrame(res[1:, 1:],columns=res[0, 1:])
        df_encoded.index = np.array(res)[1:, 0]

        selector_fitted = VarianceThreshold().fit(df_encoded)
        df_encoded = df_encoded.loc[:, selector_fitted.variances_ > 0]

        df_linkers = df.Linker.str.split("", expand=True)[range(1,len(df.Linker[0]) + 1)]
        df_linkers.columns = ["L" + str(i) for i in df_linkers.columns]

        df_res = pd.concat(
            [df_encoded, df_linkers, df[["group", "kon (M-1s-1)", "koff (s-1)", "KD (M)", "Association Level 1"]]],
            axis=1
        )

        df_train = df_res.loc[[True if "GABAAR" not in i else False for i in df_res.index]]
        df_train.to_csv(output[0])

        df_test = df_res.loc[[True if "GABAAR" in i else False for i in df_res.index]]
        df_test.to_csv(output[1])

rule regression:
    input:
        "data/dyn_biosens/csv/{linker}/kd_values_encoded_train.csv"
    output:
        "data/dyn_biosens/res/{linker}/train/res_{target}.csv",
        "data/dyn_biosens/res/{linker}/stats/stats_{target}.csv"
    run:
        df = pd.read_csv(input[0], index_col=0)

        # df.drop(index=["GlyRβ_DIM_008_5_2P", "GlyRβ_DIM_008_4_1P", "GlyRß_OCT_008_4-0-0", "GlyRß_OCT_008_3-0-1"], inplace=True)

        X = df.iloc[:, :-5].values
        y = df[wildcards.target].values

        res, indices, groups = [], [], []
        for train_index, test_index in LeaveOneOut().split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RandomForestRegressor(n_jobs=-1)
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            res.append([y_test[0], y_pred[0]])
            indices.append(df.index[test_index][0])
            groups.append(df.group[test_index[0]])

        df = pd.DataFrame(res,columns=["y_test", "y_pred"])
        df.index = indices
        df["group"] = groups
        df["level"] = "train"
        df["target"] = wildcards.target

        df["r2_mean"] = 0.0
        df["pea_mean"] = 0.0
        df["r2_ci95"] = 0.0
        df["pea_mean"] = 0.0
        df_stats = pd.DataFrame()
        for g in df.group.unique():
            r2s, peas = [], []
            for _ in range(1000):
                # bootstrapping
                indcs = \
                    np.random.choice(
                        df.loc[df.group == g, :].index,
                        size=df.shape[0],
                        replace=True
                    )
                df_tmp = df.loc[indcs, :]
                r2s.append(np.round(r2_score(df_tmp.y_test,df_tmp.y_pred),2))
                peas.append(np.round(pearsonr(df_tmp.y_test,df_tmp.y_pred)[0],2))

            df_stats = pd.concat([df_stats, pd.DataFrame({"r2s": r2s, "group": g})])

            def ci95(arr):
                s = np.std(arr)
                n = len(arr)
                z = 1.96
                m = np.mean(arr)
                return np.round(m, 2), np.round(z * (s / np.sqrt(n)), 2)

            r2_mean, r2_ci95 = ci95(r2s)
            df.loc[df.group == g, "r2_mean"] = r2_mean
            df.loc[df.group == g, "r2_ci95"] = r2_ci95

            pea_mean, pea_ci95 = ci95(peas)
            df.loc[df.group == g, "pea_mean"] = pea_mean
            df.loc[df.group == g, "pea_ci95"] = pea_ci95

        df.to_csv(output[0])
        df_stats.to_csv(output[1])

rule generate_table:
    input:
        data="data/dyn_biosens/kd_values_2nd_iter.csv",
        res=lambda wildcards:
            expand(
                f"data/dyn_biosens/res/{wildcards.linker}/train/res_{{target}}.csv",
                target=["kon (M-1s-1)", "koff (s-1)", "KD (M)",
                        # "Association Level 1"
                        ]
            )
    output:
        "data/dyn_biosens/res/{linker}/res.csv"
    run:
        df_final = reduce(
            lambda left,right: pd.merge(left,right,on='Unnamed: 0'),
            [pd.read_csv(p) for p in list(input.res)]
        )
        peptides = df_final["Unnamed: 0"]
        df_final = df_final.loc[:, [c for c in df_final.columns if "y_" in c]]
        df_final.columns = [
            "Observed Kon [M-1s-1]", "Predicted Kon [M-1s-1]", "Observed log(Koff)2 [s-1]",
            "Predicted log(Koff)2 [s-1]", "Observed log(KD)2 [M]", "Predicted log(KD)2 [M]"
        ]
        df_final.index = list(peptides)

        # import pydevd_pycharm
        # pydevd_pycharm.settrace('localhost',port=8888,stdoutToServer=True,stderrToServer=True)

        df_data = pd.read_csv(input.data)

        df_final = df_final.reindex([p for p in df_data.Peptide if p in df_final.index])
        df_final.to_csv(output[0])

rule vis:
    input:
        csv="data/dyn_biosens/csv/{linker}/kd_values_encoded_train.csv",
        res=lambda wildcards:
            expand(
                f"data/dyn_biosens/res/{wildcards.linker}/train/res_{{target}}.csv",
                target=["kon (M-1s-1)", "koff (s-1)", "KD (M)",
                        # "Association Level 1"
                        ]
            )
    output:
        "data/dyn_biosens/res/{linker}/res.html",
    run:
        df_csv = pd.read_csv(input.csv, index_col=0)
        df_csv.group = df_csv.group.apply(lambda v: "Dimeric" if v == "iter_1" else "Tetrameric & octameric")

        df = pd.concat([pd.read_csv(p,index_col=0) for p in list(input.res)])
        df["peptide"] = df.index

        cols = list(df.columns)
        cols.insert(4,cols.pop(-1))
        df = df[cols]

        df.group = df.group.apply(lambda v: "Dimeric" if v == "iter_1" else "Tetrameric & octameric")

        charts = []
        levels = list(df.level.unique())
        for i, row in enumerate(levels):
            for j, col in enumerate(sorted(df.target.unique())):

                if col == "Association Level 1":
                    continue

                df_tmp = df.loc[(df.target == col) & (df.level == row)]
                df_zero = df_tmp.iloc[0, :].copy(deep=True).to_frame().transpose()
                df_zero[["y_test", "y_pred"]] = 0, 0
                df_new = pd.concat([df_tmp, df_zero])
                c1 = alt.Chart(df_new).mark_line().encode(
                    x=alt.X("y_test:Q",title=None, axis=None),
                    y=alt.Y("y_test:Q",title=None, axis=None)
                )

                # Usage:
                # 1) Open generated HTML file in browser and save image as svg.
                # 2) Open SVG file and search for y_test, y_pred, Kd, off, on and replace hits with tspans below.
                # 3) Export SVG as png (via GIMP).
                # if "KD" in col:
                #     y_axis_title = "Predicted log(K<tspan dy='5'>D</tspan> <tspan dy='-5'>)<tspan dy='-5'>2</tspan></tspan> <tspan dy='5'> [M]</tspan>"
                #     x_axis_title = "Observed log(K<tspan dy='5'>D</tspan> <tspan dy='-5'>)<tspan dy='-5'>2</tspan></tspan> <tspan dy='5'> [M]</tspan>"
                # elif "off" in col:
                #     y_axis_title = "Predicted log(K<tspan dy='5'>off</tspan><tspan dy='-5'>)<tspan dy='-5'>2</tspan> <tspan dy='5'> [s</tspan> </tspan><tspan dy='-5' >-1</tspan><tspan dy='5'>]</tspan>"
                #     x_axis_title = "Observed log(K<tspan dy='5'>off</tspan><tspan dy='-5'>)<tspan dy='-5'>2</tspan> <tspan dy='5'> [s</tspan> </tspan><tspan dy='-5' >-1</tspan><tspan dy='5'>]</tspan>"
                # else:
                #     y_axis_title = "Predicted K<tspan dy='5'>on</tspan><tspan dy='-5'> [M</tspan><tspan dy='-5' >-1</tspan><tspan dy='5'>s<tspan dy='-5' >-1</tspan></tspan><tspan dy='5'>]</tspan>"
                #     x_axis_title = "Observed K<tspan dy='5'>on</tspan><tspan dy='-5'> [M</tspan><tspan dy='-5' >-1</tspan><tspan dy='5'>s<tspan dy='-5' >-1</tspan></tspan><tspan dy='5'>]</tspan>"

                c2 = alt.Chart(df_tmp).mark_point(filled=True, opacity=1.0, size=30).encode(
                    x=alt.X(
                        "y_test:Q",
                        axis=alt.Axis(grid=False)
                    ),
                    y=alt.Y(
                        "y_pred:Q",
                        axis=alt.Axis(grid=False)
                    ),
                    tooltip="peptide:N",
                    color=alt.Color(
                        "group:N",
                        title="Peptide",
                        scale=alt.Scale(scheme="plasma", reverse=True)
                    )
                )

                if col == "KD (M)":
                    x = 125
                    y = 20
                    y2 = 8
                    domain = [0, 250]
                elif col == "koff (s-1)":
                    x = 18
                    y = 3.3
                    y2 = 1.3
                    domain = [0, 40]
                elif col == "kon (M-1s-1)":
                    x = 38000
                    y = 6000
                    y2 = 2000
                    domain = [0, 80000]
                else:
                    x = 2.2
                    y = 3
                    y2 = 3
                    domain = [0, 250]

                def get_desc(grp):
                    res = \
                        f"R² = {df_tmp.loc[df_tmp.group == grp, 'r2_mean'].values[0]} (+-{df_tmp.loc[df_tmp.group == grp, 'r2_ci95'].values[0]})"
                    return res

                df_ct1 = pd.DataFrame({
                    "y_pred": [y, y2],
                    "y_test": [x, x],
                    "text_": [
                        get_desc("Dimeric"),
                        get_desc("Tetrameric & octameric")
                    ],
                    "group": ["Dimeric", "Tetrameric & octameric"]
                })

                c_corr_text1 = alt.Chart(df_ct1).mark_text(align="left", lineBreak="\n").encode(
                    y=alt.Y(
                        "y_pred:Q",
                        scale=alt.Scale(domain=domain),
                        axis=None
                    ),
                    x=alt.X(
                        "y_test:Q",
                        scale=alt.Scale(domain=domain),
                        axis=None
                    ),
                    text="text_:N",
                    color="group:N"
                )

                c = alt.layer(
                    c2,
                    c_corr_text1,
                    title="ABC"[j],
                ).resolve_scale(
                    x='independent',
                    y='independent'
                ).properties(
                    height=200,
                    width=200,
                ).interactive()


                def get_p(grp):
                    res = pearsonr(
                        df_csv.loc[df_csv.group == grp, col],
                        df_csv.loc[df_csv.group == grp, "Association Level 1"]
                    )[0]
                    return np.round(res,2)


                pr_iter_1 = get_p("Dimeric")
                pr_iter_2 = get_p("Tetrameric & octameric")

                if col == "KD (M)":
                    x = 13
                elif col == "koff (s-1)":
                    x = 22
                elif col == "kon (M-1s-1)":
                    x = 54000
                else:
                    x = 3

                df_ct = pd.DataFrame({
                    "Association Level 1": [37 , 35],
                    col: [x, x],
                    "text_": [f"Pearson R = {pr_iter_1}", f"Pearson R = {pr_iter_2}"],
                    "group": ["Dimeric", "Tetrameric & octameric"]
                })

                c_corr_text = alt.Chart(df_ct).mark_text(align="left").encode(
                    y=alt.Y("Association Level 1:Q", axis=alt.Axis(grid=False)),
                    x=alt.X(col + ":Q", axis=alt.Axis(grid=False)),
                    text="text_:N",
                    color="group:N"
                )

                c_corr = alt.Chart(df_csv).mark_point(
                    filled=True,
                    opacity=1.0,
                    size=30
                ).encode(
                    y=alt.Y("Association Level 1:Q", title="Association Level"),
                    x=col + ":Q",
                    color="group:N",
                    tooltip="peptide:N"
                ).properties(
                    height=200,
                    width=200,
                )

                charts.append(alt.vconcat(c, alt.layer(
                    c_corr,
                    c_corr_text,
                    title="DEF"[j]
                )))

        alt.concat(
            *charts, columns=4, spacing=20
        ).configure_axis(
            titleFont="Arial",
            titleFontSize=10
        ).configure_title(
            anchor="start",
            font="Arial",
            fontSize=16,
            fontWeight="normal"
        ).save(output[0])

rule stats:
    input:
        "data/dyn_biosens/res/left/stats/stats_{target}.csv",
        "data/dyn_biosens/res/right/stats/stats_{target}.csv"
    output:
        "data/dyn_biosens/res/left_vs_right/{target}.csv"
    run:
        from scipy.stats import ttest_rel, ttest_ind

        df_left = pd.read_csv(input[0], index_col=0)
        df_right = pd.read_csv(input[1], index_col=0)

        res = []
        for g in df_left.group.unique():
            ttest_res = ttest_ind(
                df_left.loc[df_left.group == g, "r2s"],
                df_right.loc[df_right.group == g, "r2s"]
            )
            res.append([wildcards.target, g, ttest_res[1]])

        df_res = pd.DataFrame(res, columns=["target", "group", "p-value"])
        df_res.to_csv(output[0])

rule feature_importance:
    input:
        "data/dyn_biosens/csv/{linker}/kd_values_encoded_train.csv"
    output:
        "data/dyn_biosens/res/{linker}/feat_imp/data.csv",
        "data/dyn_biosens/res/{linker}/feat_imp/vis.html"
    run:
        df_train = pd.read_csv(input[0], index_col=0)

        charts = []
        for target in sorted(df_train.columns[-4:]):

            X_train = df_train.iloc[:, :-5].values
            y_train = df_train[target].values

            clf = RandomForestRegressor(n_jobs=-1)
            clf.fit(X_train,y_train)

            features = df_train.iloc[:, :-5].columns
            # features = list(features[:-14]) + [f"L{i}" for i in range(1,8)]
            importances = clf.feature_importances_
            # importances = list(importances[:-14]) + [sum(p) for p in chunked(importances[-14:],2)]

            df_feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': importances,
                "level": "feat_imp",
                "target": target
            })

            df_feat_imp.to_csv(output[0])

            c = alt.Chart(df_feat_imp, title=target).mark_bar().encode(
                x=alt.X('Feature',sort=alt.SortArray(list(df_feat_imp.columns))),
                y=alt.Y('Importance',scale=alt.Scale(domain=[0.0, 1.0]))
            ).properties(
                height=250,
                width=300,
            )

            df_tmp = df_train.copy(deep=True)
            old_columns = df_tmp.columns[15:-5]
            for cols in chunked(df_train.columns[15:-5],2):
                df_tmp[cols] = \
                    df_train.loc[:, cols].apply(
                        lambda row:
                            ["J", "J"] if all(row == [1, 0]) else
                            ["O", "O"] if all(row == [0, 1]) else
                            ["-", "-"],
                        axis=1,
                        result_type="expand"
                    )

            x, y = np.meshgrid(df_tmp[old_columns].columns,df_tmp.index)
            z = df_tmp[old_columns].values

            source = pd.DataFrame({'x': x.ravel(),
                                   'y': y.ravel(),
                                   'z': z.ravel()})

            c2 = alt.Chart(source).mark_rect().encode(
                x=alt.X('x:O',  title=None, sort=alt.SortArray(list(
                    df_tmp[old_columns].columns
                ))),
                y=alt.Y('y:O', title=None),
                color='z:N'
            ).properties(
                height=600,
                width=300
            )

            df_tmp = df_train.copy(deep=True)
            old_columns = df_tmp.columns[15:-5]
            new_columns = []
            for cols in chunked(df_train.columns[15:-5],2):
                new_col = "-".join(cols)
                new_columns.append(new_col)
                df_tmp[new_col] = \
                    df_train.loc[:, cols].apply(
                        lambda row:
                            "J" if all(row == [1, 0]) else
                            "O" if all(row == [0, 1]) else
                            "-",
                        axis=1,
                        result_type="expand"
                    )

            df_tmp.drop(columns=df_tmp[old_columns].columns, inplace=True)

            x, y = np.meshgrid(df_tmp[new_columns].columns, df_tmp.index)
            z = df_tmp[new_columns].values

            source = pd.DataFrame({'x': x.ravel(),
                                   'y': y.ravel(),
                                   'z': z.ravel()})

            c3 = alt.Chart(source).mark_rect().encode(
                x=alt.X('x:O', title=None, sort=alt.SortArray(list(
                    df_tmp[new_columns].columns
                ))),
                y=alt.Y('y:O', title=None),
                color='z:N'
            ).properties(
                height=600,
                width=150
            )

            charts.append((c & c2 & c3).resolve_scale(
                color="independent"
            ))

        alt.hconcat(*charts).save(output[1])