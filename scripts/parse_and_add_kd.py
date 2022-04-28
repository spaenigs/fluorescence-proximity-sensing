import pandas as pd

from glob import glob

import os

kd_values = pd.read_csv("data/fps/kd_values.csv", index_col=0)

for p in glob("data/fps/csv/all/*.csv"):
    if not os.path.exists(p.replace("all", "all_kd")):
        df = pd.read_csv(p, index_col=0)
        df_tmp = df.loc[df.y == 1].copy()
        for n, s in df_tmp.iterrows():
            _, _, name = n.split("_")
            tmp = kd_values.loc[kd_values.Sequence == name, "KD (M)"]
            kd, std_ = [float(b.replace(" ", "").replace("E", "e"))
                        for b in tmp.str.split(" ± ", expand=True).iloc[0, :]]
            df_tmp.loc[n, "y"] = kd
        df_tmp.to_csv(p.replace("all", "all_kd"))
    else:
        print("skipping")
