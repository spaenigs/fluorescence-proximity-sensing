from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import pandas as pd
import numpy as np

df_tmp = pd.read_csv("data/fps/csv/all_kd/dist_freq_dn_100_dc_100.csv", index_col=0)
df_tmp = df_tmp.loc[df_tmp.y < 150, :].drop_duplicates()
X, y = df_tmp.iloc[:, :-1].values, df_tmp["y"].values

res = []
for train_index, test_index in LeaveOneOut().split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = RandomForestRegressor(n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    res.append([y_test[0], y_pred[0]])

df = pd.DataFrame(res, columns=["y_test", "y_pred"])

print("R2", r2_score(df.y_test, df.y_pred), "Rearson R", pearsonr(df.y_test, df.y_pred))

r2s, peas = [], []
for _ in range(1000):
    # bootstrapping
    indcs = \
        np.random.choice(df.index, size=df.shape[0], replace=True)
    df_tmp = df.loc[indcs, :]
    r2s.append(r2_score(df_tmp.y_test, df_tmp.y_pred))
    peas.append(pearsonr(df_tmp.y_test, df_tmp.y_pred)[0])


def ci95(arr):
    s = np.std(arr)
    n = len(arr)
    z = 1.96
    m = np.mean(arr)
    return m, z * (s / np.sqrt(n))


print(ci95(r2s))
print(ci95(peas))
