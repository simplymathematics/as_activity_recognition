import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pathlib

import pandas as pd
from data.pre_process import PATH_FEATURES, PATH_TIME_WINDOWS
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
save_fig_dir = "figures"

if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)


features_train = pd.read_csv(pathlib.Path(PATH_FEATURES, "train.csv"))
y_train = np.load(str(pathlib.Path(PATH_TIME_WINDOWS, "data_train_y.npy")))
features_test = pd.read_csv(pathlib.Path(PATH_FEATURES, "test.csv"))
y_test = np.load(str(pathlib.Path(PATH_TIME_WINDOWS, "data_test_y.npy")))

columns_order = features_train.columns.values
X_train = features_train[columns_order]
X_test = features_test[columns_order]


## Visualize data in 2D

X_merged = pd.concat((X_train, X_test))
Y_merged = np.hstack((y_train, y_test))

pca = PCA(n_components=2)
pca.fit(X_merged.T)
X_merged_2d = pca.components_.T

plt.figure()
plt.plot(X_merged_2d[Y_merged == 0, 0], X_merged_2d[Y_merged == 0, 1], '*r', label='Running')
plt.plot(X_merged_2d[Y_merged == 2, 0], X_merged_2d[Y_merged == 2, 1], '*g', label='Walking')
plt.plot(X_merged_2d[Y_merged == 1, 0], X_merged_2d[Y_merged == 1, 1], '*b', label='Standing')

plt.title("Data Visualization in 2D (PCA)")
plt.ylabel('2nd PCA Basis')
plt.xlabel('1st PCA Basis')
plt.legend()
plt.savefig(os.path.join(save_fig_dir, "PCA_vis.png"), bbox_inches='tight', pad_inches=0)

n_folds = 5
n_runs = 5
kf = KFold(n_splits=n_folds)
scores_dim = []

## Accuracy VS PCA dimensionality

dims = [i for i in range(353, 1, -13)] + [1]
for dim in dims:

    scores_dim.append(0)

    pca = PCA(n_components=dim)
    pca.fit(X_merged.T)
    X_merged_d = pca.components_.T

    for n_run in range(n_runs):
        for train_idx, test_idx in kf.split(X_merged):

            clf = RandomForestClassifier()
            clf.fit(X_merged_d[train_idx], Y_merged[train_idx])
            scores_dim[-1] += clf.score(X_merged_d[test_idx], Y_merged[test_idx])

    scores_dim[-1] /= n_folds * n_runs

plt.figure()
plt.title("Data Dimensionality vs Accuracy")
plt.plot(dims, scores_dim, '-o')
plt.ylabel('Accuracy')
plt.xlabel('PCA-dims')

plt.savefig(os.path.join(save_fig_dir, "PCA_dim_acc.png"), bbox_inches='tight', pad_inches=0)

