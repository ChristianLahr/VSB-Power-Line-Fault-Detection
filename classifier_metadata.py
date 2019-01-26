import pandas as pd
import  numpy as np

import os, gc
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from scipy.stats import expon, uniform, norm
from scipy.stats import randint, poisson

rand_seed =23
np.random.seed(rand_seed)
xsize = 12.0
ysize = 8.0

dir_train = 'assets/original_kaggle/train'
n_train_samples = 10000

train_metadata = pd.DataFrame.from_csv(dir_train + '/metadata_train_incl_extraFeatures.csv', sep=';')

feature_names = ["phase"] + train_metadata.columns[4:].tolist()
print('features:', feature_names, '\n')

### importance of features
Fvals, pvals = f_classif(train_metadata[feature_names], train_metadata["target"])

print("F-value | P-value | Feature Name")
print("--------------------------------")

for i, col in enumerate(feature_names):
    print("%.4f"%Fvals[i]+" | "+"%.4f"%pvals[i]+" | "+col)

feature_names_important = feature_names
# choose features wih P-values > 0.01
for i, col in enumerate(feature_names):
    if pvals[i] <= 0.01:
        feature_names_important.remove(col)

print('\nchosen features:', feature_names_important)


### classifier
def mcc(y_true, y_pred, labels=None, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight).ravel()
    mcc = (tp*tn - fp*fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    return mcc

mcc_scorer = make_scorer(mcc)

from sklearn.metrics import f1_score
f1_scorer = f1_score # make_scorer(f1_score)

import lightgbm as lgbm
lgbm_classifier = lgbm.LGBMClassifier(boosting_type='gbdt', max_depth=-1, subsample_for_bin=200000, objective="binary",
                                      class_weight=None, min_split_gain=0.0, min_child_weight=0.001, subsample=1.0,
                                      subsample_freq=0, random_state=rand_seed, n_jobs=1, silent=True, importance_type='split')

# from sklearn.svm import SVC
# svm_classifier = SVC(gamma='auto')
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train_metadata[feature_names_important], train_metadata["target"], test_size=0.5)
# svm_classifier.fit(X_train, y_train)
# p_test = svm_classifier.predict(X_test)
# p_test.sum() --> 0
# mcc(y_test, p_test)

param_distributions = {
    "num_leaves": randint(16, 48),
    "learning_rate": expon(),
    "reg_alpha": expon(),
    "reg_lambda": expon(),
    "colsample_bytree": uniform(0.25, 1.0),
    "min_child_samples": randint(10, 30),
    "n_estimators": randint(50, 250)
}

clf = RandomizedSearchCV(lgbm_classifier, param_distributions, n_iter=100, scoring=mcc_scorer, fit_params=None, n_jobs=1, iid=True,
                         refit=True, cv=5, verbose=1, random_state=rand_seed, error_score=-1.0, return_train_score=True)
clf.fit(train_metadata[feature_names_important], train_metadata["target"])

print(clf.best_score_)
clf_best = clf.best_estimator_
fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)
lgbm.plot_importance(clf.best_estimator_, ax=ax)
plt.show()

clf.predict(X_test)
