#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate , train_test_split , StratifiedShuffleSplit
from sklearn import linear_model, metrics
from sklearn.linear_model import SGDClassifier as SGD

train_directory = "data/train.csv"

train = pd.read_csv(train_directory, header=0 )

# exclude 'nr.employed' & 'euribor3m' \
# cause correlation between \
# 'nr.employed' & 'euribor3m' & 'emp.var.rate' is to HIGH (~.9)
# see pic 'corr.bmp'
train = train.drop((['nr.employed', 'euribor3m']), axis=1)
train['y'] = [1 if y == 'yes' else 0 for y in train['y']]


categorical_columns = [c for c in train.columns if train[c].dtype.name == 'object']
numerical_columns   = [c for c in train.columns if train[c].dtype.name != 'object']

full_train_cats = pd.get_dummies(train[categorical_columns])

train = pd.concat((train[numerical_columns], full_train_cats), axis=1)
train = pd.DataFrame(train, dtype=float)

X = train.drop(('y'), axis=1)
y = train['y']

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2)

# learn forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_data, train_labels)

# get errors
err_train = np.mean(train_labels != rf.predict(train_data))
err_test = np.mean(test_labels != rf.predict(test_data))
print('Errors *train* *test*:')
print(err_train, err_test)

# choose important features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

#########################################
# !! here shows graphic, where we see \ #
# importances of features               #
# see pic 'features_importances.bmp'    #
#########################################
'''
columns = X.columns
d_first = 58
plt.figure(figsize=(8, 8))
plt.title("Feature importances")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(columns)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first])
plt.show()
'''
# build new models with different count of features \
# taken from previous model 
# result in graphic "count_features.bmp"

##############################################
# !! here shows graphic, where we see \      #
# dependence of test_error on features_count #
##############################################
'''
width = 50

total_res = []
for features_count in range(1, 58+1):
       res_feature = []
       for _ in range(1, width+1):
              # select first features
              selected_features = X.columns[indices[:features_count]]
              X_selected = train[selected_features]

              train_data_selected, test_data_selected, train_labels_selected, test_labels_selected = train_test_split(X_selected, y, test_size = 0.2)

              # learn new forest
              rf_selected = RandomForestClassifier(n_estimators=100, random_state=11)
              rf_selected.fit(train_data_selected, train_labels_selected)

              # get errors
              err_train_selected = np.mean(train_labels_selected != rf_selected.predict(train_data_selected))
              err_test_selected = np.mean(test_labels_selected != rf_selected.predict(test_data_selected))
              # print(f'Errors (selected {features_count}) *train* *test*:')
              # print(features_count, '\t:',err_train_selected, err_test_selected)
              res_feature.append(err_test_selected)
       total_res.append(sum(res_feature)/len(res_feature))

plt.plot(range(len(total_res)), total_res)
plt.show()
'''
# Summary: from 10+ features taken \
# there are no significant test_error difference

# Build new model on 10 significant features of previous model
features_count = 10
selected_features = X.columns[indices[:features_count]]
X_selected = train[selected_features]

train_data_selected, test_data_selected, train_labels_selected, test_labels_selected = train_test_split(X_selected, y, test_size = 0.2)

rf_selected = RandomForestClassifier(n_estimators=100)
rf_selected.fit(train_data_selected, train_labels_selected)

err_train_selected = np.mean(train_labels_selected != rf_selected.predict(train_data_selected))
err_test_selected = np.mean(test_labels_selected != rf_selected.predict(test_data_selected))

print(err_train_selected, err_test_selected)

print('###############################')
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
scores = cross_validate(rf_selected, X_selected, y=y, cv=sss)
print(scores)

pred_labels = rf_selected.predict(test_data_selected)

roc_auc = metrics.roc_auc_score(test_labels_selected, pred_labels)
print(roc_auc)

sss_orig = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=7)
scores_orig = cross_validate(rf, X, y=y, cv=sss_orig)
print(scores_orig)

pred_labels_orig = rf.predict(test_data)

roc_auc_orig = metrics.roc_auc_score(test_labels, pred_labels_orig)
print(roc_auc_orig)
exit()

classifier.fit(train_data , train_labels)

pred_labels = classifier.predict(test_data)

auc = metrics.roc_auc_score(test_labels, pred_labels)

