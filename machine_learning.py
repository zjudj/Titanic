#!/usr/bin/python3.6
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def loss(actual, predicted):
    d = np.abs(actual - predicted)
    return (len(actual) - np.sum(d))/len(actual)


train_data = pd.read_csv(r'train_data.csv')
test_data = pd.read_csv(r'test_data.csv')

y_train = train_data['Survived'].values
X_train = train_data.drop('Survived', axis=1).values
X_test = test_data.drop('Survived', axis=1).values

kf = KFold(n_splits=10, random_state=2019)
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, bootstrap=True, max_features=0.4)
sum_loss = 0
df_y_pred = pd.DataFrame()
fold = 1
for train_index, test_index in kf.split(X_train):
    kf_X_train = X_train[train_index]
    kf_y_train = y_train[train_index]
    kf_X_test = X_train[test_index]
    kf_y_test = y_train[test_index]

    rf.fit(kf_X_train, kf_y_train)
    kf_y_pred = rf.predict(kf_X_test)
    single_loss = loss(kf_y_test, kf_y_pred)
    print(single_loss)
    sum_loss += single_loss

    y_pred = rf.predict(X_test)
    df_y_pred['fold_'+str(fold)] = y_pred
    fold += 1
print("average loss:%s" % (sum_loss/10))

df_y_pred['sum'] = df_y_pred.sum(1)

print((df_y_pred['sum'] > 0) & (df_y_pred['sum'] < 10))
test_data_2 = test_data.loc[(df_y_pred['sum'] > 0) & (df_y_pred['sum'] < 10)]
print(test_data_2, len(test_data_2))

print('ok')
sub = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
sub['Survived'] = list(map(int, ))
sub.to_csv("sub_rf.csv", index=False)

