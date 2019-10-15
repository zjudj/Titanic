#!/usr/bin/python3.6
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv(r'data/train.csv')
test_data = pd.read_csv(r'data/test.csv')
all_data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

# 查看缺失值
missing = all_data.isnull().sum()
print(missing)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.show()

# 用均值填补Fare的缺失值
all_data['Fare'].fillna(np.mean(all_data['Fare']), inplace=True)

# 处理Age的缺失值
all_data['Age'].fillna(all_data['Age'].median(), inplace=True)

# 处理Sex
all_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

# 处理Embarked
all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)

# 删除无效特征
delete_columns = ['PassengerId', 'Ticket', 'Cabin']
all_data.drop(delete_columns, axis=1, inplace=True)

# 增加特征
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = 1
all_data['IsAlone'].loc[all_data['FamilySize'] > 1] = 0

all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
all_data.drop(['Name'], axis=1, inplace=True)

all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 5)
all_data['FareBin'] = pd.qcut(all_data['Fare'], 4)

stat_min = 10
all_data['Title'] = all_data['Title'].apply(lambda x: 'Miss' if x == 'Mlle' or x == 'Ms' else x)
all_data['Title'] = all_data['Title'].apply(lambda x: 'Mrs' if x == 'Mme' else x)
title_names = (all_data['Title'].value_counts() < stat_min)
print(title_names)
all_data['Title'] = all_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)
print(all_data['Title'].value_counts())

label = LabelEncoder()
encode_features = ['AgeBin', 'FareBin']
for feature in encode_features:
    all_data[feature+'_Code'] = label.fit_transform(all_data[feature])
    all_data.drop([feature], axis=1, inplace=True)

feature_scatter = ['Embarked', 'Title']
for feature in feature_scatter:
    all_data = all_data.join(pd.get_dummies(all_data[feature], prefix=feature))
    all_data.drop([feature], axis=1, inplace=True)

train_data = all_data[:len(train_data)]
test_data = all_data[len(train_data):]

train_data.to_csv('train_data.csv', index=0)
test_data.to_csv('test_data.csv', index=0)
