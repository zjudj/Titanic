#!/usr/bin/python3.6
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
age_avg = all_data['Age'].mean()
age_std = all_data['Age'].std()
all_data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

# 处理Sex
all_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

# 处理Embarked
all_data['Embarked'].fillna('S', inplace=True)
all_data['Embarked'] = all_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 删除无效特征
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
all_data.drop(delete_columns, axis=1, inplace=True)

train_data = all_data[:len(train_data)]
test_data = all_data[len(train_data):]

y_train = train_data['Survived']
X_train = train_data.drop('Survived', axis=1)
X_test = test_data.drop('Survived', axis=1)
y_train.to_csv('y_train.csv', index=0)
X_train.to_csv('X_train.csv', index=0)
X_test.to_csv('X_test.csv', index=0)
