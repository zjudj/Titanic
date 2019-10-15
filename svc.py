from sklearn.svm import SVC
import pandas as pd
import numpy as np
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


svc = SVC(C=10, gamma=0.0007, max_iter=-1)
sum_loss = 0
df_y_pred = pd.DataFrame()
fold = 1
for train_index, test_index in kf.split(X_train):
    kf_X_train = X_train[train_index]
    kf_y_train = y_train[train_index]
    kf_X_test = X_train[test_index]
    kf_y_test = y_train[test_index]

    svc.fit(kf_X_train, kf_y_train)
    kf_y_pred = svc.predict(kf_X_test)
    single_loss = loss(kf_y_test, kf_y_pred)
    # print(single_loss)
    sum_loss += single_loss

    y_pred = svc.predict(X_test)
    df_y_pred['fold_'+str(fold)] = y_pred
    fold += 1
print("average loss:%s" % (sum_loss/10))

df_y_pred['sum'] = df_y_pred.sum(1)

df_y_pred['sub'] = 0
df_y_pred['sub'].loc[df_y_pred['sum'] > 5] = 1

sub = pd.DataFrame(pd.read_csv('data/test.csv')['PassengerId'])
sub['Survived'] = df_y_pred['sub'].values
sub.to_csv("sub_svc.csv", index=False)
