##
import calendar
from typing import Any

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
##
df = pd.read_csv(filepath_or_buffer="./bank/bank-additional-full.csv", sep=";")
df_test = pd.read_csv(filepath_or_buffer="./bank/bank-additional.csv", sep=";")
# set description
df.info()
print(df.describe().to_string())
##
categorical= ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
numerical=['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
outputs= 'y'
##
sns.countplot(x=df['y'], palette='pastel', edgecolor='.6')
plt.title('Distribution of classes')
# we have a dataset, for which the number of results is highly imbalanced
##
"""duration: last contact duration, in seconds (numeric). Important note:
  this attribute highly affects the output target (e.g., if duration=0 then y="no").
   Yet, the duration is not known before a call is performed. Also, after the end of 
   the call y is obviously known. Thus, this input should only be included for benchmark 
   purposes and should be discarded if the intention is to have a realistic predictive model."""
df.drop('duration', axis=1, inplace=True)
##
# data visualization
for x in df.loc[:, df.columns != 'additional']:
    # print(x)
    df['additional'] = 0
    test = df.groupby(by=x).agg({'additional': 'size'}).reset_index()
    test.rename(columns={'additional': 'total'}, inplace=True)
    plt.bar(test[x], test['total'], align='center')
    plt.title(x + ' pre preprocess')
    if x in ('job', 'education'):
        plt.xticks(rotation=30, fontsize=7)
    plt.show()
df.drop('additional', axis=1, inplace=True)
##
# Preprocessing To count correlations between columns and create models, we need to encode categorical values to numbers.
# Date columns will be mapped in numbers to preserve order. Other categorical values will be one-hot encoded. If there
# is few values of some category they will be chanded into other value. The Point of it is to reduce the number of
# columns after one-hot encoding. Changes in columns:

df.loc[(df.default == 'yes'), 'default'] = 'unknown'
df.default = (df.default == 'unknown') * 1
df.loc[(df.housing == 'unknown'), 'housing'] = 'no'
df.housing = (df.housing == 'yes') * 1
df.loc[(df.loan == 'unknown'), 'loan'] = 'no'
df.loan = (df.loan == 'yes') * 1
df.loc[(df.marital == 'unknown'), 'marital'] = 'married'
df.contact = (df.contact == 'telephone') * 1
df.y = (df.y == 'yes') * 1
df.month = [list(calendar.month_abbr).index(item.title()) for item in df.month]
df.day_of_week = [list(calendar.day_abbr).index(item.title()) + 1 for item in df.day_of_week]

enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(df[['marital']]).toarray(), columns=enc.categories_[0])
df = df.join(enc_df)
df = df.drop('marital', axis=1)

enc_df = pd.DataFrame(enc.fit_transform(df[['job']]).toarray(), columns=enc.categories_[0])
df = df.join(enc_df)
df = df.drop('job', axis=1)

enc_df = pd.DataFrame(enc.fit_transform(df[['education']]).toarray(),
                      columns=['education' + '_' + sub for sub in enc.categories_[0]])
df = df.join(enc_df)
df = df.drop('education', axis=1)

enc_df = pd.DataFrame(enc.fit_transform(df[['poutcome']]).toarray(),
                      columns=['poutcome' + '_' + sub for sub in enc.categories_[0]])
df = df.join(enc_df)
df = df.drop('poutcome', axis=1)
##
# Quantile preprocessing
# https://towardsdatascience.com/the-definitive-way-to-deal-with-continuous-variables-in-machine-learning-edb5472a2538
df_quantiles = df.copy()
df_quantiles['additional'] = 0
info = df_quantiles.groupby(by='pdays').agg({'additional': 'size'}).reset_index()
info.rename(columns={'additional': 'total'}, inplace=True)
print(info)
# 0 - not contacted before
# 1 - contacted before
df_quantiles['contacted_before'] = np.where(df_quantiles['pdays'] == 999, 0, 1)
info = df_quantiles.groupby(by='contacted_before').agg({'additional': 'size'}).reset_index()
info.rename(columns={'additional': 'total'}, inplace=True)
print(info)
df_quantiles.drop('pdays', axis=1, inplace=True)
df_quantiles.drop('additional', axis=1, inplace=True)
# Since almost all entries are for the custommers, who never have been conntacted. To preprocess the data we change it
# into discrete type with two possible values 0 means, customer has never been contacted, 1 means one has been contacted
##
set = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
for x in set:
    vec = np.arange(0.00, 1, 0.05)
    if x in ['emp.var.rate', 'nr.employed']:
        vec = np.arange(0.05, 1, 0.05)
    print(x)
    quantiles = np.quantile(df_quantiles[x], vec)
    quantiles = pd.DataFrame(np.squeeze(quantiles), columns=['values'])
    quantiles['total_percentage'] = 0
    quantiles = quantiles.groupby(by='values').agg({'total_percentage': 'size'}).reset_index()
    quantiles['total_percentage'] *= 5
    quantiles['total_percentage'] = quantiles['total_percentage'].cumsum() - quantiles['total_percentage']
    print(quantiles)
    df_quantiles[x] = pd.qcut(df_quantiles[x].values, 20, duplicates='drop', labels=quantiles['total_percentage'].to_numpy())
    df_quantiles[x] = df_quantiles[x].astype('int64')
##
df_quantiles['additional'] = 0
for x in df_quantiles.loc[:, df_quantiles.columns != 'additional']:
    # print(x)
    test = df_quantiles.groupby(by=x).agg({'additional': 'size'}).reset_index()
    test.rename(columns={'additional': 'total'}, inplace=True)
    plt.bar(test[x], test['total'], align='center')
    plt.title(x + ' post quantile preprocess')
    if x in ('job', 'education'):
        plt.xticks(rotation=30, fontsize=9)
    plt.show()
df_quantiles.drop('additional', axis=1, inplace=True)
##
# visualization of the data after preprocessing
print(df_quantiles.columns)

with pd.option_context('display.max_columns', 42):
  print(df_quantiles.describe(include='all'))

mask = np.zeros_like(df_quantiles.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('ticks'):
    f, ax = plt.subplots(figsize=(12,12))
    ax = sns.heatmap(df.corr(method ='pearson'), mask=mask, vmax=.3,annot=True,fmt=".0%",linewidth=0.5,square=False,cmap='Purples')
    plt.title('Correlation matrix')

df_correlations = df_quantiles.corr().copy()
for x in df_correlations.columns:
    print(x)
    print(df_correlations[x].sort_values(ascending=False)[1:])
    print("\n")
##
# standard scale preprocessing
set = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
sc = StandardScaler()
df_standard_scale = df.copy()
for x in set:
    df_standard_scale[x] =sc.fit_transform(df_standard_scale[[x]])
##
# visualization after standard scale preprocessing
df_standard_scale['additional'] = 0
for x in df_standard_scale.loc[:, df_standard_scale.columns != 'additional']:
    # print(x)
    test = df_standard_scale.groupby(by=x).agg({'additional': 'size'}).reset_index()
    test.rename(columns={'additional': 'total'}, inplace=True)
    plt.bar(test[x], test['total'], align='center')
    plt.title(x + ' post standard scale preprocess')
    if x in ('job', 'education'):
        plt.xticks(rotation=30, fontsize=9)
    plt.show()
df_standard_scale.drop('additional', axis=1, inplace=True)
##
mask = np.zeros_like(df_standard_scale.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('ticks'):
    f, ax = plt.subplots(figsize=(12,12))
    ax = sns.heatmap(df.corr(method ='pearson'), mask=mask, vmax=.3,annot=True,fmt=".0%",linewidth=0.5,square=False,cmap='Purples')
    plt.title('Correlation matrix')
##
df_correlations = df_standard_scale.corr().copy()
for x in df_correlations.columns:
    print(x)
    print(df_correlations[x].sort_values(ascending=False)[1:])
    print("\n")
##

# Data modeling

# Logistics regression
##
# numerical values not preprocessed
X = df.drop(labels='y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)
C_set = [0.01, 1, 1000]
for c in C_set:
    lr = LogisticRegression(random_state=0, C=c, max_iter=10000)
    lr.fit(X_train, y_train)
    print("for C = " + str(c))
    print("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_train, lr.predict(X_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_test, lr.predict(X_test)))
##
# numerical values preprocessed using quantalization
X_quantiles = df_quantiles.drop(labels='y', axis=1)
y_quantiles = df_quantiles['y']
X_quantiles_train, X_quantiles_test, y_quantiles_train, y_quantiles_test = train_test_split(X_quantiles, y_quantiles, stratify=y_quantiles, test_size=0.3, random_state=0)
for c in C_set:
    lr = LogisticRegression(random_state=0, C=c, max_iter=10000)
    lr.fit(X_quantiles_train, y_quantiles_train)
    print("for C = " + str(c))
    print("Train - Accuracy :", metrics.accuracy_score(y_quantiles_train, lr.predict(X_quantiles_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_train, lr.predict(X_quantiles_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_quantiles_test, lr.predict(X_quantiles_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_test, lr.predict(X_quantiles_test)))
##
# numerical columns preprocessed using standardization with the function provided by sklearn
X_standard_scale = df_standard_scale.drop(labels='y', axis=1)
y_standard_scale = df_standard_scale['y']
X_standard_scale_train, X_standard_scale_test, y_standard_scale_train, y_standard_scale_test = \
    train_test_split(X_standard_scale, y_standard_scale, stratify=y_standard_scale, test_size=0.3, random_state=0)
for c in C_set:
    lr = LogisticRegression(random_state=0, C=c, max_iter=10000)
    lr.fit(X_standard_scale_train, y_standard_scale_train)
    print("for C = " + str(c))
    print("Train - Accuracy :", metrics.accuracy_score(y_standard_scale_train, lr.predict(X_standard_scale_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_train, lr.predict(X_standard_scale_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_standard_scale_test, lr.predict(X_standard_scale_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_test, lr.predict(X_standard_scale_test)))
##
# Decision tree
Crit =["gini", "entropy"]
min_samples_split_number = [2, 4, 8]
##
# numerical values not preprocessed
for min_samples_split_n in min_samples_split_number:
  print("min_samples_split = " + str(min_samples_split_n))
  for crit in Crit:
    clf = tree.DecisionTreeClassifier(random_state=0, criterion=crit, min_samples_split = min_samples_split_n)
    clf.fit(X_train, y_train)
    print("citerion " + crit)
    print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_train, clf.predict(X_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_test, clf.predict(X_test)))
##
# numerical values preprocessed using quantalization
for min_samples_split_n in min_samples_split_number:
  print("min_samples_split = " + str(min_samples_split_n))
  for crit in Crit:
      clf = tree.DecisionTreeClassifier(random_state=0, criterion=crit, min_samples_split = min_samples_split_n)
      clf.fit(X_quantiles_train, y_quantiles_train)
      print("citerion " + crit)
      print("Train - Accuracy :", metrics.accuracy_score(y_quantiles_train, clf.predict(X_quantiles_train)))
      print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_train, clf.predict(X_quantiles_train)))
      print("Test - Accuracy :", metrics.accuracy_score(y_quantiles_test, clf.predict(X_quantiles_test)))
      print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_test, clf.predict(X_quantiles_test)))
##
# numerical columns preprocessed using standardization with the function provided by sklearn
for min_samples_split_n in min_samples_split_number:
  print("min_samples_split = " + str(min_samples_split_n))
  for crit in Crit:
      clf = tree.DecisionTreeClassifier(random_state=0, criterion=crit, min_samples_split = min_samples_split_n)
      clf.fit(X_standard_scale_train, y_standard_scale_train)
      print("citerion " + crit)
      print("Train - Accuracy :", metrics.accuracy_score(y_standard_scale_train, clf.predict(X_standard_scale_train)))
      print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_train, clf.predict(X_standard_scale_train)))
      print("Test - Accuracy :", metrics.accuracy_score(y_standard_scale_test, clf.predict(X_standard_scale_test)))
      print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_test, clf.predict(X_standard_scale_test)))
##
# K-Nearest Neighbors
number_neighbors = [2, 5, 10]
algorithm_set = ['auto', 'ball_tree', 'kd_tree', 'brute']
##
# numerical values not preprocessed
for x in number_neighbors:
  for alg in algorithm_set:
    print("n_neighbors: " + str(x) + " algorithm: " + alg)
    clf = KNeighborsClassifier(n_neighbors=x, p=2, metric='minkowski', algorithm=alg)
    clf.fit(X_train, y_train)
    print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_train, clf.predict(X_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_test, clf.predict(X_test)))
    print("\n")
##
# numerical columns preprocessed using quantalization
for x in number_neighbors:
  for alg in algorithm_set:
    print("n_neighbors: " + str(x) + " algorithm: " + alg)
    clf = KNeighborsClassifier(n_neighbors=x, p=2, metric='minkowski', algorithm=alg)
    clf.fit(X_quantiles_train, y_quantiles_train)
    print("Train - Accuracy :", metrics.accuracy_score(y_quantiles_train, clf.predict(X_quantiles_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_train, clf.predict(X_quantiles_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_quantiles_test, clf.predict(X_quantiles_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_test, clf.predict(X_quantiles_test)))
    print("\n")
##
# numerical columns preprocessed using standardization with the function provided by sklearn
for x in number_neighbors:
  for alg in algorithm_set:
    print("n_neighbors: " + str(x) + " algorithm: " + alg)
    clf = KNeighborsClassifier(n_neighbors=x, p=2, metric='minkowski', algorithm=alg)
    clf.fit(X_standard_scale_train, y_standard_scale_train)
    print("Train - Accuracy :", metrics.accuracy_score(y_standard_scale_train, clf.predict(X_standard_scale_train)))
    print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_train, clf.predict(X_standard_scale_train)))
    print("Test - Accuracy :", metrics.accuracy_score(y_standard_scale_test, clf.predict(X_standard_scale_test)))
    print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_test, clf.predict(X_standard_scale_test)))
    print("\n")
##
# MLPClassifier
solvers = ['lbfgs', 'sgd', 'adam']
##
# numerical columns not preprocessed
for x in solvers:
  print("solver: " + x)
  clf = MLPClassifier(random_state=1, max_iter=1000, solver=x).fit(X_train, y_train)
  print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
  print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_train, clf.predict(X_train)))
  print("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
  print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_test, clf.predict(X_test)))
  print('\n')
##
# numerical columns preprocessed using quantalization
for x in solvers:
  print("solver: " + x)
  clf = MLPClassifier(random_state=1, max_iter=1000, solver=x).fit(X_quantiles_train, y_quantiles_train)
  print("Train - Accuracy :", metrics.accuracy_score(y_quantiles_train, clf.predict(X_quantiles_train)))
  print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_train, clf.predict(X_quantiles_train)))
  print("Test - Accuracy :", metrics.accuracy_score(y_quantiles_test, clf.predict(X_quantiles_test)))
  print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_quantiles_test, clf.predict(X_quantiles_test)))
  print('\n')
##
# numerical columns preprocessed using standardization with the function provided by sklearn
for x in solvers:
  print("solver: " + x)
  clf = MLPClassifier(random_state=1, max_iter=1000, solver=x).fit(X_standard_scale_train, y_standard_scale_train)
  print("Train - Accuracy :", metrics.accuracy_score(y_standard_scale_train, clf.predict(X_standard_scale_train)))
  print("Train - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_train, clf.predict(X_standard_scale_train)))
  print("Test - Accuracy :", metrics.accuracy_score(y_standard_scale_test, clf.predict(X_standard_scale_test)))
  print("Test - Confusion matrix :\n", metrics.confusion_matrix(y_standard_scale_test, clf.predict(X_standard_scale_test)))
  print('\n')