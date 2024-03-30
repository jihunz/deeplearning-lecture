from sklearn.datasets import load_iris

iris = load_iris()

print(f"IRIS Shape: {iris.data.shape}")
print(f"IRIS Features: {iris.feature_names}")
print(f"IRIS Target: {iris.target}")

import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = pd.Series(iris.target)
print(iris_df.head())

import matplotlib.pyplot as plt

setosa = iris_df[iris_df["target"] == 0]
versicolor = iris_df[iris_df["target"] == 1]
virginica = iris_df[iris_df["target"] == 2]


X = iris_df.iloc[:, :4]
y = iris_df.iloc[:, -1]

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def iris_knn(X, y, k):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)


for k in range(3, 15, 2):
    scores = iris_knn(X, y, k)
    print(f'K 가 {k:d} 일 때 정확도: {scores:.3f}')


