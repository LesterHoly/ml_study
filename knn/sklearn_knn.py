# -*- coding:utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# raw_data_x 是特征. raw_data_y是标签, 0 为良性, 1 为恶性
raw_data_x = [
    [3.393533211, 2.331273381],
    [3.110073483, 1.781539638],
    [1.343853454, 3.368312451],
    [3.582294121, 4.679917921],
    [2.280362211, 2.866990212],
    [7.423436752, 4.685324231],
    [5.745231231, 3.532131321],
    [9.172112222, 2.511113104],
    [7.927841231, 3.421455345],
    [7.939831414, 0.791631213]
]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
x = [8.90933607318, 3.365731514]

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

knn_classifier = KNeighborsClassifier(n_neighbors=6)

knn_classifier.fit(x_train, y_train)

x_predict = np.reshape(x, (1, -1))

y_predict = KNeighborsClassifier.predict(x_predict)
