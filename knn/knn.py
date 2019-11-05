# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

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

x_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)

plt.scatter(
    x_train[y_train == 0, 0],
    x_train[y_train == 0, 1],
    color='g',
    label='Tumor Size'
)
plt.scatter(
    x_train[y_train == 1, 0],
    x_train[y_train == 1, 1],
    color='r',
    label='Time'
)
plt.xlabel('Tumor Size')
plt.ylabel('Time')
plt.axis([0, 10, 0, 5])
plt.show()


x = [8.90933607318, 3.365731514]

distances = [sqrt(np.sum(x_train_ - x) ** 2) for x_train_ in x_train]

nearest = np.argsort(distances)

print('nearest: ', nearest)

k = 6

top_k_y = [y_train[i] for i in nearest[:k]]
print('top_k_y: ', top_k_y)

votes = Counter(top_k_y)
print('votes: ', votes)

predict_y = votes.most_common(1)[0][0]
print('predict_y: ', predict_y)
