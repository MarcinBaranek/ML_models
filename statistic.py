import numpy as np
from scipy.interpolate import interp1d
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functions import multi_dimensional_cpf, compute_probability

# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#
# def empirical_cpf(data: np.array):
#     n_features = data.shape[1]
#     n_cases = data.shape[0] + 1
#     _values = (1 / n_cases) * np.arange(1, n_cases, 1)
#     functions = []
#     for feature in range(n_features):
#         functions.append(interp1d(data[:, feature], _values))
#
#     return functions
#
#
# def empirical_pdf(arg, functions, h=1.e-5):
#     results = 0
#     for feature in range(len(functions)):
#         diff = (functions[feature](arg[feature]) - functions[feature](arg[feature] - h)) / h
#         results += diff
#
#     return results


class EmpiricalClassifier:
    labels = np.empty(shape=(1,))
    h = 1.e-4
    cpf = {}

    def __init__(self, h=1.e-4):
        self.h = h

    def fit(self, data, labels):
        self.labels = np.unique(labels)
        for label in self.labels:
            self.cpf[label] = multi_dimensional_cpf(data[np.where(labels == label), :])

    def predict(self, data):
        results = np.empty(shape=(1, 1))
        for label in self.cpf:
            probability = compute_probability(data, cpf=self.cpf[label], h=self.h)
            probability = np.reshape(probability, newshape=(-1,))
            if results.shape == (1, 1):
                results = probability
            else:
                results = np.column_stack((results, probability))

        return results
