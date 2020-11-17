import numpy as np
from scipy.interpolate import interp1d


def one_dimensional_cpf(data):
    n_samples = data.shape[0]

    return interp1d(x=data,
                    y=(1 / n_samples) * np.arange(1, n_samples + 1, 1),
                    kind="linear", assume_sorted=False,
                    fill_value=(0, 1), bounds_error=False)


def multi_dimensional_cpf(data):
    multi_cpf = {}
    for i in range(data.shape[1]):
        multi_cpf[i] = one_dimensional_cpf(data[:, i])

    return multi_cpf


def compute_one_dimensional_probability(data, cpf, h=1.e-4):
    return (cpf(data - h * np.ones(shape=data.shape)) - cpf(data)) / h


def compute_probability(data, cpf, h):
    probability = np.ones(shape=data[0])
    iterator = 0
    for distributor in cpf:
        probability * compute_one_dimensional_probability(data[:, iterator],
                                                          cpf[distributor], h)
        iterator += 1

    return probability