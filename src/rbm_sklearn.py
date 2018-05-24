
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import BernoulliRBM


# Load Data
def RBM():
    filename = "../data/smaller.dta"
    raw_data = open(filename,'rt')
    data = np.loadtxt(raw_data, delimiter = " ")
    X = data[:,:3]
    Y = data[:, 3]
    print(X)
    print(Y)
    print("training on RBM")
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    rbm.fit(X, Y)
    predictions = rbm.transform(X)
    params = rbm.get_params()
    print("predictions = ", predictions)
    print("rbm = ", rbm)
    print("params = ", params)


if __name__ == '__main__':
    RBM()

