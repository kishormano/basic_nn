import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def rl(Z):
    return np.maximum(0, Z)

def sftmx(Z):
    expZ = np.exp(Z - np.max(Z, axis=0))
    return expZ / np.sum(expZ, axis=0)

def fwd_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = rl(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sftmx(Z2)
    return Z1, A1, Z2, A2

def encode(Y):
    eY = np.zeros((Y.size, Y.max() + 1))
    eY[np.arange(Y.size), Y] = 1
    eY = eY.T
    return eY

def rl_(Z):
    return Z > 0

def bc_prop(Z1, A1, Z2, A2, W2, X, Y):
    nc = encode(Y)
    dZ2 = A2 - nc
    dW2 = 1 / Y.size * dZ2.dot(A1.T)
    db2 = 1 / Y.size * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * rl_(Z1)
    dW1 = 1 / Y.size * dZ1.dot(X.T)
    db1 = 1 / Y.size * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, a):
    W1 = W1 - a * dW1
    b1 = b1 - a * db1
    W2 = W2 - a * dW2
    b2 = b2 - a * db2
    return W1, b1, W2, b2

def pred(Q):
    return np.argmax(Q, 0)

def acc(P, Y):
    print(P, Y)
    return np.sum(P == Y) / Y.size

def grad_desc(X, Y, K, a):
    W1, b1, W2, b2 = init_params()
    for i in range(K):
        Z1, A1, Z2, A2 = fwd_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bc_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, a)
        if i % 10 == 0:
            print("Iteration:", i)
            print("Accuracy:", acc(pred(A2), Y))
    return W1, b1, W2, b2
