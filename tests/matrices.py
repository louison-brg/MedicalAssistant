import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=2 , centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print("dim X:", X.shape)
print("dim y:", y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.show()

def init_params():
    w = np.random.rand(2, 1)
    b = np.random.rand(1)
    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w, b):
    z = np.dot(X, w) + b
    a = sigmoid(z)
    return a
def compute_loss(y, a):
    m = y.shape[0]
    loss = - (1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    return loss

def gradient(X, y, a):
    m = y.shape[0]
    dw = (1/m) * np.dot(X.T, (a - y))
    db = (1/m) * np.sum(a - y)
    return dw, db

def update_params(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b



