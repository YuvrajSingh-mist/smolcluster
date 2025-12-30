# model.py
import numpy as np

# activation functions
def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(x.dtype)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class SimpleNN:
    def __init__(self, input_dim=784, hidden=128, out=10):
        self.W1 = np.random.randn(input_dim, hidden) * 0.01
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, out) * 0.01
        self.b2 = np.zeros((1, out))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[0]
        grads = {}
        Y_onehot = np.eye(self.b2.shape[1])[Y]
        dZ2 = self.A2 - Y_onehot
        grads["dW2"] = self.A1.T @ dZ2 / m
        grads["db2"] = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_grad(self.Z1)
        grads["dW1"] = X.T @ dZ1 / m
        grads["db1"] = np.sum(dZ1, axis=0, keepdims=True) / m
        return grads

    def apply_gradients(self, grads, lr=0.1):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def get_weights(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2 = weights