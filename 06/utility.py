import numpy as np

class Utility:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # revised to handle batch input
    def softmax(self, a):
        c = np.max(a, axis=1, keepdims=True)  # max per sample
        exp_a = np.exp(a - c)  # prevent overflow
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)  # sum per sample
        y = exp_a / sum_exp_a
        return y

    def cross_entropy_error_batch(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        delta = 1e-7  # To avoid log(0)

        return -np.sum(t * np.log(y + delta)) / batch_size
    
    def numerical_gradient(self, f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)  # Initialize gradient array

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            original_value = x[idx]

            x[idx] = original_value + h
            fxh1 = f(x)  # f(x + h)

            x[idx] = original_value - h
            fxh2 = f(x)  # f(x - h)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = original_value  # Restore original value
            it.iternext()   

        return grad