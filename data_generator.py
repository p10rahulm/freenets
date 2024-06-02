import numpy as np

def generate_data(n, num_samples=1000, noise_std=0.1):
    x = np.random.rand(num_samples, 1)
    y = x ** n + noise_std * np.random.randn(num_samples, 1)
    return x, y
