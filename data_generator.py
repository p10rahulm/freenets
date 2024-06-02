import numpy as np

def generate_data_simple(n, num_samples=1000, noise_std=0.1):
    x = np.random.rand(num_samples, 1)
    y = x ** n + noise_std * np.random.randn(num_samples, 1)
    return x, y

def generate_data(n, num_samples=1000, noise_std=0.1):
    x = np.random.rand(num_samples, 1)
    coefficients = np.random.randn(n+1)
    y = np.zeros_like(x)
    for i in range(n+1):
        y += coefficients[i] * (x ** i)
    y += noise_std * np.random.randn(num_samples, 1)
    return x, y
