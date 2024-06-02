import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_results(model, x, y):
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)
        test_y = model(test_x)

    plt.scatter(x, y, color='blue', label='Original data')
    plt.plot(test_x.numpy(), test_y.numpy(), color='red', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
