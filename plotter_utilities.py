import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import seaborn as sns


def plot_results(model, x, y, coefficients, final_loss, output_dir, filename, degrees, num_neurons):
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)
        test_y = model(test_x)

    # Sort the polynomial terms by degree from highest to lowest
    terms = sorted(zip(degrees, coefficients), reverse=True)
    polynomial_terms = [f"{coeff:.2f}x^{degree}" if degree != 0 else f"{coeff:.2f}" for degree, coeff in terms]
    polynomial_str = "y = " + " + ".join(polynomial_terms)
    title_str = f'Polynomial: {polynomial_str}\nFinal MSE Loss: {final_loss:.4f}, Number of Neurons: {num_neurons}'

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Original data', alpha=0.6)
    plt.plot(test_x.numpy(), test_y.numpy(), color='red', label='Fitted line')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title_str, fontsize=16)
    plt.legend()

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(filepath, bbox_inches='tight')

    # Show the plot
    plt.show()
    plt.close()
