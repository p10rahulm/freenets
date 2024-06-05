import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import seaborn as sns

def format_polynomial(coefficients, degrees):
    terms = sorted(zip(degrees, coefficients), reverse=True)
    polynomial_terms = []
    for degree, coeff in terms:
        if degree == 0:
            polynomial_terms.append(f"{coeff:.2f}")
        elif coeff < 0:
            polynomial_terms.append(f"- {-coeff:.2f}x^{degree}")
        else:
            polynomial_terms.append(f"+ {coeff:.2f}x^{degree}")
    polynomial_str = " ".join(polynomial_terms).replace("+ -", "- ")
    if polynomial_str.startswith("+ "):
        polynomial_str = polynomial_str[2:]  # Remove the leading "+ " if present
    return polynomial_str

def plot_results(model, x, y, coefficients, mse, mae, rmse, r2, mape, output_dir, filename, degrees, num_neurons):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device  # Get the device from model parameters
        test_x = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32).to(device)
        test_y = model(test_x).cpu()  # Ensure the tensor is moved to the CPU

    polynomial_str = format_polynomial(coefficients, degrees)
    title_str = f'Polynomial: y = {polynomial_str}\nNumber of Neurons: {num_neurons}'
    
    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Original data', alpha=0.6)
    plt.plot(test_x.cpu().numpy(), test_y.numpy(), color='red', label='Fitted line')  # Ensure test_x is moved to the CPU
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(title_str, fontsize=16)
    legend = plt.legend()

    # Calculate the position for metrics text based on the legend position
    bbox = legend.get_window_extent().transformed(plt.gca().transAxes.inverted())
    text_x = bbox.x0 + bbox.width / 2
    if bbox.y0 > 0.5:
        text_y = bbox.y0 - 0.05  # Place below the legend
        verticalalignment = 'top'
    else:
        text_y = bbox.y0 + bbox.height + 0.05  # Place above the legend
        verticalalignment = 'bottom'

    # Display metrics inside the graph
    metrics_text = (f'MSE: {mse:.4f}\n'
                    f'MAE: {mae:.4f}\n'
                    f'RMSE: {rmse:.4f}\n'
                    f'R²: {r2:.4f}\n'
                    f'MAPE: {mape:.2f}%')
    plt.gca().text(text_x, text_y, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment=verticalalignment, horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', alpha=0.1))
    
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(filepath, bbox_inches='tight')

    # Show the plot
    plt.show()
    plt.close()
