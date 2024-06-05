from data_generator import generate_polynomial
from train import Trainer
from plotter_utilities import plot_results
from utilities import calculate_metrics
import os
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Allow PyTorch to use multiple GPUs

def main():
    input_size = 1
    n = 10           # Number of neurons
    num_samples = 1000
    learning_rate = 0.05
    num_epochs = 1000
    step_size = 500
    gamma = 0.95
    maximum_degree = 2*n-1  # Example maximum degree

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate data
    x, y, coefficients, degrees = generate_polynomial(n, maximum_degree, num_samples, noise_std=0.01)

    # Train the model
    trainer = Trainer(input_size, n, learning_rate, step_size, gamma, device)
    trainer.train(x, y, num_epochs)

    # Get the trained model
    model = trainer.get_model()

    # Calculate predictions and metrics
    with torch.no_grad():
        x_train = torch.tensor(x, dtype=torch.float32).to(device)
        outputs = model(x_train).cpu().numpy()  # Move outputs back to CPU for metric calculations
        mse, mae, rmse, r2, mape = calculate_metrics(y, outputs)

    # Plot the results
    output_dir = "outputs/plots/polynomials"
    filename = f"plot_{'_'.join([f'{coeff:.2f}x^{degree}'.replace('.', 'p') for coeff, degree in zip(coefficients, degrees) if degree != 0])}_loss_{mse:.4f}".replace(' ', '')
    
    plot_results(model, x, y, coefficients, mse, mae, rmse, r2, mape, output_dir, filename, degrees, n)

if __name__ == "__main__":
    main()
