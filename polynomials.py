from data_generator import generate_polynomial
from train import Trainer
from plotter_utilities import plot_results
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# device = torch.device("cpu")

def main():
    input_size = 1
    n = 4           # Number of neurons
    num_samples = 1000
    learning_rate = 0.01
    num_epochs = 10000
    step_size = 500
    gamma = 0.75
    maximum_degree = 2*n-1  # Example maximum degree

    # Generate data
    x, y, coefficients, degrees = generate_polynomial(n, maximum_degree, num_samples, noise_std=0.01)

    # Train the model
    trainer = Trainer(input_size, n, learning_rate, step_size, gamma)
    trainer.train(x, y, num_epochs)

    # Get the trained model
    model = trainer.get_model()

    # Calculate final loss
    with torch.no_grad():
        x_train = torch.tensor(x, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)
        outputs = model(x_train)
        criterion = torch.nn.MSELoss()
        final_loss = criterion(outputs, y_train).item()

    # Plot the results
    output_dir = "outputs/plots/polynomials"
    filename = f"plot_{'_'.join([f'{coeff:.2f}x^{degree}'.replace('.', 'p') for coeff, degree in zip(coefficients, degrees) if degree != 0])}_loss_{final_loss:.4f}".replace(' ', '')
    plot_results(model, x, y, coefficients, final_loss, output_dir, filename, degrees, n)

if __name__ == "__main__":
    main()
