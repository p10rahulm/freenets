from data_generator import generate_data
from train import Trainer
from plotter_utilities import plot_results
import os


os.environ["CUDA_VISIBLE_DEVICES"]=""
# device = torch.device("cpu")

def main():
    input_size = 1
    n = 40           # Number of neurons
    num_samples = 1000
    learning_rate = 0.01
    num_epochs = 2500

    # Generate data
    x, y = generate_data(n, num_samples, noise_std=0.01)

    # Train the model
    trainer = Trainer(input_size, n, learning_rate)
    trainer.train(x, y, num_epochs)

    # Get the trained model
    model = trainer.get_model()

    # Plot the results
    plot_results(model, x, y)

if __name__ == "__main__":
    main()
