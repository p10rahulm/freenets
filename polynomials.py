from data_generator import generate_data
from train import Trainer
from plotter_utilities import plot_results

def main():
    input_size = 1
    n = 4           # Number of neurons
    num_samples = 1000
    learning_rate = 0.001
    num_epochs = 500

    # Generate data
    x, y = generate_data(n, num_samples)

    # Train the model
    trainer = Trainer(input_size, n, learning_rate)
    trainer.train(x, y, num_epochs)

    # Get the trained model
    model = trainer.get_model()

    # Plot the results
    plot_results(model, x, y)

if __name__ == "__main__":
    main()
