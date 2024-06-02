import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define Square ReLU activation function
class SquareReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# Define the custom neural network
class FreeNet(nn.Module):
    def __init__(self, input_size, n):
        super(CustomNN, self).__init__()
        self.n = n
        self.neurons = nn.ModuleList()

        # Create neurons with appropriate connections
        for i in range(n):
            if i == 0:
                self.neurons.append(nn.Linear(input_size, 1))
            else:
                self.neurons.append(nn.Linear(input_size + i, 1))

        # Output layer
        self.output = nn.Linear(n, 1)
        self.activation = SquareReLU()

    def forward(self, x):
        outputs = []
        for i in range(self.n):
            if i == 0:
                out = self.neurons[i](x)
            else:
                previous_outputs = torch.cat(outputs, dim=1)
                combined_input = torch.cat((x, previous_outputs), dim=1)
                out = self.neurons[i](combined_input)
            out = self.activation(out)
            outputs.append(out)

        final_input = torch.cat(outputs, dim=1)
        output = self.output(final_input)
        return output


# Generate sample data
def generate_data(n, num_samples=1000, noise_std=0.1):
    x = np.random.rand(num_samples, 1)
    y = x ** n + noise_std * np.random.randn(num_samples, 1)
    return x, y


# Training parameters
input_size = 1
n = 4  # Number of neurons
num_samples = 1000
learning_rate = 0.001
num_epochs = 500

# Generate data
x, y = generate_data(n, num_samples)
x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# Create model
model = CustomNN(input_size, n)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_x = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)
    test_y = model(test_x)

# Plot the results
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='Original data')
plt.plot(test_x.numpy(), test_y.numpy(), color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
