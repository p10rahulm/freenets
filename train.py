import torch
import torch.nn as nn
import torch.optim as optim
from freenet import FreeNet
from data_generator import generate_data


class Trainer:
    def __init__(self, input_size, n, learning_rate=0.001):
        self.model = FreeNet(input_size, n)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, x_train, y_train, num_epochs=500):
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(num_epochs):
            self.model.train()

            # Forward pass
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    def get_model(self):
        return self.model
