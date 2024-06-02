import torch
import torch.nn as nn
import torch.nn.functional as F


# Define Square ReLU activation function
class SquareReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# Define the custom neural network
class FreeNet(nn.Module):
    def __init__(self, input_size, n):
        super(FreeNet, self).__init__()
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
