import torch
import torch.nn as nn
import torch.optim as optim
from freenet import FreeNet

class Trainer:
    def __init__(self, input_size, n, learning_rate=0.001, step_size=500, gamma=0.5, device='cpu'):
        self.device = device
        model = FreeNet(input_size, n).to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        self.model = model.to(device)
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train(self, x_train, y_train, num_epochs=500):
        x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        for epoch in range(num_epochs):
            self.model.train()
            
            # Forward pass
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Step the scheduler
            self.scheduler.step()
            
            if (epoch+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}')

    def get_model(self):
        return self.model
