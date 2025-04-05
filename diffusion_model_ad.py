import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def diffusion_process(x, noise_level):
    noise = torch.randn_like(x) * noise_level
    return x + noise

def train_diffusion_model(model, data, num_epochs, learning_rate, noise_level):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in data:
            noisy_batch = diffusion_process(batch, noise_level)
            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")