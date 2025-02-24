import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the RNN class
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        # Reshape input for RNN (batch_size, seq_length, input_size)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Forward pass through RNN
        out, _ = self.rnn(x)  # out: (batch_size, seq_length, hidden_size)
        
        # Only take the last time step's output
        out = out[:, -1, :]
        
        # Final classification layer
        out = self.fc(out)
        return out

# Create an instance of the RNN
model = SimpleRNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Prepare the dataset (now treated as sequences)
# Original data: [[1,0], [0,1], [1,1], [0,0]] becomes sequences of length 2
data = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

# Create dataset and dataloader
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy: {100 * correct / total}%')
