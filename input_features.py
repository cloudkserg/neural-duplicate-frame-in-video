import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ChewingOutDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

class ChewingOutModel(nn.Module):
    def __init__(self, input_size):
        super(ChewingOutModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out

# Example usage
X = torch.randn(1000, 16)
y = torch.randn(1000, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input data using a standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors and create PyTorch datasets and data loaders
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = ChewingOutDataset(X_train, y_train)
test_dataset = ChewingOutDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Instantiate the model
input_size = 16
model = ChewingOutModel(input_size)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

# Train the model
for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        # Normalize the inputs
        inputs = (inputs - torch.mean(inputs)) / torch.std(inputs)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, labels in test_loader:
            # Normalize the inputs
            inputs = (inputs - torch.mean(inputs)) / torch.std(inputs)

            # Forward pass
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
