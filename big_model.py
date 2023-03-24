import torch
import torch.nn as nn
import torch.nn.functional as F

class DuplicateFrameDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_layers, fc_layers):
        super(DuplicateFrameDetector, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers
        
        # Define the CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(1024 * 2 * 2 + hidden_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, x):
        # Pass the input through the CNN layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # Flatten the CNN output
        x = x.view(-1, 1024 * 2 * 2)
        
        # Pass the input through the LSTM layers
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        
        # Concatenate the LSTM output and the CNN output
        out = torch.cat((out[:, -1, :], x), dim=1)
        
        # Pass the concatenated output through the fully connected layers
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        out = F.relu(out)
        out = self.fc6(out)
        out = torch.sigmoid(out)

        return out
      
model = DuplicateFrameDetector(input_size=1024 * 2 * 2, hidden_size=128, num_layers=2, cnn_layers=6, fc_layers=6)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
  for i, (frames, label) in enumerate(train_loader):
    frames = frames.to(device)
    label = label.to(device)

outputs = model(frames)
loss = criterion(outputs, label)

# Backward and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print training progress
if (i + 1) % 10 == 0:
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
  correct = 0
  total = 0
  for frames, label in test_loader:
    frames = frames.to(device)
    label = label.to(device)
    outputs = model(frames)
    predicted = (outputs >= 0.5).float()
    total += label.size(0)
    correct += (predicted == label).sum().item()

print('Accuracy of the model on the test set: {} %'.format(100 * correct / total))
