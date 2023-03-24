import torch
import torch.nn as nn

class DuplicateFrameDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DuplicateFrameDetector, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x has shape (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Pass the input through the LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Pass the final hidden state through a fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# Define the model hyperparameters
input_size = 64 # size of each frame
hidden_size = 128
num_layers = 2

# Instantiate the model
model = DuplicateFrameDetector(input_size, hidden_size, num_layers)

# Define the input sequence
seq_length = 10 # length of the input sequence
batch_size = 4 # number of video frames in each batch
input_seq = torch.randn(batch_size, seq_length, input_size)

# Make a prediction
model.eval()
with torch.no_grad():
    output = model(input_seq)
    predicted_duplicate = torch.sigmoid(output).squeeze().tolist()

# Print the predicted duplicate values for each frame
print(predicted_duplicate)
