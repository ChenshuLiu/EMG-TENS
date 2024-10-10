import torch
import torch.nn as nn

class BinaryGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BinaryGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Load the contents of the .pth file
model_contents = torch.load('/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Model/24_5_9 LSTM/lstm_50_model.pth')

# Check the type of the loaded object
if isinstance(model_contents, dict):
    print("The file contains a state dictionary.")
elif isinstance(model_contents, torch.nn.Module):
    print("The file contains the entire model.")
else:
    print("Unknown contents.")
