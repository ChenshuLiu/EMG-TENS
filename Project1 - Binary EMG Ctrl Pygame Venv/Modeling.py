import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import Data_process

# Check if MPS (Metal Performance Shaders) is available
is_mps_available = torch.backends.mps.is_available()
print(f"MPS available: {is_mps_available}")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = '/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Data/24_5_7 data'
model_save_dir = '/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Model/24_5_9 LSTM'

seq_length = 1000
input_size = 1 # dimension of input
num_classes = 2 # binary classification
batch_size = 128
step = 1

input, label = Data_process.load_file(data_dir, seq_length, step)
# print(input.shape)
# print(label.shape)
# print(label[1])
dataset = TensorDataset(input, label)
train_loader = DataLoader(dataset, batch_size = batch_size)

class BinaryGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BinaryGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

hidden_size = 128
num_layers = 2

LSTMmodel = BinaryGestureLSTM(input_size, hidden_size, num_layers, num_classes)
LSTMmodel.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(LSTMmodel.parameters())

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = LSTMmodel(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim = 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.numel() # total number of element in tensor
        if batch_idx % 200 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {total_loss / (batch_idx + 1):.4f}, " 
                f"Accuracy: {correct / total_samples:.4f}")
        
    print(f"Epoch [{epoch + 1}/{num_epochs}] finished, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total_samples:.4f}")
    torch.save(LSTMmodel, f'{model_save_dir}/lstm_{epoch+1}_model.pth')