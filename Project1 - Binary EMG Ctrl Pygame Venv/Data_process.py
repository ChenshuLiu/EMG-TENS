'''
This code is used to create data structures suitable for LSTM models
Specifally, this is used when training the model (reference to Modeling.py file)
'''

import pandas as pd
import os
import torch

data_dir = '/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Data/24_5_7 data'

def input_creation(df, length, step):
    signal = torch.tensor(df.iloc[:, 1], dtype = torch.float32)
    label = torch.tensor(df.iloc[:, 2], dtype = torch.int32)
    input_signal = []
    input_label = []
    for start_idx in range(0, df.shape[0]-length, step):
        end_idx = start_idx + length
        input_signal.append(signal[start_idx:end_idx].unsqueeze(1)) #2 dimensional tensor by unsqueezing
        #input_label.append(label[start_idx:end_idx]) all time step labeled
        input_label.append(label[end_idx])
    input_signal = torch.stack(input_signal)
    input_label = torch.stack(input_label)
    return input_signal, input_label

def load_file(data_dir, length, step):
    # prep for input of lstm from files
    input_signal_all = torch.empty((0, length, 1))
    input_label_all = torch.empty((0))
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        file_df = pd.read_csv(file_path)
        input_signal, input_label = input_creation(file_df, length, step)
        input_signal_all = torch.cat((input_signal_all,input_signal), dim = 0)
        input_label_all = torch.cat((input_label_all, input_label), dim = 0)
    return input_signal_all, input_label_all

# length = 1000
# step = 1
# input, label =load_file(data_dir, length, step)
# print(input.shape)
# print(label.shape)
# print(input[2001, :, :])
# print(label[2001])
