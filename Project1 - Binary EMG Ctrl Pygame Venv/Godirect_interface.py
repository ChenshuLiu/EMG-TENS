from gdx import gdx # need to have the local gdx folder in directory
from pynput import keyboard
from Godirect_interface_func import *
import torch
import torch.nn as nn
import pygame
import threading

class BinaryGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BinaryGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

##### Model Loading #####
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
model_path = '/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Model/24_5_9 LSTM/lstm_50_model.pth'
binary_classification_model = torch.load(model_path)
binary_classification_model = binary_classification_model.eval()
binary_classification_model.to(device)
#########################

##### Sensor Setup #####
gdx = gdx.gdx()
period = period_calc(200)
gdx.open(connection='ble', device_to_open = 'GDX-EKG 0U4016M7')
gdx.select_sensors() # rectified emg
gdx.start(period) 
column_headers= gdx.enabled_sensor_info()   # returns a string with sensor description and units
print('\n')
print(column_headers)
########################

##### Interface control #####
pressed_key = None
def on_press(key):
    global pressed_key
    pressed_key = key
# Create a keyboard listener
listener = keyboard.Listener(on_press=on_press)
# Start the listener
listener.start()

pygame.init()
pygame_icon = pygame.image.load('EMG Analysis/Binary EMG Ctrl Pygame Venv/pygame_icon.png')
pygame.display.set_icon(pygame_icon)
win_width, win_height = 1000, 500
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('EMG Signal Actuation (Binary)')
box_x_coord = 50
box_y_coord = 225
box_width, box_height = 50, 50
velocity = 10
#############################

##### Data Collection & Signal Processing #####
seq_length = 1000
emg_signal_bufferstorage = RollingBuffer(seq_length)
run = True
while run:
    if pressed_key == keyboard.KeyCode.from_char('q'):
        break
    else:
        measurements = gdx.read()
        # print(measurements)
        emg_signal_bufferstorage.add(measurements[-1])
        if len(emg_signal_bufferstorage.get()) == seq_length:
            input_signal = torch.tensor(emg_signal_bufferstorage.get()).to(torch.float32)
            input_signal = torch.unsqueeze(input_signal, dim = 0)
            input_signal = torch.unsqueeze(input_signal, dim = 2)
            input_signal = input_signal.to(device)
            with torch.no_grad():
                output_prediction = binary_classification_model(input_signal)
                label = torch.argmax(output_prediction, dim = 1)
                print(label)
        else: # when length of data collected is not sufficient to feed in mode
            if len(emg_signal_bufferstorage.get()) % 100 == 0:
                print(f"Collecting {''.join(['■']*int(len(emg_signal_bufferstorage.get())/100))}")

##### Termination #####
listener.stop()
gdx.stop()
gdx.close()
#######################