from gdx import gdx # need to have the local gdx folder in directory
from pynput import keyboard
from Godirect_interface_func import *
import torch
import torch.nn as nn
import pygame
import sys
from threading import Thread, Event
from queue import Queue

# https://vernierst.github.io/godirect-examples/python/#confirm-installation-of-the-godirect-module 

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
period = period_calc(10)
gdx.open(connection='ble', device_to_open = 'GDX-EKG 0U4016M7')
gdx.select_sensors() # rectified emg
gdx.start(period) 
column_headers= gdx.enabled_sensor_info()   # returns a string with sensor description and units
print('\n')
print(column_headers)
########################

pygame.init()
pygame_icon = pygame.image.load('EMG Analysis/Binary EMG Ctrl Pygame Venv/pygame_icon.png')
pygame.display.set_icon(pygame_icon)
win_width, win_height = 1000, 500
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('EMG Signal Actuation (Binary)')
clock = pygame.time.Clock()

box_rect = pygame.Rect(100, 100, 50, 50)
box_color = (255, 0, 0)
velocity = 3

run = True
#############################
def main():
    global run
    seq_length = 1000
    signal_queue = Queue()
    emg_signal_bufferstorage = RollingBuffer(seq_length)
    signal_queue.put(emg_signal_bufferstorage)
    label_queue = Queue()
    stop_event = Event()
    emg_thread = Thread(target = emg_signal_classification, args = (label_queue, signal_queue))
    emg_thread.start()
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        if not label_queue.empty():
            label = label_queue.get()
            print(f"buffer in main: {label}")
            if label == 1:
                box_rect.x += velocity
            elif label == 0:
                box_rect.x -= velocity

        win.fill((0, 0, 0))
        pygame.draw.rect(win, box_color, box_rect)
        pygame.display.update()
        clock.tick(50)
    stop_event.set()
    emg_thread.join()
    gdx.stop()
    gdx.close()
    pygame.quit()
    sys.exit()

##### Data Collection & Signal Processing #####
def input_process(input_signal):
    input_signal = torch.tensor(input_signal).to(torch.float32)
    input_signal = torch.unsqueeze(input_signal, dim = 0)
    input_signal = torch.unsqueeze(input_signal, dim = 2)
    input_signal = input_signal.to(device) # change to mps
    return input_signal

def emg_signal_classification(label_queue, signal_queue):
    print("accessing emg_function!")
    global run
    while run:
        measurements = gdx.read()
        emg_signal_bufferstorage = signal_queue.get()
        emg_signal_bufferstorage.add(measurements[-1])
        signal_queue.put(emg_signal_bufferstorage)
        if len(emg_signal_bufferstorage.get()) == 1000:
            input_signal = input_process(emg_signal_bufferstorage.get())
            with torch.no_grad():
                output_prediction = binary_classification_model(input_signal)
                #print(output_prediction)
                label = torch.argmax(output_prediction, dim = 1).item()
                print(f"signal processing: {label}")
                label_queue.put(label)
        else: # when length of data collected is not sufficient to feed in mode
            #print("checking length, waiting...")
            #print(len(emg_signal_bufferstorage.get()))
            if len(emg_signal_bufferstorage.get()) % 100 == 0:
                print(f"Collecting {''.join(['■']*int(len(emg_signal_bufferstorage.get())/100))}")

if __name__ == "__main__":
    main()