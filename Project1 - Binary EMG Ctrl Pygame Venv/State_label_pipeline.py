'''
This file is intended for generating csv files from vernier recording software
Originally, the vernier recorded csv append different rounds of recording by column
This code would separate the individual recordings and label them
'''

import numpy as np
import pandas as pd

csv_path = "/Users/liuchenshu/Documents/Research/EMG/EMG Analysis/Data/24_5_7_2 data.csv"
def label(df, columns, action_stamp, interval):
    # action_stamp: the time step where an action was taken
    # interval: the duration of action (certain sampling rate for how long?)
    df = df[columns]
    labels = np.zeros(df.shape[0], dtype=int)
    for idx in action_stamp:
        labels[idx:idx+interval] = 1
    df['State'] = labels
    return df

def file_label(csv_path):
    whole_df = pd.read_csv(csv_path)
    for col_idx in range(0, whole_df.shape[1], 2):
        df = whole_df.iloc[:, [col_idx,col_idx+1]]
        df_labeled = label(df, df.columns, [i*200 for i in [10, 30, 50, 70, 90]], 10*200)
        recording_name = df.columns[0].split(':')[0]
        root_path = '/'.join(csv_path.split('/')[:-1])
        file_name = csv_path.split('/')[-1].split('.')[0]
        df_labeled.to_csv(f'{root_path}/24_5_7 data/{file_name}_{recording_name}_labeled.csv', index = False)
    print("done!")

file_label(csv_path = csv_path)