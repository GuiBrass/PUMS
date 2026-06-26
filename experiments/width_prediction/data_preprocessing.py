import os
import numpy as np
import pandas as pd

def extract_values(filename):
    # Split the filename by underscore
    parts = filename.split('_')
    
    # Ensure there are enough parts
    if len(parts) < 4:
        raise ValueError("Filename does not have enough parts")
    # Extract the relevant values
    # finger = parts[2]
    date = parts[4]
    time = parts[5]
    width = parts[6]
    return date+"_"+time, width

def finger_paring(csv_fs):
    data_dict = {}
    for file in csv_fs:
        time, width = extract_values(os.path.basename(file).replace(".csv", ""))
        d = pd.read_csv(file).to_numpy()
        if time in data_dict.keys():
            data_dict[time] = [np.concatenate([data_dict[time], d], axis=1), width]
        else:
            data_dict[time] = d
    data = []
    calib_data = []
    widths = []
    for t in data_dict.keys():
        v = len(np.shape(data_dict[t][0]))
        if v>=2:
            if float(data_dict[t][1]) == 0.0:
                calib_data.append(data_dict[t][0])
            else:
                data.append(data_dict[t][0])
                widths.append(float(data_dict[t][1]))
        else:
            print("removed for missing data")
    return data, calib_data, widths