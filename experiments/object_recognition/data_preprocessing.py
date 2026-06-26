import glob
import numpy as np
import os
import pandas as pd

def get_files(folder_path):
    folder_path_calib = folder_path + "/calib/t2"
    folder_path_empty = folder_path + "/empty/t2"
    folder_path_cube = folder_path + "/cube/t2"
    folder_path_screw_driver = folder_path + "/screw_driver/t2"
    folder_path_cylinder = folder_path + "/cylinder/t2"
    folder_path_tennis = folder_path + "/tennis/t2"
    folder_path_can = folder_path + "/can/t2"
    folder_path_pyramid = folder_path + "/pyramid/t2"
    folder_path_cup = folder_path + "/cup/t2"
    folder_path_puck = folder_path + "/puck/t2"
    
    csv_files_calib = glob.glob(os.path.join(folder_path_calib, '*.csv'))
    csv_files_empty = glob.glob(os.path.join(folder_path_empty, '*.csv'))
    csv_files_cube = glob.glob(os.path.join(folder_path_cube, '*.csv'))
    csv_files_screw_driver = glob.glob(os.path.join(folder_path_screw_driver, '*.csv'))
    csv_files_cylinder = glob.glob(os.path.join(folder_path_cylinder, '*.csv'))
    csv_files_tennis = glob.glob(os.path.join(folder_path_tennis, '*.csv'))
    csv_files_can = glob.glob(os.path.join(folder_path_can, '*.csv'))
    csv_files_pyramid = glob.glob(os.path.join(folder_path_pyramid, '*.csv'))
    csv_files_cup = glob.glob(os.path.join(folder_path_cup, '*.csv'))
    csv_files_puck = glob.glob(os.path.join(folder_path_puck, '*.csv'))
    return csv_files_calib, csv_files_empty, csv_files_cube, csv_files_screw_driver, csv_files_cylinder, csv_files_tennis, csv_files_can, csv_files_pyramid, csv_files_cup, csv_files_puck

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
    return date+"_"+time

def finger_paring(csv_fs, obj):
    data_dict = {}
    for file in csv_fs:
        time = extract_values(os.path.basename(file).replace(".csv", ""))
        d = pd.read_csv(file).to_numpy()
        if time in data_dict.keys():
            data_dict[time] = [np.concatenate([data_dict[time], d], axis=1), obj]
        else:
            data_dict[time] = d
    data = []
    objs = []
    for t in data_dict.keys():
        v = len(np.shape(data_dict[t][0]))
        if v>=2:
            data.append(data_dict[t][0])
            objs.append(data_dict[t][1])
        else:
            print("removed for missing data")
    return data, objs