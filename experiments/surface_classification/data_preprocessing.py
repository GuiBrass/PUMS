from datetime import datetime
import glob
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
import os
import open3d as o3d
import pandas as pd

def extract_num(filename):
    # Split the filename by underscore
    parts = filename.split('_')
    
    # Ensure there are enough parts
    # if len(parts) < 4:
    #     raise ValueError("Filename does not have enough parts")
    # Extract the relevant values
    target_num = int(parts[1])
    return target_num

def extract_date_time(filename):
    """
    Extracts a datetime object from the filename.
    Example filename: 'surf_2025-08-14_13-58-02.csv'
    """
    base = os.path.basename(filename)
    # Extract the date-time part
    date_str = "_".join(base.split('_')[4:6])  # e.g., '2025-08-14_13-58-02'
    return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")


def get_files(folder_path):
    folder_path_calib = folder_path + "/calib"
    folder_path_inputs = folder_path + "/input_surfs"
    
    csv_files_calib = glob.glob(os.path.join(folder_path_calib, '*.csv'))
    csv_files_inputs = glob.glob(os.path.join(folder_path_inputs, '*.csv'))

    csv_files_calib = sorted(
        csv_files_calib,
        key=lambda f: extract_date_time(os.path.basename(f).replace(".csv", ""))
    )

    csv_files_inputs = sorted(
        csv_files_inputs,
        key=lambda f: extract_date_time(os.path.basename(f).replace(".csv", ""))
    )
    
    return csv_files_calib, csv_files_inputs

def extract_csv(csv_fs):
    data = []
    for file in csv_fs:
        data.append(pd.read_csv(file).to_numpy())
    return data

def shapes_to_multilabel(shape_list, num_shapes=12):
    y = np.zeros(num_shapes, dtype=int)
    for s in shape_list:
        y[s - 1] = 1
    return y

def to_onehot(class_id, num_classes=12):
    """Convert integer class_id (1–12) → one hot vector length num_classes."""
    onehot = np.zeros(num_classes, dtype=np.float32)
    onehot[class_id - 1] = 1.0
    return onehot

