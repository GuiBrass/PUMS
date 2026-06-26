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
    
    target_num = int(parts[1])
    return target_num

def extract_date_time(filename):
    base = os.path.basename(filename)
    # Extract the date-time part
    date_str = "_".join(base.split('_')[4:6])  # e.g., '2025-08-14_13-58-02'
    return datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

def get_SC_replace_files(folder_path):
    folder_path_pred_cp = folder_path + "/SC_pred_cp"
    folder_path_tar_cp = folder_path + "/SC_tar_cp"
    ply_files_pred_cp = glob.glob(os.path.join(folder_path_pred_cp, '*.ply'))
    ply_files_tar_cp = glob.glob(os.path.join(folder_path_tar_cp, '*.ply'))

    ply_files_pred_cp = sorted(
    ply_files_pred_cp, 
    key=lambda f: extract_num(os.path.basename(f).replace(".ply", ""))
    )
    ply_files_tar_cp = sorted(
    ply_files_tar_cp, 
    key=lambda f: extract_num(os.path.basename(f).replace(".ply", ""))
    )
    return ply_files_pred_cp, ply_files_tar_cp


def get_files(folder_path):
    folder_path_calib = folder_path + "/calib"
    folder_path_inputs = folder_path + "/input_surfs"
    folder_path_test_inputs = folder_path + "/test_input_surfs"
    folder_path_targets = folder_path + "/target_surfs"
    folder_path_test_targets = folder_path + "/test_target_surfs"
    
    csv_files_calib = glob.glob(os.path.join(folder_path_calib, '*.csv'))
    csv_files_inputs = glob.glob(os.path.join(folder_path_inputs, '*.csv'))
    csv_files_test_inputs = glob.glob(os.path.join(folder_path_test_inputs, '*.csv'))
    ply_files_targets = glob.glob(os.path.join(folder_path_targets, '*.ply'))
    ply_files_test_targets = glob.glob(os.path.join(folder_path_test_targets, '*.ply'))

    csv_files_calib = sorted(
        csv_files_calib,
        key=lambda f: extract_date_time(os.path.basename(f).replace(".csv", ""))
    )

    csv_files_inputs = sorted(
        csv_files_inputs,
        key=lambda f: extract_date_time(os.path.basename(f).replace(".csv", ""))
    )

    csv_files_test_inputs = sorted(
        csv_files_test_inputs,
        key=lambda f: extract_date_time(os.path.basename(f).replace(".csv", ""))
    )

    ply_files_targets = sorted(
    ply_files_targets, 
    key=lambda f: extract_num(os.path.basename(f).replace(".ply", ""))
    )

    ply_files_test_targets = sorted(
    ply_files_test_targets, 
    key=lambda f: extract_num(os.path.basename(f).replace(".ply", ""))
    )
    
    return csv_files_calib, csv_files_inputs, ply_files_targets, csv_files_test_inputs, ply_files_test_targets

def extract_csv(csv_fs):
    data = []
    for file in csv_fs:
        data.append(pd.read_csv(file).to_numpy())
    return data

def pointcloud_to_heightmap(points, grid_size=(60, 15)):
    # Extract coordinates
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    
    # Normalize x, y to [0, 1] range
    xs_norm = (xs - xs.min()) / (xs.max() - xs.min())
    ys_norm = (ys - ys.min()) / (ys.max() - ys.min())

    # Map to grid indices
    xi = (xs_norm * (grid_size[0] - 1)).astype(int)
    yi = (ys_norm * (grid_size[1] - 1)).astype(int)

    # Initialize heightmap
    heightmap = np.full(grid_size, np.nan)  # NaN for empty cells

    # Fill heightmap with highest Z in each cell
    for x_idx, y_idx, z_val in zip(xi, yi, zs):
        if np.isnan(heightmap[x_idx, y_idx]) or z_val > heightmap[x_idx, y_idx]:
            if z_val < 0.1:
                heightmap[x_idx, y_idx] = 0
            else:
                heightmap[x_idx, y_idx] = z_val

    # Replace NaNs with 0 or interpolated values
    heightmap = np.nan_to_num(heightmap, nan=0.0)
    heightmap = np.fliplr(heightmap)
    
    return heightmap

def extract_ply_global_min_max(ply_fs):
    heightmaps = []
    all_z = []
    n_f = 0
    for file in ply_fs:
        pcd = o3d.io.read_point_cloud(file) 
        n_f+=1
        points = np.asarray(pcd.points)
        hm = pointcloud_to_heightmap(points)
        heightmaps.append(hm)
        all_z.append(hm)

    all_z = np.array(all_z)
    global_min = all_z.min()
    global_max = all_z.max()

    # Optionally normalize each heightmap to [0,1] using global min/max
    heightmaps_normalized = [(hm - global_min) / (global_max - global_min) for hm in heightmaps]

    return np.array(heightmaps_normalized), global_min, global_max

import numpy as np

def random_patch_drop(y, max_patch_size=(8, 4), drop_prob=0.3):
    H, W = y.shape
    y_dropped = y.copy()

    n_patches = np.random.randint(3, 5)

    for _ in range(n_patches):
        if np.random.rand() > drop_prob:
            continue

        # Random patch size
        patch_h = np.random.randint(4, max_patch_size[0] + 1)
        patch_w = np.random.randint(2, max_patch_size[1] + 1)

        # Random top-left corner (ensure fits in image)
        top = np.random.randint(0, H - patch_h + 1)
        left = np.random.randint(0, W - patch_w + 1)

        # Drop patch (set to zero or optionally to mean)
        y_dropped[top:top + patch_h, left:left + patch_w] = 0

    return y_dropped

def augment_dataset_with_patch_drop(x_train, y_train, max_patch_size=(8,4), drop_prob=0.3):
    N = len(x_train)
    x_aug = np.zeros((N, *x_train.shape[1:]), dtype=x_train.dtype)
    y_aug = np.zeros((N, *y_train.shape[1:]), dtype=y_train.dtype)

    for i in range(N):
        x_aug[i] = x_train[i]                # input stays the same
        y_aug[i] = random_patch_drop(y_train[i],
                                     max_patch_size=max_patch_size,
                                     drop_prob=drop_prob)

    # Concatenate original and augmented datasets
    x_doubled = np.concatenate([x_train, x_aug], axis=0)
    y_doubled = np.concatenate([y_train, y_aug], axis=0)

    return x_doubled, y_doubled

def apply_gaussian_blur_to_targets(targets, sigma=1.0):
    """
    Apply Gaussian blur to each target heightmap.
    - targets: np.array of shape (N, H, W)
    - sigma: standard deviation of Gaussian kernel
    Returns: blurred version of targets
    """
    blurred = np.zeros_like(targets)
    for i in range(targets.shape[0]):
        blurred[i] = gaussian_filter(targets[i], sigma=sigma)
    return blurred

def normalize_data(inputs, targets):
    scaler = MinMaxScaler()
    inputs_scaled = scaler.fit_transform(inputs.reshape(len(inputs), -1)).reshape(inputs.shape)
    glob_min, glob_max = targets.min(), targets.max()
    targets_scaled = (targets - glob_min) / (glob_max - glob_min)
    return inputs_scaled, targets_scaled, scaler, glob_min, glob_max

class PC0Remover:
    def __init__(self):
        self.mean = None
        self.pc0 = None
        self.s0 = None

    def fit(self, data):
        """Fit on training/validation data only."""
        N = data.shape[0]
        X = data.reshape(N, -1)

        # Store mean
        self.mean = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mean

        # PCA on centered data
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        
        # Save PC0 and singular value
        self.pc0 = Vh[0, :]       # shape (features,)
        self.s0 = S[0]

    def transform(self, data):
        """Apply PC0 removal using parameters from training set."""
        N = data.shape[0]
        X = data.reshape(N, -1)

        # center with training mean
        X_centered = X - self.mean

        # compute PC0 scores using stored PC0
        pc0_scores = X_centered @ self.pc0

        # reconstruct PC0 contribution
        pc0_contribution = np.outer(pc0_scores, self.pc0)

        # remove the effect
        X_clean = X_centered - pc0_contribution

        # add back the mean to stay in original space
        X_clean += self.mean

        return X_clean.reshape(data.shape)
    
class LastPCsRemover:
    def __init__(self, n_components_to_remove=1):
        self.n_components_to_remove = n_components_to_remove
        self.mean = None
        self.pcs = None

    def fit(self, data):
        """
        Fit PCA on training data and store the last N PCs.
        """
        N = data.shape[0]
        X = data.reshape(N, -1)

        self.mean = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mean

        _, _, Vh = np.linalg.svd(X_centered, full_matrices=False)

        # Store the last N PCs
        self.pcs = Vh[-self.n_components_to_remove:, :]

    def transform(self, data):
        N = data.shape[0]
        X = data.reshape(N, -1)

        X_centered = X-self.mean
        scores = X_centered @ self.pcs.T

        contribution = scores @ self.pcs

        X_clean = X_centered - contribution
        X_clean += self.mean

        return X_clean.reshape(data.shape)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
