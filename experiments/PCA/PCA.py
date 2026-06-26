import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from utils import remove_pc0

# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=../../nogit_datasets/dataset_object_recognition/dataset_07-07-25/,../../nogit_datasets/dataset_object_recognition/dataset_08-07-25/ python PCA.py
# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=../../nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25/ python PCA.py
# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=../../nogit_datasets/dataset_width_prediction/dataset_15-07-25/ python PCA.py
# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=../../nogit_datasets/dataset_tool_force_position/final_aruco_TPUspat_VGB/data/ python PCA.py

# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=./nogit_datasets/dataset_object_recognition/dataset_07-07-25/,./nogit_datasets/dataset_object_recognition/dataset_08-07-25/ python PCA.py
# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=./nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25/ python PCA.py
# PYTHONPATH=..:${PYTHONPATH} FOLDER_PATHS=./nogit_datasets/dataset_width_prediction/dataset_15-07-25/ python PCA.py

folder_paths = os.getenv('FOLDER_PATHS').split(',')

# assert('..' in os.getenv('PYTHONPATH'))
# Compute project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print("Computed project_root =", project_root)
# Path to the 'experiments' folder where object_recognition lives
exp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if exp_path not in sys.path:
    sys.path.insert(0, exp_path)

# Add to sys.path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Adding to path:", exp_path)

fingers = []
borders = None
ds = None
if 'object_recognition' in os.getenv('FOLDER_PATHS'):
    from object_recognition import data_preprocessing as dp

    ds = 'object_recognition'

    class_order = ['empty', 'cube', 'screw_driver', 'tennis', 'can', 'cylinder', 'pyramid', 'cup', 'puck']
    classes_calib = ['calib','empty', 'cube', 'screw_driver', 'tennis', 'can', 'cylinder', 'pyramid', 'cup', 'puck']
    
    csv_files = {}
    for folder_path in folder_paths:
        print(folder_path)
        lsts = dp.get_files(folder_path)

        for klass, lst in zip(classes_calib, lsts):
            if klass in csv_files:
                csv_files[klass] += lst
            else:
                csv_files[klass] = lst

    #calibration
    calib_data, _ = dp.finger_paring(csv_files['calib'], 0)
    baseline = np.mean(calib_data, axis=0)
    print('baseline:', np.shape(baseline))

    #plt.imshow(baseline); plt.show()

    inputs = {}
    for klass, idx in zip(class_order, range(9)):
        inputs[klass], _ = dp.finger_paring(csv_files[klass], idx)
        print(f'{klass}:', type(inputs[klass]), np.shape(inputs[klass]))

    for i in [1, 2]:
        finger = np.stack(sum([[d[:, 16*(i-1):16*i] for d in inputs[k]] for k in class_order], start=[]), axis=0)
        print(f'finger {i}:', finger.shape)
        

        # UNCOMMENT TO REMOVE MEAN OF EACH PD
        col_means = finger.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
        finger = finger - col_means

        fingers.append(finger)
    
    class_counts = []
    for klass in class_order:
        class_counts.append(np.shape(inputs[klass])[0])

    # Compute cumulative borders
    borders = np.cumsum(class_counts)

elif 'surface_estimation' in os.getenv('FOLDER_PATHS'):
    from surface_estimation import data_preprocessing as dp

    ds = 'surface_estimation'

    csv_files = []
    for folder_path in folder_paths:
        csv_files_calib, csv_files_inputs, ply_files_targets, csv_files_test_inputs, ply_files_test_targets = dp.get_files(folder_path)
        csv_files += csv_files_inputs

    inputs = np.array(dp.extract_csv(csv_files))
    print("inputs shape", np.shape(inputs))
    
    finger = np.stack(inputs, axis=0)
    print(f'finger:', finger.shape)

    # UNCOMMENT TO REMOVE MEAN OF EACH PD
    col_means = finger.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
    finger = finger - col_means

    fingers.append(finger)

elif 'width_prediction' in os.getenv('FOLDER_PATHS'):
    from width_prediction import data_preprocessing as dp
    import glob

    ds = 'width_prediction'

    csv_files = []
    for folder_path in folder_paths:
        csv_files += (
            glob.glob(os.path.join(folder_path, "acquisition", "*.csv"))
            + glob.glob(os.path.join(folder_path, "calib", "*.csv"))
        )

    # Load dataset
    inputs, calib_pd_values, target_widths = dp.finger_paring(csv_files)
    inputs = np.array(inputs)
    print("inputs shape:", inputs.shape)

    for i in [1, 2]:
        finger = np.stack(
            [sample[:, 16*(i-1):16*i] for sample in inputs],
            axis=0
        )

        print(f"finger {i} shape:", finger.shape)

        # remove mean of each PD
        col_means = finger.mean(axis=1, keepdims=True)
        finger = finger - col_means

        fingers.append(finger)

elif 'tool_force_position' in os.getenv('FOLDER_PATHS'):
    import pickle
    import glob
    import pandas as pd

    ds = 'tool_force_position'

    dataset_path = "../../nogit_datasets/dataset_tool_force_position/final_aruco_TPUspat_VGB/data/"

    dataset = []

    for folder_path in folder_paths:

        for file in os.listdir(dataset_path):
            filepath = os.path.join(dataset_path, file)

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            pd_values = np.array(data["proprioception"])  # (T, D)

            # remove mean per sample
            pd_values = pd_values - np.mean(pd_values, axis=1, keepdims=True)

            dataset.append(pd_values)

    inputs = np.stack(dataset, axis=0)

    print("inputs shape:", inputs.shape)

    fingers.append(inputs)

    borders = None

else:
    raise Exception('Unknown dataset')

for i in range(1, 1+len(fingers)):
    finger = fingers[i-1]

    # UNCOMMENT TO REMOVE PC0
    # finger = remove_pc0(finger)

    X = np.reshape(finger, (finger.shape[0], np.prod(finger.shape[1:])))
    print("X shape:", X.shape)

    X -= np.mean(X, axis=0)

    var = np.var(X)

    print('variance:', var)

    U, S, Vh = np.linalg.svd(X)
    print('svd', U.shape, S.shape, Vh.shape)
    print('sum singular values', np.sum(S))

    # Variance explained
    explained_var = S**2
    explained_ratio = explained_var / np.sum(explained_var)
    cumulative_ratio = np.cumsum(explained_ratio)


    # Number of PCs needed for 99% variance
    n99 = np.argmax(cumulative_ratio >= 0.99) + 1

    plt.figure(1)

    unexplained_variance = 1 - np.concatenate([[0], cumulative_ratio])

    plt.plot(
        np.arange(len(cumulative_ratio) + 1),
        unexplained_variance,
        '.-'
    )

    if n99 <= 200:
        plt.axvline(
            n99,
            color='red',
            linestyle='--',
            linewidth=2
        )

    plt.axhline(
        0.01,
        color='gray',
        linestyle=':'
    )

    plt.xlabel("Principal component index", fontsize=14)
    plt.ylabel("Cumulative Unexplained Variance", fontsize=14)

    # 5. Set the log scale for the Y-axis
    plt.yscale('log')

    from matplotlib.ticker import ScalarFormatter
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().ticklabel_format(style='plain', axis='x')

    plt.xlim(0, 200)
    plt.ylim(0.000001, 1)

    plt.grid(True, which="both", alpha=0.3)
    # plt.legend()

    plt.savefig(
        f'../../figures/PCA/{ds}/singular_values/singular_values_finger{i}_PD_mean_removed.png'
    )
    
    fig = plt.figure(1+i, figsize=(14, 10))
    for k in range(0,120,10):
        plt.subplot(3, 4, int(k/10)+1)
        sgn = np.sign(np.trace(Vh[k, :].reshape(16, 16))) # arbitrary sign on all components
        plt.imshow(sgn * Vh[k, :].reshape(16, 16))
        plt.title(f'Component {k}', fontsize=14)

    fig.text(0.5, 0.05, 'Photodetector ID', ha='center', va='center', fontsize=16)

    fig.text(0.08, 0.5, 'LED ID', ha='center', va='center', rotation='vertical', fontsize=16)
    # plt.savefig(f'./figures/PCA/components_finger{i}_{ds}.png')
    plt.savefig(f'../../figures/PCA/{ds}/components/components_finger{i}_PD_mean_removed.png')    
    # plt.savefig(f'./figures/PCA/components_finger{i}_{ds}_PD_mean_and_PC0_removed.png')
    # plt.savefig(f'./figures/PCA/components_finger{i}_{ds}_PC0_removed.png')

    # For X.shape = (n, 16*16), get U.shape = (n, n), S.shape = (16*16,), Vh.shape = (16*16, 16*16)
    # Hence, X[i, j] = sum_k U[i, k] S[k] Vh[k, j] ---> S[k] U[i, k] is the weight of component Vk[k, :] in X[i, :]
    plt.figure(10+i)
    for k in range(5, -1, -1):
        if k == 0: c = 'k'
        else: c = ''
        W = S[k] * U[:, k]
        plt.plot(W / np.std(W), c + '.', label=f'Component {k}')
    if borders is not None:
        for b in borders[:-1]:   # skip final border (total samples)
                plt.axvline(x=b, linestyle='--', color='gray', alpha=0.7)
    plt.ylabel(f'weight')
    plt.xlabel('sample')
    plt.title(f'finger {i}')    
    plt.legend()
    # plt.savefig(f'./figures/PCA/weights_finger{i}_{ds}.png')
    plt.savefig(f'../../figures/PCA/{ds}/weigths_over_acquisition/weights_finger{i}_PD_mean_removed.png')
    # plt.savefig(f'./figures/PCA/weights_finger{i}_{ds}_PD_mean_and_PC0_removed.png')
    # plt.savefig(f'./figures/PCA/weights_finger{i}_{ds}_PC0_removed.png')

plt.show()
