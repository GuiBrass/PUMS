import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

from data_preprocessing import get_files, finger_paring
from model_builder import build_dual_finger_classifier, build_transformer_classifier
from results_analysis import plot_train_val_loss, plot_CM

from data_preprocessing import finger_paring
from model_builder import build_dual_finger_classifier, build_transformer_classifier
from results_analysis import plot_CM

#Get dataset
folder_path = "./nogit_datasets/dataset_object_recognition/dataset_07-07-25"
folder_path_2 = "./nogit_datasets/dataset_object_recognition/dataset_08-07-25"


csv_files_calib, csv_files_empty, csv_files_cube, csv_files_screw_driver, csv_files_cylinder, csv_files_tennis, csv_files_can, csv_files_pyramid, csv_files_cup, csv_files_puck = get_files(folder_path)
csv_files_calib_2, csv_files_empty_2, csv_files_cube_2, csv_files_screw_driver_2, csv_files_cylinder_2, csv_files_tennis_2, csv_files_can_2, csv_files_pyramid_2, csv_files_cup_2, csv_files_puck_2 = get_files(folder_path_2)

csv_files_calib = csv_files_calib + csv_files_calib_2
csv_files_empty = csv_files_empty + csv_files_empty_2
csv_files_cube = csv_files_cube + csv_files_cube_2
csv_files_screw_driver = csv_files_screw_driver + csv_files_screw_driver_2
csv_files_cylinder = csv_files_cylinder + csv_files_cylinder_2
csv_files_tennis = csv_files_tennis + csv_files_tennis_2
csv_files_can = csv_files_can + csv_files_can_2
csv_files_pyramid = csv_files_pyramid + csv_files_pyramid_2
csv_files_cup = csv_files_cup + csv_files_cup_2
csv_files_puck = csv_files_puck + csv_files_puck_2

#calibration
calib_data, _ = finger_paring(csv_files_calib, 0)
baseline = np.mean(calib_data, axis=0)
print(np.shape(baseline))

inputs_empty, targets_empty = finger_paring(csv_files_empty, 0)
print("empty :", np.shape(inputs_empty))
inputs_cube, targets_cube = finger_paring(csv_files_cube, 1)
print("cube :", np.shape(inputs_cube))
inputs_screw_driver, targets_screw_driver = finger_paring(csv_files_screw_driver, 2)
print("screw_driver :", np.shape(inputs_screw_driver))
inputs_tennis, targets_tennis = finger_paring(csv_files_tennis, 3)
print("tennis :", np.shape(inputs_tennis))
inputs_can, targets_can = finger_paring(csv_files_can, 4)
print("can :", np.shape(inputs_can))
inputs_cylinder, targets_cylinder = finger_paring(csv_files_cylinder, 5)
print("cylinder :", np.shape(inputs_cylinder))
inputs_pyramid, targets_pyramid = finger_paring(csv_files_pyramid, 6)
print("pyramid :", np.shape(inputs_pyramid))
inputs_cup, targets_cup = finger_paring(csv_files_cup, 7)
print("cup :", np.shape(inputs_cup))
inputs_puck, targets_puck = finger_paring(csv_files_puck, 8)
print("puck :", np.shape(inputs_puck))
inputs = np.concatenate([inputs_empty, inputs_cube, inputs_screw_driver, inputs_tennis, inputs_can, inputs_cylinder, inputs_pyramid, inputs_cup, inputs_puck], axis=0)
# inputs = inputs-baseline
col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means
targets = np.concatenate([targets_empty, targets_cube, targets_screw_driver, targets_tennis, targets_can, targets_cylinder, targets_pyramid, targets_cup, targets_puck], axis=0)

def flip_fingers(X):
    """
    Swap finger 1 and finger 2.
    Assumes shape (N, 16, 32, 1)
    """
    X_flipped = X.copy()

    # Split fingers
    finger1 = X[:, :8, :, :]
    finger2 = X[:, 8:16, :, :]

    # Swap
    X_flipped[:, :8, :, :] = finger2
    X_flipped[:, 8:16, :, :] = finger1

    return X_flipped

l2lambda = 1e-2 #https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2
input_shape = (16, 32, 1)
l2_reg = regularizers.l2(1e-2)

# architectures = ['cold', 'finetuning', 'transfer']
architectures = ['cold', 'transfer']

N_FOLDS = 10
N_SAMPLES = 15

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = []
ACCs = {arch: np.zeros((N_FOLDS, N_SAMPLES)) for arch in architectures}
LOSSEs = {arch: np.zeros((N_FOLDS, N_SAMPLES)) for arch in architectures}

for fold, (train_idx, test_idx) in enumerate(skf.split(inputs, targets.ravel())):
    X_train_full = inputs[train_idx]
    X_test = inputs[test_idx]

    y_train_full = targets[train_idx]
    y_test = targets[test_idx]

    scaler = StandardScaler()

    X_train_full_flat = X_train_full.reshape(X_train_full.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_full = scaler.fit_transform(X_train_full_flat).reshape(-1,16,32,1)
    X_test = scaler.transform(X_test_flat).reshape(-1,16,32,1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )
    X_train_flipped = flip_fingers(X_train)
    X_val_flipped   = flip_fingers(X_val)
    X_test_flipped  = flip_fingers(X_test)

    y_train_flipped = y_train.copy()
    y_val_flipped   = y_val.copy()
    y_test_flipped  = y_test.copy()

    num_classes = len(np.unique(y_train_flipped))
    y_train_flat = y_train_flipped.flatten()

    # Precompute indices per class
    class_indices = {
        c: np.where(y_train_flat == c)[0]
        for c in range(num_classes)
    }

    for arch in architectures:
        for samp in range(N_SAMPLES):
        
            results_path = f'./figures/figure_8_increase_shots_transfer_learning/flip_finger_OR/fold_{fold}/{arch}/{samp}/'
            model_path = f'./nogit_NN/object_classification/flip_finger/fold_{fold}/{arch}/{samp}/'
            
            if os.path.exists(f'{results_path}/metrics.txt'):
                with open(f'{results_path}/metrics.txt', "r") as f:
                    lines = f.readlines()

                metrics = {}
                for line in lines:
                    key, val = line.strip().split("=")
                    metrics[key.strip()] = float(val)

                ACCs[arch][fold, samp] = metrics["test_accuracy"]
                LOSSEs[arch][fold, samp] = metrics["test_loss"]
                print(f'Results of exp {fold}, arch {samp}, samp {samp} acquired in results file')
            else:
                surface_model = load_model("./nogit_NN/surface_estimation/preprocessing_comparison/SE_Transformer_CNN_hybrid_mean_PD_removed.keras", compile=False)

                if arch == "cold":
                    model = build_dual_finger_classifier(surface_model=surface_model, unlocked_weights=True, cold_start=True)
                if arch == "finetuning":
                    model = build_dual_finger_classifier(surface_model=surface_model, unlocked_weights=True)
                    model.load_weights(f'./nogit_NN/object_classification/architecture_comparison/reducing_dataset/fold_{fold}/OR_two_bodies_cold_start/0/OR_two_bodies_cold_start.weights.h5')
                if arch == "transfer":
                    model = build_dual_finger_classifier(surface_model=surface_model, unlocked_weights=True)
                    model.load_weights(f'./nogit_NN/object_classification/architecture_comparison/reducing_dataset/fold_{fold}/OR_two_bodies_cold_start/0/OR_two_bodies_cold_start.weights.h5')
                    for layer in model.layers[-21:]:
                        if hasattr(layer, 'kernel_initializer'):
                            layer.kernel.assign(
                                layer.kernel_initializer(
                                    shape=layer.kernel.shape
                                )
                            )
                        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                            layer.bias.assign(
                                layer.bias_initializer(
                                    shape=layer.bias.shape
                                )
                            )

                    for layer in model.layers[:-21]:
                        layer.trainable = False

                    # Keep last 2 trainable
                    for layer in model.layers[-21:]:
                        layer.trainable = True

                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-4),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                if samp > 0:
                    selected_indices = []

                    for c in range(num_classes):
                        selected_indices.extend(class_indices[c][:samp])

                    selected_indices = np.array(selected_indices)

                    X_train_samp = X_train_flipped[selected_indices]
                    y_train_samp = y_train_flipped[selected_indices]

                    print(y_train_samp)
                    
                    if os.path.exists(model_path + f'{arch}_{samp}.weights.h5'):
                        print("Model was trained previously")
                    else:
                        checkpoint_cb_cut_LEDs = ModelCheckpoint(
                            filepath=model_path + f'{arch}_{samp}.weights.h5',
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True
                        )

                        history = model.fit(
                            X_train_samp, y_train_samp,
                            validation_data=(X_val_flipped, y_val_flipped),
                            epochs=100,
                            batch_size=32,
                            callbacks=[checkpoint_cb_cut_LEDs]
                        )

                test_loss, test_accuracy = model.evaluate(X_test_flipped, y_test_flipped)
                ACCs[arch][fold, samp] = test_accuracy
                LOSSEs[arch][fold, samp] = test_loss
                print("Flipped fingers test loss: ", test_loss)
                print("Flipped fingers test accuracy: ", test_accuracy)
                if not os.path.exists(f'{results_path}/metrics.txt'):
                    os.makedirs(results_path, exist_ok=True)
                    with open(os.path.join(results_path, "metrics.txt"), "w") as f:
                        f.write(f"test_accuracy = {test_accuracy:.4f}\n")
                        f.write(f"test_loss = {test_loss:.4f}\n")

exp_values = np.arange(N_SAMPLES)

plt.figure(figsize=(10,6))

box_width = 0.35
offset = box_width/2 + 0.03
# box_width = 0.25
# offset = box_width/2 + 0.17

ax = plt.gca()

colors = {
    "cold": "blue",
    # "finetuning": "orange",
    "transfer": "green"
}

offsets = {
    "cold": -offset,
    # "finetuning": 0,
    "transfer": offset,
}

boxplots = {}

for i, arch in enumerate(architectures):
    
    positions = exp_values + offsets[arch]

    box_data = [ACCs[arch][:, s] * 100 for s in range(N_SAMPLES)] 

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color=colors[arch]),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )
    data_mean = np.mean(box_data, axis=1)
    ax.plot(positions, data_mean, '^-', color=colors[arch], label='mean')
    boxplots[arch] = bp

# legend_elements = [
#     Patch(facecolor="skyblue", edgecolor="black", label="Cold start"),
#     Patch(facecolor="orange", edgecolor="black", label="Fine Tuning"),
#     Patch(facecolor="purple", edgecolor="black", label="Transfer Learning")
# ]

# ax.legend(handles=legend_elements, loc="lower right")

# ax.set_title("Test Accuracy While Increasing the Flipped Samples (k=10)")
ax.set_xlabel("Number of Samples", fontsize=14)
ax.set_ylabel("Test Accuracy (%)", fontsize=14)

ax.set_xticks(exp_values)
ax.set_xticklabels(exp_values, fontsize=12)
plt.yticks(fontsize=12)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "./figures/figure_8_increase_shots_transfer_learning/flip_finger_OR/accuracy_boxplot_summary.png",
    dpi=300
)

plt.show()