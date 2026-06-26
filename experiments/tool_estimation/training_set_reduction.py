import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers


from build_model import build_hybrid_transformer_for_tool

np.random.seed(100)
tf.random.set_seed(100)

dataset_path = "./nogit_datasets/dataset_tool_force_position/"
data_old_folder = dataset_path + "final_aruco_TPUspat_VGB/data/"
models_folder = "./nogit_NN/tool_force_pos_estimation/"

A1 = 0.00054844
A2 = 0.00056262
A3 = 0.00055019
B = -118.98137156

dataset_old = []
for file in os.listdir(data_old_folder):
    filepath = os.path.join(data_old_folder, file)

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # for key, value in data.items():
    #     print(f"\nKey: {key}")
    #     print("Type:", type(value))

    # print("shape pos:", np.shape(data["effector_pos"]))
    # print("val pos:", data["effector_pos"])
    positions = np.median(data["effector_pos"], axis=0)
    pos_x = positions[0]
    pos_y = positions[1]
    pos_z = positions[2]

    # print("shape ori:", np.shape(data["effector_orientation"]))
    # print("val ori:", data["effector_orientation"])
    orientations = np.median(data["effector_orientation"], axis=0)
    phi = orientations[0]
    theta = orientations[1]
    psi = orientations[2]

    l1 = np.mean(data["loadcell_B1"])
    l2 = np.mean(data["loadcell_B2"])
    l3 = np.mean(data["loadcell_B3"])

    total_mass = A1*l1+A2*l2+A3*l3+B
    # print("mass: ", total_mass)

    dataset_old.append({
        "pos_x": pos_x,
        "pos_y": pos_y,
        "pos_z": pos_z,
        "phi": phi,
        "theta": theta,
        "psi": psi,
        "pressure_command":data["pressure_command"],
        "total_mass": total_mass,
        "proprioception": data["proprioception"],
    })

df_old = pd.DataFrame(dataset_old)

df_old["pressure_command"] = df_old["pressure_command"].apply(
    lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
)

df_old["total_mass"] = df_old["total_mass"].apply(
    lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
)
min_mass = df_old["total_mass"].min()
print("min mass=", min_mass)
print("max mass=", df_old["total_mass"].max())
print("min pos=", df_old["pos_y"].min())
print("max pos=", df_old["pos_y"].max())

df_old["total_mass_corrected"] = df_old["total_mass"] - min_mass
df_old["force_N"] = df_old["total_mass_corrected"]*9.81/1000

zero_pressure_idx = df_old[df_old["pressure_command"] == 0].index
n_remove = 30
n_remove = min(n_remove, len(zero_pressure_idx))
np.random.seed(42)
drop_idx = np.random.choice(zero_pressure_idx, size=n_remove, replace=False)
df_old = df_old.drop(drop_idx).reset_index(drop=True)

print(f"Removed {n_remove} samples with 0 pressure.")
print(f"dataset size: ", len(df_old))

pd_values = np.stack(df_old["proprioception"].values)

col_means = pd_values.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = pd_values - col_means

inputs = inputs.reshape(-1, 16, 16, 1)

targets = df_old[["pos_y", "force_N"]].values

input_shape = (16, 16, 1)

N_FOLDS = 10
N_REDUCTION = 10

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

past_files = []
all_preds = []
all_targets = []
rmse = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    results = []
    # Make model dir
    models_folder = f'./nogit_NN/tool_force_pos_estimation/training_set_reduction/fold_{fold}/'
    os.makedirs(models_folder, exist_ok=True)

    # Make results dir
    results_folder = f'./figures/figure_10_tool_force_pos_estimation/training_set_reduction/fold_{fold}/'
    os.makedirs(results_folder, exist_ok=True)
    save_file = f"{results_folder}/results"

    if os.path.exists(save_file):
        print("path exists")
        df_check = pd.read_csv(save_file)
        if len(df_check) >= 9 and not df_check["trainset_reduced"].isnull().any():
            print(f"--> Fold {fold} is fully computed. Skipping training.")
            past_files.append(save_file)  # Ensure it is tracked for plotting
            continue
    
    for red in range(N_REDUCTION):

        X_train_full = inputs[train_idx]
        X_test = inputs[test_idx]

        y_train_full = targets[train_idx]
        y_test = targets[test_idx]

        scaler = StandardScaler()
        y_scaler = StandardScaler()

        N_train = X_train_full.shape[0]
        N_test = X_test.shape[0]

        X_train_flat = X_train_full.reshape(N_train, -1)
        X_test_flat = X_test.reshape(N_test, -1)

        # Scale
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat = scaler.transform(X_test_flat)

        # Reshape back
        X_train_full = X_train_flat.reshape(N_train, 16, 16, 1)
        X_test = X_test_flat.reshape(N_test, 16, 16, 1)

        y_train_full = y_scaler.fit_transform(y_train_full)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=1
        )

        remove_fraction = red * 0.10

        if remove_fraction > 0:
            # Calculate number to remove
            n_remove = int(remove_fraction * len(X_train))

            # Choose random indices to remove
            np.random.seed(1 + red + fold*100)  # reproducible but different each exp
            indices_to_remove = np.random.choice(len(X_train), n_remove, replace=False)

            # Create reduced train set
            mask = np.ones(len(X_train), dtype=bool)
            mask[indices_to_remove] = False

            X_train_reduced = X_train[mask]
            y_train_reduced = y_train[mask]

        else:
            # exp = 0 → nothing removed
            X_train_reduced = X_train
            y_train_reduced = y_train
        


        if os.path.exists(f'{models_folder}/{red}.weights.h5'):
                print("Model was trained previously")
        else:
            print("Training the model...")
            model = build_hybrid_transformer_for_tool(
                img_size=input_shape,
                output_size=2,
                patch_size=4,
                projection_dim=128,
                transformer_layers=4,
                CNN_layers_size=[128, 256]
            )
            checkpoint = ModelCheckpoint(filepath=f'{models_folder}/{red}.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
            history = model.fit(
                X_train_reduced, y_train_reduced,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint]
            )
        
        model = build_hybrid_transformer_for_tool(
                img_size=input_shape,
                output_size=2,
                patch_size=4,
                projection_dim=128,
                transformer_layers=4,
                CNN_layers_size=[128, 256]
            )
        model.load_weights(f'{models_folder}/{red}.weights.h5')
        # model.summary()

        pred_norm = model.predict(X_test)
        pred_val = model.predict(X_val)

        unscaled_y_pred = y_scaler.inverse_transform(pred_norm)
        # unscaled_y_test = y_scaler.inverse_transform(y_test)

        unscaled_y_pred_val = y_scaler.inverse_transform(pred_val)
        unscaled_y_val = y_scaler.inverse_transform(y_val)

        print("Pred mean:", np.mean(unscaled_y_pred, axis=0))
        print("True mean:", np.mean(y_test, axis=0))

        if red == 0:
            all_preds.append(unscaled_y_pred)
            all_targets.append(y_test)
            rmse.append(np.sqrt(np.mean((unscaled_y_pred - y_test) ** 2, axis=0)))

        mae_real = np.mean(np.abs(unscaled_y_pred - y_test), axis=0)
        print(mae_real)


        mae_val = np.mean(np.abs(unscaled_y_pred_val - unscaled_y_val), axis=0)
        print(mae_val)

        results.append([
            red,
            mae_real[0],
            mae_real[1]
        ])

        df_results = pd.DataFrame(
            results,
            columns=["trainset_reduced", "mae_pos_mm", "mae_force_N"]
        )
        
        df_results.to_csv(save_file, index=False)
        past_files.append(save_file)

# all_preds = np.vstack(all_preds)
# all_targets = np.vstack(all_targets)

# print("rmse = ", np.mean(rmse, axis=0))


all_mae_pos = []
all_mae_force = []

for file in past_files:
    df = pd.read_csv(file)
    all_mae_pos.append(df["mae_pos_mm"].values)
    all_mae_force.append(df["mae_force_N"].values)

all_mae_pos = np.array(all_mae_pos)
all_mae_force = np.array(all_mae_force)

N_REDUCTION = all_mae_pos.shape[1]
exp_values = np.arange(N_REDUCTION) * 10

plt.figure(figsize=(10,6))

ax1 = plt.gca()
ax2 = ax1.twinx()

# Prepare box data
box_data_pos = [all_mae_pos[:, r]*10 for r in range(N_REDUCTION)]
box_data_force = [all_mae_force[:, r]*1000 for r in range(N_REDUCTION)]

box_width = 3
offset = box_width/2 + 0.25

# Position error boxplots

ax1.boxplot(
        box_data_pos,
        positions=exp_values - offset,
        widths=3,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

data_mean_pos = np.mean(box_data_pos, axis=1)
ax1.plot(exp_values - offset, data_mean_pos, '^-', color='blue', label='mean')

ax2.boxplot(
        box_data_force,
        positions=exp_values + offset,
        widths=3,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color='green'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

data_mean_force = np.mean(box_data_force, axis=1)
ax2.plot(exp_values + offset, data_mean_force, '^-', color='green', label='mean')

ax1.set_xlabel("Training Data Removed (%)", fontsize=14)
ax1.set_ylabel("Position MAE (mm)", fontsize=14)
ax2.set_ylabel("Force MAE (mN)", fontsize=14)

ax1.set_xticks(exp_values)
ax1.set_xticklabels(exp_values)
# ax1.set_title("Tool Estimation: Test MAE vs Training Set Reduction")
ax1.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "./figures/figure_10_tool_force_pos_estimation/training_set_reduction/mae_pos_force_boxplot.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(12, 5))

# # --- POSITION ---
# plt.subplot(1, 2, 1)
# plt.scatter(all_targets[:, 0]*10, all_preds[:, 0]*10, s=7, alpha=0.5)

# # min_val = min(all_targets[:, 0])
# # max_val = max(all_targets[:, 0])
# # plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # perfect prediction line

# plt.xlabel("True Position (mm)", fontsize=14)
# plt.ylabel("Predicted Position (mm)", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.title("Position: Predictions vs Targets")
# plt.grid(True)

# # --- FORCE ---
# plt.subplot(1, 2, 2)
# plt.scatter(all_targets[:, 1]*1000, all_preds[:, 1]*1000, s=7, alpha=0.5)

# min_val = min(all_targets[:, 1])
# max_val = max(all_targets[:, 1])
# plt.plot([min_val, max_val], [min_val, max_val], 'r--')

# plt.xlabel("True Force (mN)", fontsize=14)
# plt.ylabel("Predicted Force (mN)", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.title("Force: Predictions vs Targets")
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(
#     "./figures/figure_10_tool_force_pos_estimation/training_set_reduction/pred_vs_target.png",
#     dpi=300
# )
# plt.show()
