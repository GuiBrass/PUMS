import ast
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
from results_analysis import calculate_error, plot_mae_mm_with_sensor_reduction_merged


np.random.seed(100)
tf.random.set_seed(100)

dataset_path = "./nogit_datasets/dataset_tool_force_position/"
calib_file = dataset_path + "loadcell_calib_coeff.pkl"
data_folder = dataset_path + "data/"
data_old_folder = dataset_path + "final_aruco_TPUspat_VGB/data/"
models_folder = "./nogit_NN/tool_force_pos_estimation/"

A1 = 0.00054844
A2 = 0.00056262
A3 = 0.00055019
B = -118.98137156

# ====== Charger coefficients de calibration ======
with open(calib_file, "rb") as f:
    calib = pickle.load(f)

dataset_old = []
for file in os.listdir(data_old_folder):
    filepath = os.path.join(data_old_folder, file)

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    # for key, value in data.items():
    #     print(f"\nKey: {key}")
    #     print("Type:", type(value))

    # print(np.shape(data["mean_tool_eff_pos"]))
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

pd_values = np.stack(df_old["proprioception"].values)

col_means = pd_values.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = pd_values - col_means 

inputs = inputs.reshape(-1, 16, 16, 1)

targets = df_old[["pos_y", "force_N"]].values

input_shape = (16, 16, 1)

N_FOLDS = 10

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

past_files = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    results = []
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

    os.makedirs(f"./results/tool_estimation/sensor_reduction/fold_{fold}/", exist_ok=True)

    save_file = f"./results/tool_estimation/sensor_reduction/fold_{fold}/sensor_reduction_results.csv"

    if os.path.exists(save_file):
        print("path exists")
        df_check = pd.read_csv(save_file)
        if len(df_check) >= 16 and not df_check["missing_sensors"].isnull().any():
            print(f"--> Fold {fold} is fully computed. Skipping training.")
            past_files.append(save_file)  # Ensure it is tracked for plotting
            continue

    for i in range(16):
        
        X_train_cut_LEDs = X_train.copy()
        X_val_cut_LEDs = X_val.copy()
        X_test_cut_LEDs = X_test.copy()
        X_train_cut_PDs = X_train.copy()
        X_val_cut_PDs = X_val.copy()
        X_test_cut_PDs = X_test.copy()
        X_train_cut_both = X_train.copy()
        X_val_cut_both = X_val.copy()
        X_test_cut_both = X_test.copy()
        rng = np.random.default_rng(seed=fold * 100 + i)
        if os.path.exists(save_file):
            df = pd.read_csv(save_file)
            row_df = df.loc[df["missing_sensors"] == i]

            if not row_df.empty:
                row = row_df.iloc[0]
                leds_to_remove = np.array(ast.literal_eval(row["leds_removed"]), dtype=int)
                pds_to_remove = np.array(ast.literal_eval(row["pds_removed"]), dtype=int)
            else:
                # File exists, but this cut hasn't been computed yet
                if i > 0:
                    leds_to_remove = rng.choice(16, size=i, replace=False)
                    pds_to_remove = rng.choice(16, size=i, replace=False)
                else:
                    leds_to_remove = np.array([], dtype=int)
                    pds_to_remove = np.array([], dtype=int)
        else:
            if i > 0:
                leds_to_remove = np.random.choice(16, size=i, replace=False)
                pds_to_remove = np.random.choice(16, size=i, replace=False)

            else:
                leds_to_remove = np.array([], dtype=int)
                pds_to_remove = np.array([], dtype=int)
            
        X_train_cut_LEDs[:, leds_to_remove, :, :] = 0
        X_val_cut_LEDs[:, leds_to_remove, :, :] = 0
        X_test_cut_LEDs[:, leds_to_remove, :, :] = 0

        
        X_train_cut_PDs[:, :, pds_to_remove, :] = 0
        X_val_cut_PDs[:, :, pds_to_remove, :] = 0
        X_test_cut_PDs[:, :, pds_to_remove, :] = 0

        # LEDs
        X_train_cut_both[:, leds_to_remove, :, :] = 0
        X_val_cut_both[:, leds_to_remove, :, :] = 0
        X_test_cut_both[:, leds_to_remove, :, :] = 0

        # PDs
        X_train_cut_both[:, :, pds_to_remove, :] = 0
        X_val_cut_both[:, :, pds_to_remove, :] = 0
        X_test_cut_both[:, :, pds_to_remove, :] = 0

        # print(X_train_cut_both[0])
        
        led_path = f'./nogit_NN/tool_force_pos_estimation/sensor_reduction/fold_{fold}/ViT_cut_{i}_LEDs.keras'
        pd_path = f'./nogit_NN/tool_force_pos_estimation/sensor_reduction/fold_{fold}/ViT_cut_{i}_PDs.keras'
        both_path = f'./nogit_NN/tool_force_pos_estimation/sensor_reduction/fold_{fold}/ViT_cut_{i}_BOTH.keras'

        if os.path.exists(led_path):
            print(f"Loading LED model: {led_path}")
            model_cut_LEDs = load_model(led_path)
        else:
            print(f"⚠️ Training LED model: cut {i}")
            model_cut_LEDs = build_hybrid_transformer_for_tool(
                img_size=(16,16,1),
                output_size=2,
                patch_size=4,
                projection_dim=128,
                transformer_layers=4,
                CNN_layers_size=[128, 256]
            )

            checkpoint_cb_cut_LEDs = ModelCheckpoint(filepath=led_path, monitor='val_loss', save_best_only=True)

            history_cut_LEDs = model_cut_LEDs.fit(
                X_train_cut_LEDs, y_train,
                validation_data=(X_val_cut_LEDs, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_LEDs]
            )
        if os.path.exists(pd_path):
            print(f"Loading PD model: {pd_path}")
            model_cut_PDs = load_model(pd_path)
        else:
            print(f"⚠️ Training PDs model: cut {i}")
            model_cut_PDs = build_hybrid_transformer_for_tool(
                img_size=(16,16,1),
                output_size=2,
                patch_size=4,
                projection_dim=128,
                transformer_layers=4,
                CNN_layers_size=[128, 256]
            )

            checkpoint_cb_cut_PDs = ModelCheckpoint(filepath=pd_path, monitor='val_loss', save_best_only=True)

            history_cut_PDs = model_cut_PDs.fit(
                X_train_cut_PDs, y_train,
                validation_data=(X_val_cut_PDs, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_PDs]
            )
        
        if os.path.exists(both_path):
            print(f"Loading BOTH model: {both_path}")
            model_cut_both = load_model(both_path)
        else:
            print(f"⚠️ Training BOTH model: cut {i}")
            model_cut_both = build_hybrid_transformer_for_tool(
                img_size=(16, 16, 1),
                output_size=2,
                patch_size=4,
                projection_dim=128,
                transformer_layers=4,
                CNN_layers_size=[128, 256]
            )

            checkpoint_cb_cut_both = ModelCheckpoint(
                filepath=both_path,
                monitor='val_loss',
                save_best_only=True
            )

            history_cut_both = model_cut_both.fit(
                X_train_cut_both, y_train,
                validation_data=(X_val_cut_both, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_both]
            )

        mae_LEDs = calculate_error(model_cut_LEDs, y_scaler, X_test_cut_LEDs, y_test)
        mae_PDs = calculate_error(model_cut_PDs, y_scaler, X_test_cut_PDs, y_test)
        mae_both = calculate_error(model_cut_both, y_scaler, X_test_cut_both, y_test)

        results.append([
            i,
            leds_to_remove.tolist(),
            pds_to_remove.tolist(),
            mae_LEDs[0], mae_LEDs[1],
            mae_PDs[0], mae_PDs[1],
            mae_both[0], mae_both[1]
        ])

        df_results = pd.DataFrame(
            results,
            columns=[
                "missing_sensors",
                "leds_removed",
                "pds_removed",
                "mae_pos_LED", "mae_force_LED",
                "mae_pos_PD", "mae_force_PD",
                "mae_pos_both", "mae_force_both"
            ]
        )
        df_results.to_csv(save_file, index=False)
        past_files.append(save_file)


dfs = [pd.read_csv(f) for f in past_files]

max_sensors_removed = 15
x_ticks = np.arange(0, max_sensors_removed + 1)  # [0, 1, 2, ..., 15]
max_both_pairs = max_sensors_removed // 2        # 15 // 2 = 7 pairs max
x_both_positions = np.arange(0, max_both_pairs + 1) * 2  # [0, 2, 4, ..., 15]

offset = 0.23
width = 0.2

all_led_pos = np.stack([df["mae_pos_LED"].values for df in dfs])[:, :max_sensors_removed + 1] * 10
all_pd_pos = np.stack([df["mae_pos_PD"].values for df in dfs])[:, :max_sensors_removed + 1] * 10

led_data_pos = [all_led_pos[:, i] for i in range(all_led_pos.shape[1])]
pd_data_pos = [all_pd_pos[:, i] for i in range(all_pd_pos.shape[1])]

all_both_pos = np.stack([df["mae_pos_both"].values for df in dfs])[:, :max_both_pairs + 1] * 10
both_data_pos = [all_both_pos[:, i] for i in range(all_both_pos.shape[1])]

all_led_force = np.stack([df["mae_force_LED"].values for df in dfs])[:, :max_sensors_removed + 1] * 1000
all_pd_force = np.stack([df["mae_force_PD"].values for df in dfs])[:, :max_sensors_removed + 1] * 1000

led_data_force = [all_led_force[:, i] for i in range(all_led_force.shape[1])]
pd_data_force = [all_pd_force[:, i] for i in range(all_pd_force.shape[1])]

all_both_force = np.stack([df["mae_force_both"].values for df in dfs])[:, :max_both_pairs + 1] * 1000
both_data_force = [all_both_force[:, i] for i in range(all_both_force.shape[1])]

fig, axes = plt.subplots(
    2, 1,
    figsize=(11.5, 6),
    sharex=True,
    gridspec_kw={'hspace': 0.08}
)

ax1, ax2 = axes

ax1.boxplot(
    led_data_pos,
    positions=np.arange(len(led_data_pos)) - offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='blue'),
    flierprops=dict(marker='o', markersize=6)
)

ax1.boxplot(
    pd_data_pos,
    positions=np.arange(len(pd_data_pos)),
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='green'),
    flierprops=dict(marker='o', markersize=6)
)

ax1.boxplot(
    both_data_pos,
    positions=x_both_positions + offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='purple'),
    flierprops=dict(marker='o', markersize=6)
)

# Mean curves
ax1.plot(
    np.arange(len(led_data_pos)) - offset,
    np.mean(all_led_pos, axis=0),
    '^-',
    color='blue',
    markersize=4
)

ax1.plot(
    np.arange(len(pd_data_pos)),
    np.mean(all_pd_pos, axis=0),
    '^-',
    color='green',
    markersize=4
)

ax1.plot(
    x_both_positions + offset,
    np.mean(all_both_pos, axis=0),
    '^-',
    color='purple',
    markersize=4
)

ax1.set_ylabel("Position MAE (mm)", fontsize=14)
ax1.grid(True, axis='both', alpha=0.3)
ax1.set_xlim(-0.6, max_sensors_removed + 0.6)

# =========================================================
# BOTTOM PLOT — FORCE
# =========================================================

ax2.boxplot(
    led_data_force,
    positions=np.arange(len(led_data_force)) - offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='blue'),
    flierprops=dict(marker='o', markersize=6)
)

ax2.boxplot(
    pd_data_force,
    positions=np.arange(len(pd_data_force)),
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='green'),
    flierprops=dict(marker='o', markersize=6)
)

ax2.boxplot(
    both_data_force,
    positions=x_both_positions + offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='purple'),
    flierprops=dict(marker='o', markersize=6)
)

# Mean curves
ax2.plot(
    np.arange(len(led_data_force)) - offset,
    np.mean(all_led_force, axis=0),
    '^-',
    color='blue',
    markersize=4
)

ax2.plot(
    np.arange(len(pd_data_force)),
    np.mean(all_pd_force, axis=0),
    '^-',
    color='green',
    markersize=4
)

ax2.plot(
    x_both_positions + offset,
    np.mean(all_both_force, axis=0),
    '^-',
    color='purple',
    markersize=4
)

ax2.set_ylabel("Force MAE (mN)", fontsize=14)
ax2.set_xlabel("Total Number of Removed Sensors", fontsize=14)
ax2.grid(True, axis='both', alpha=0.3)
ax2.set_xlim(-0.6, max_sensors_removed + 0.6)

# Shared x ticks
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([str(x) for x in x_ticks])

ax1.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

plt.tight_layout()

plt.savefig(
    "./results/tool_estimation/sensor_reduction/error_over_sensor_reduction.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()