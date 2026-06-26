import ast
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from data_preprocessing import finger_paring
from build_model import build_CNN_model, build_CNN_encoder_decoder, build_hybrid_transformer_for_width, build_dual_finger_estimator

random.seed(100)
tf.random.set_seed(100)
np.random.seed(100)

#Get dataset
folder_path = "./nogit_datasets/dataset_width_prediction/dataset_15-07-25"

csv_files = glob.glob(os.path.join(folder_path+"/acquisition", '*.csv')) + glob.glob(os.path.join(folder_path+"/calib", '*.csv'))


#calibration
inputs, calib_pd_values, target_widths = finger_paring(csv_files)
inputs = np.array(inputs)
baseline = np.mean(calib_pd_values, axis=0)
print(np.shape(baseline))

inputs = inputs-baseline
col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means

inputs_reshape = inputs.reshape(np.shape(inputs)[0], -1)
targets_widths_reshape = np.array(target_widths).reshape(np.shape(target_widths)[0], -1)

input_shape = (16, 32, 1)
l2_reg = regularizers.l2(1e-3)

past_files = []
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=100)

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_reshape)):
    tf.random.set_seed(100+fold)
    X_train_full = inputs_reshape[train_idx]
    X_test = inputs_reshape[test_idx]

    y_train_full = targets_widths_reshape[train_idx]
    y_test = targets_widths_reshape[test_idx]

    scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    y_train_full = y_scaler.fit_transform(y_train_full)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=24
    )

    X_train = X_train.reshape(-1, 16, 32, 1)
    X_val = X_val.reshape(-1, 16, 32, 1)
    X_test = X_test.reshape(-1, 16, 32, 1)

    y_train = np.array(y_train).reshape(-1, 1)
    y_val = np.array(y_val).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    np.random.seed(100)
    save_file = f"./results/width_prediction/sensor_reduction/fold_{fold}/sensor_reduction_results.csv"
    if os.path.exists(save_file):
        df_check = pd.read_csv(save_file)
        if len(df_check) >= 16 and not df_check["missing_sensors"].isnull().any():
            print(f"--> Fold {fold} is fully computed. Skipping training.")
            past_files.append(save_file)  # Ensure it is tracked for plotting
            continue
    if os.path.exists(save_file):
        df_existing = pd.read_csv(save_file)
        results = df_existing.values.tolist()
    else:
        results = []
    for i in range(16):
        rng = np.random.default_rng(seed=fold * 100 + i)
        
        X_train_cut_LEDs = X_train.copy()
        X_val_cut_LEDs = X_val.copy()
        X_test_cut_LEDs = X_test.copy()
        X_train_cut_PDs = X_train.copy()
        X_val_cut_PDs = X_val.copy()
        X_test_cut_PDs = X_test.copy()
        X_train_cut_both = X_train.copy()
        X_val_cut_both   = X_val.copy()
        X_test_cut_both  = X_test.copy()

        
        results_folder = f'./figures/figure_7_sensor_reduction/width_prediction/fold_{fold}/'
        os.makedirs(results_folder, exist_ok=True)

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
                    pds_to_remove = np.concatenate([pds_to_remove, pds_to_remove + 16])
                else:
                    leds_to_remove = np.array([], dtype=int)
                    pds_to_remove = np.array([], dtype=int)
        
        else:
            if i > 0:
                leds_to_remove = np.random.choice(16, size=i, replace=False)
                pds_to_remove = np.random.choice(16, size=i, replace=False)
                pds_to_remove = np.concatenate([pds_to_remove, pds_to_remove + 16])

            else:
                leds_to_remove = np.array([], dtype=int)
                pds_to_remove = np.array([], dtype=int)

        # LEDs (rows)
        X_train_cut_LEDs[:, leds_to_remove, :, :] = 0
        X_val_cut_LEDs[:, leds_to_remove, :, :]   = 0
        X_test_cut_LEDs[:, leds_to_remove, :, :]  = 0

        # PDs (columns — already duplicated correctly)
        X_train_cut_PDs[:, :, pds_to_remove, :] = 0
        X_val_cut_PDs[:, :, pds_to_remove, :]   = 0
        X_test_cut_PDs[:, :, pds_to_remove, :]  = 0

        # LEDs
        X_train_cut_both[:, leds_to_remove, :, :] = 0
        X_val_cut_both[:, leds_to_remove, :, :]   = 0
        X_test_cut_both[:, leds_to_remove, :, :]  = 0

        # PDs
        X_train_cut_both[:, :, pds_to_remove, :] = 0
        X_val_cut_both[:, :, pds_to_remove, :]   = 0
        X_test_cut_both[:, :, pds_to_remove, :]  = 0
        
        led_path = f'./nogit_NN/width_prediction/sensor_reduction/fold_{fold}/ViT_cut_{i}_LEDs.weights.h5'
        pd_path = f'./nogit_NN/width_prediction/sensor_reduction/fold_{fold}/ViT_cut_{i}_PDs.weights.h5'
        both_path = f'./nogit_NN/width_prediction/sensor_reduction/fold_{fold}/ViT_cut_{i}_BOTH.weights.h5'
        np.random.seed(100 + fold)
        if i == 0:
            prev_path = f'./nogit_NN/width_prediction/architecture_comparison/reducing_dataset/fold_{fold}/WP_one_big_body_bigger_CNN/0/WP_one_big_body_bigger_CNN.weights.h5'
            if os.path.exists(prev_path):
                print(f"Loading LED and PD model: {prev_path}")
                model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                model.load_weights(prev_path)
                
            model_cut_LEDs = model
            model_cut_PDs = model
            model_cut_both = model
        
        else:
            if os.path.exists(led_path):
                print(f"Loading LED model: {led_path}")
                model_cut_LEDs = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                model_cut_LEDs.load_weights(led_path)
            else:
                print(f"⚠️ Training LED model: cut {i}")
                model_cut_LEDs = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                checkpoint_cb_cut_LEDs = ModelCheckpoint(
                    filepath=led_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
                history = model_cut_LEDs.fit(
                        X_train_cut_LEDs, y_train,
                        validation_data=(X_val_cut_LEDs, y_val),
                        epochs=150,
                        batch_size=32,
                        callbacks=[checkpoint_cb_cut_LEDs]
                    )
            
            if os.path.exists(pd_path):
                print(f"Loading PD model: {pd_path}")
                model_cut_PDs = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                model_cut_PDs.load_weights(pd_path)
            else:
                print(f"⚠️ Training PD model: cut {i}")
                model_cut_PDs = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                checkpoint_cb_cut_PDs = ModelCheckpoint(
                    filepath=pd_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
                history = model_cut_PDs.fit(
                        X_train_cut_PDs, y_train,
                        validation_data=(X_val_cut_PDs, y_val),
                        epochs=150,
                        batch_size=32,
                        callbacks=[checkpoint_cb_cut_PDs]
                    )
                
            if os.path.exists(both_path):
                print(f"Loading BOTH model: {both_path}")
                model_cut_both = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                model_cut_both.load_weights(both_path)
            else:
                print(f"⚠️ Training BOTH model: cut {i}")
                model_cut_both = build_hybrid_transformer_for_width(img_size=(16,32,1), CNN_layers_size=(128,256))
                checkpoint_cb_cut_both = ModelCheckpoint(
                    filepath=both_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
                history = model_cut_both.fit(
                        X_train_cut_both, y_train,
                        validation_data=(X_val_cut_both, y_val),
                        epochs=150,
                        batch_size=32,
                        callbacks=[checkpoint_cb_cut_both]
                    )
            
        #LED
        y_pred_LED = model_cut_LEDs.predict(X_test_cut_LEDs)
        unscaled_y_pred_LED = y_scaler.inverse_transform(y_pred_LED)
        unscaled_y_test_LED = y_test
        errors_LED = unscaled_y_pred_LED.flatten() - unscaled_y_test_LED.flatten()
        mae_in_mm_LED = np.mean(abs(unscaled_y_test_LED-unscaled_y_pred_LED))
        rmse_LED = np.sqrt(np.mean(errors_LED**2))

        #PD
        y_pred_PD = model_cut_PDs.predict(X_test_cut_PDs)
        unscaled_y_pred_PD = y_scaler.inverse_transform(y_pred_PD)
        unscaled_y_test_PD = y_test
        errors_PD = unscaled_y_pred_PD.flatten() - unscaled_y_test_PD.flatten()
        mae_in_mm_PD = np.mean(abs(unscaled_y_test_PD-unscaled_y_pred_PD))
        rmse_PD = np.sqrt(np.mean(errors_PD**2))

        #Both
        y_pred_both = model_cut_both.predict(X_test_cut_both)
        unscaled_y_pred_both = y_scaler.inverse_transform(y_pred_both)
        unscaled_y_test_both = y_test
        errors_both = unscaled_y_pred_both.flatten() - unscaled_y_test_both.flatten()
        mae_in_mm_both = np.mean(abs(unscaled_y_test_both-unscaled_y_pred_both))
        rmse_both = np.sqrt(np.mean(errors_both**2))

        results.append([
            i,
            leds_to_remove.tolist(),
            pds_to_remove.tolist(),
            mae_in_mm_LED, rmse_LED,
            mae_in_mm_PD, rmse_PD,
            mae_in_mm_both, rmse_both
        ])
        df_results = pd.DataFrame(
            results,
            columns=[
                "missing_sensors",
                "leds_removed",
                "pds_removed",
                "mae_in_mm_LED", "rmse_LED",
                "mae_in_mm_PD", "rmse_PD",
                "mae_in_mm_both", "rmse_both"
            ]
        )
        
        df_results.to_csv(save_file, index=False)
        past_files.append(save_file)


dfs = [pd.read_csv(f) for f in past_files]

max_sensors_removed = 15
x_labels = dfs[0]["missing_sensors"].values
x_pos = np.arange(len(x_labels)) + 1 # 1, 2, 3, ...


all_led = np.stack([df["mae_in_mm_LED"].values for df in dfs])
all_pd = np.stack([df["mae_in_mm_PD"].values for df in dfs])


led_data = [all_led[:, i] for i in range(all_led.shape[1])]
pd_data = [all_pd[:, i] for i in range(all_pd.shape[1])]

max_both_pairs = max_sensors_removed // 2  # 15 // 2 = 7 pairs (14 sensors)
    
all_both = np.stack([df["mae_in_mm_both"].values for df in dfs])[:, :max_both_pairs + 1]
both_data = [all_both[:, i] for i in range(all_both.shape[1])]
    
# Map the "Both" x-positions to the actual total sensors removed (0, 2, 4, ..., 14)
x_both_positions = np.arange(0, max_both_pairs + 1) * 2 + 1
print(x_both_positions)


offset = 0.23
width = 0.2


plt.figure(figsize=(11, 6))


plt.boxplot(
led_data,
positions=x_pos - offset,
widths=width,
showfliers=True,
medianprops=dict(color='red'),
boxprops=dict(color='blue'),
whiskerprops=dict(color='black'),
capprops=dict(color='black'),
flierprops=dict(marker='o', markersize=6)
)


plt.boxplot(
pd_data,
positions=x_pos,
widths=width,
showfliers=True,
medianprops=dict(color='red'),
boxprops=dict(color='green'),
whiskerprops=dict(color='black'),
capprops=dict(color='black'),
flierprops=dict(marker='o', markersize=6)
)

plt.boxplot(
both_data,
positions=x_both_positions + offset,
widths=width,
showfliers=True,
medianprops=dict(color='red'),
boxprops=dict(color='purple'),
whiskerprops=dict(color='black'),
capprops=dict(color='black'),
flierprops=dict(marker='o', markersize=6)
)


led_mean = np.mean(all_led, axis=0)
pd_mean = np.mean(all_pd, axis=0)
both_mean = np.mean(all_both, axis=0)


plt.plot(x_pos - offset, led_mean, '^-', color='blue', label='LED mean')
plt.plot(x_pos, pd_mean, '^-', color='green', label='PD mean')
# Both line only connects the points at 0, 2, 4, ..., 14
plt.plot(x_both_positions + offset, both_mean, '^-', color='purple', label='Both mean (half LED / half PD)')


# plt.xlabel("Number of removed sensors")
plt.ylabel("MAE (mm)", fontsize=14)
plt.xlabel("Total Number of Removed Sensors", fontsize=14)
plt.grid(True, axis='both', alpha=0.3)


plt.xticks(x_pos, x_labels, fontsize=12) # ← labels are decoupled
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("./results/width_prediction/sensor_reduction/sensor_reduction_impact.png")
plt.show()
