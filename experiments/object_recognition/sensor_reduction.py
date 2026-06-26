import ast
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

from data_preprocessing import get_files, finger_paring
from model_builder import build_dual_finger_classifier, build_transformer_classifier
from results_analysis import plot_train_val_loss, plot_CM

from data_preprocessing import finger_paring
from model_builder import build_dual_finger_classifier, build_transformer_classifier
from results_analysis import plot_CM, plot_sensor_reduction

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

l2lambda = 1e-2 #https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2
input_shape = (16, 32, 1)
l2_reg = regularizers.l2(1e-2)

# past_files = ["./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_0.csv", 
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_1.csv",
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_2.csv",
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_3.csv",
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_4.csv", 
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_5.csv",
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_6.csv", 
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_7.csv", 
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_8.csv",
#               "./figures/figure_7_sensor_reduction/object_recognition/sensor_reduction_results_9.csv"]

past_files = []
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(inputs, targets.ravel())):
    
    save_file = f"./results/object_recognition/sensor_reduction/fold_{fold}/sensor_reduction_results.csv"
    results_folder = f'./results/object_recognition/sensor_reduction/fold_{fold}/'
    os.makedirs(results_folder, exist_ok=True)

    if os.path.exists(save_file):
        df_check = pd.read_csv(save_file)
        if len(df_check) >= 16 and not df_check["missing_sensors"].isnull().any():
            print(f"--> Fold {fold} is fully computed. Skipping training.")
            past_files.append(save_file)  # Ensure it is tracked for plotting
            continue

    results = []
    
    X_train_full = inputs[train_idx]
    X_test = inputs[test_idx]

    y_train_full = targets[train_idx]
    y_test = targets[test_idx]

    scaler = StandardScaler()

    X_train_full_flat = X_train_full.reshape(X_train_full.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_train_full = scaler.fit_transform(X_train_full_flat)
    X_test = scaler.transform(X_test_flat)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )

    X_train = X_train.reshape(-1,16,32,1)
    X_val = X_val.reshape(-1,16,32,1)
    X_test = X_test.reshape(-1,16,32,1)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    for i in range(0, 16):
        rng = np.random.default_rng(seed=fold * 100 + i)

        if os.path.exists(save_file):
            df = pd.read_csv(save_file)
            row_df = df.loc[df["missing_sensors"] == i]
        else:
            df = pd.DataFrame()
            row_df = pd.DataFrame()

        # Check if individual step was already recorded in this partial run
        if not row_df.empty:
            print(f"Skipping fold {fold} cut {i} (already in CSV)")
            row = row_df.iloc[0]
            leds_to_remove = np.array(ast.literal_eval(row["leds_removed"]), dtype=int)
            pds_to_remove = np.array(ast.literal_eval(row["pds_removed"]), dtype=int)
            
            # Reconstruct the results item to preserve it when writing back
            results.append(row.tolist())
            continue

        print(f"\nFold {fold} | Cut {i}")

        if i > 0:
            leds_to_remove = rng.choice(16, size=i, replace=False)
            pds_to_remove = rng.choice(16, size=i, replace=False)
            pds_to_remove = np.concatenate([pds_to_remove, pds_to_remove + 16])
        else:
            leds_to_remove = np.array([], dtype=int)
            pds_to_remove = np.array([], dtype=int)

        X_train_cut_LEDs = X_train.copy()
        X_val_cut_LEDs = X_val.copy()
        X_test_cut_LEDs = X_test.copy()
        X_train_cut_PDs = X_train.copy()
        X_val_cut_PDs = X_val.copy()
        X_test_cut_PDs = X_test.copy()
        X_train_cut_both = X_train.copy()
        X_val_cut_both   = X_val.copy()
        X_test_cut_both  = X_test.copy()

        # LEDs (rows)
        X_train_cut_LEDs[:, leds_to_remove, :, :] = 0
        X_val_cut_LEDs[:, leds_to_remove, :, :]   = 0
        X_test_cut_LEDs[:, leds_to_remove, :, :]  = 0

        # PDs (columns)
        X_train_cut_PDs[:, :, pds_to_remove, :] = 0
        X_val_cut_PDs[:, :, pds_to_remove, :]   = 0
        X_test_cut_PDs[:, :, pds_to_remove, :]  = 0

        # Both
        X_train_cut_both[:, leds_to_remove, :, :] = 0
        X_val_cut_both[:, leds_to_remove, :, :]   = 0
        X_test_cut_both[:, leds_to_remove, :, :]  = 0
        X_train_cut_both[:, :, pds_to_remove, :] = 0
        X_val_cut_both[:, :, pds_to_remove, :]   = 0
        X_test_cut_both[:, :, pds_to_remove, :]  = 0
        
        led_path = f'./nogit_NN/object_classification/sensor_reduction/fold_{fold}/ViT_cut_{i}_LEDs.weights.h5'
        pd_path = f'./nogit_NN/object_classification/sensor_reduction/fold_{fold}/ViT_cut_{i}_PDs.weights.h5'
        both_path = f'./nogit_NN/object_classification/sensor_reduction/fold_{fold}/ViT_cut_{i}_BOTH.weights.h5'

        # --- LED Model ---
        if os.path.exists(led_path):
            print(f"Loading LED model: {led_path}")
            model_cut_LEDs = build_transformer_classifier(img_size=(16,32,1))
            model_cut_LEDs.load_weights(led_path)
        else:
            print(f"⚠️ Training LED model: cut {i}")
            model_cut_LEDs = build_transformer_classifier(img_size=(16,32,1))
            checkpoint_cb_cut_LEDs = ModelCheckpoint(
                filepath=led_path, monitor='val_loss', save_best_only=True, save_weights_only=True
            )
            model_cut_LEDs.fit(
                X_train_cut_LEDs, y_train, validation_data=(X_val_cut_LEDs, y_val),
                epochs=200, batch_size=32, callbacks=[checkpoint_cb_cut_LEDs]
            )
        
        # --- PD Model ---
        if os.path.exists(pd_path):
            print(f"Loading PD model: {pd_path}")
            model_cut_PDs = build_transformer_classifier(img_size=(16,32,1))
            model_cut_PDs.load_weights(pd_path)
        else:
            print(f"⚠️ Training PD model: cut {i}")
            model_cut_PDs = build_transformer_classifier(img_size=(16,32,1))
            checkpoint_cb_cut_PDs = ModelCheckpoint(
                filepath=pd_path, monitor='val_loss', save_best_only=True, save_weights_only=True
            )
            model_cut_PDs.fit(
                X_train_cut_PDs, y_train, validation_data=(X_val_cut_PDs, y_val),
                epochs=200, batch_size=32, callbacks=[checkpoint_cb_cut_PDs]
            )
            
        # --- Both Model ---
        if os.path.exists(both_path):
            print(f"Loading BOTH model: {both_path}")
            model_cut_both = build_transformer_classifier(img_size=(16,32,1))
            model_cut_both.load_weights(both_path)
        else:
            print(f"⚠️ Training BOTH model: cut {i}")
            model_cut_both = build_transformer_classifier(img_size=(16,32,1))
            checkpoint_cb_cut_both = ModelCheckpoint(
                filepath=both_path, monitor='val_loss', save_best_only=True, save_weights_only=True
            )
            model_cut_both.fit(
                X_train_cut_both, y_train, validation_data=(X_val_cut_both, y_val),
                epochs=200, batch_size=32, callbacks=[checkpoint_cb_cut_both]
            )
                
        test_loss_LEDs, test_accuracy_LEDs = model_cut_LEDs.evaluate(X_test_cut_LEDs, y_test)
        test_loss_PDs, test_accuracy_PDs = model_cut_PDs.evaluate(X_test_cut_PDs, y_test)
        test_loss_both, test_accuracy_both = model_cut_both.evaluate(X_test_cut_both, y_test)
        
        results.append([
            i,
            leds_to_remove.tolist(),
            pds_to_remove.tolist(),
            test_loss_LEDs, test_accuracy_LEDs,
            test_loss_PDs, test_accuracy_PDs,
            test_loss_both, test_accuracy_both
        ])
        
        df_results = pd.DataFrame(
            results,
            columns=[
                "missing_sensors", "leds_removed", "pds_removed",
                "test_loss_LEDs", "test_accuracy_LEDs",
                "test_loss_PDs", "test_accuracy_PDs",
                "test_loss_both", "test_accuracy_both"
            ]
        )
        df_results = df_results.sort_values(by="missing_sensors")
        df_results.to_csv(save_file, index=False)

    past_files.append(save_file)

plot_sensor_reduction(past_files, saveplotpath=f"./results/object_recognition/sensor_reduction/error_over_sensor_reduction")