import ast
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_preprocessing import extract_csv, extract_ply_global_min_max, get_files, PC0Remover
from model_builder import build_hybrid_transformer, rmse_loss
from results_analysis import plot_mae_mm_with_sensor_reduction_merged, calculate_error

#Get dataset
folder_path = "./nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25"


csv_files_calib, csv_files_inputs, ply_files_targets, csv_files_test_inputs, ply_files_test_targets = get_files(folder_path)
calib = extract_csv(csv_files_calib)
calib = np.array(calib)
inputs = np.array(extract_csv(csv_files_inputs))
targets, glob_min, glob_max = extract_ply_global_min_max(ply_files_targets)

col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means

past_files = []

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):

    save_dir = f"./results/surface_estimation/sensor_reduction/fold_{fold}/"
    save_file = f"{save_dir}/sensor_reduction_results.csv"

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_file):
        df_results = pd.read_csv(save_file)
        results = df_results.values.tolist()
    else:
        results = []
        df_results = pd.DataFrame(columns=[
            "missing_sensors",
            "leds_removed",
            "pds_removed",
            "mae_LED", "mae_mm_LED", "rmse_mm_LED",
            "mae_PD", "mae_mm_PD", "rmse_mm_PD",
            "mae_both", "mae_mm_both", "rmse_mm_both"
        ])

    for i in range(16):

        if not df_results[df_results["missing_sensors"] == i].empty:
            print(f"Skipping cut {i} (already computed)")
            continue

        print(f"\nFold {fold} | Cut {i}")

        rng = np.random.default_rng(seed=fold * 100 + i)

        X_train_full = inputs[train_idx]
        y_train_full = targets[train_idx]

        X_test = inputs[test_idx]
        y_test = targets[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=fold
        )

        X_train_cut_LEDs = X_train.copy()
        X_val_cut_LEDs = X_val.copy()
        X_test_cut_LEDs = X_test.copy()

        X_train_cut_PDs = X_train.copy()
        X_val_cut_PDs = X_val.copy()
        X_test_cut_PDs = X_test.copy()

        X_train_cut_both = X_train.copy()
        X_val_cut_both = X_val.copy()
        X_test_cut_both = X_test.copy()

        row_df = df_results.loc[df_results["missing_sensors"] == i]

        if not row_df.empty:

            row = row_df.iloc[0]

            leds_to_remove = np.array(
                ast.literal_eval(row["leds_removed"]),
                dtype=int
            )

            pds_to_remove = np.array(
                ast.literal_eval(row["pds_removed"]),
                dtype=int
            )

        else:

            if i > 0:
                leds_to_remove = rng.choice(16, size=i, replace=False)
                pds_to_remove = rng.choice(16, size=i, replace=False)
            else:
                leds_to_remove = np.array([], dtype=int)
                pds_to_remove = np.array([], dtype=int)


        # LEDs masking
        X_train_cut_LEDs[:, leds_to_remove, :] = 0
        X_val_cut_LEDs[:, leds_to_remove, :] = 0
        X_test_cut_LEDs[:, leds_to_remove, :] = 0

        # PDs masking
        X_train_cut_PDs[:, :, pds_to_remove] = 0
        X_val_cut_PDs[:, :, pds_to_remove] = 0
        X_test_cut_PDs[:, :, pds_to_remove] = 0

        # BOTH masking
        X_train_cut_both[:, leds_to_remove, :] = 0
        X_val_cut_both[:, leds_to_remove, :] = 0
        X_test_cut_both[:, leds_to_remove, :] = 0

        X_train_cut_both[:, :, pds_to_remove] = 0
        X_val_cut_both[:, :, pds_to_remove] = 0
        X_test_cut_both[:, :, pds_to_remove] = 0

        model_dir = f'./nogit_NN/surface_estimation/sensor_reduction/fold_{fold}/'
        os.makedirs(model_dir, exist_ok=True)

        led_path = f'{model_dir}/ViT_cut_{i}_LEDs.keras'
        pd_path = f'{model_dir}/ViT_cut_{i}_PDs.keras'
        both_path = f'{model_dir}/ViT_cut_{i}_BOTH.keras'

        if os.path.exists(led_path):

            print(f"Loading LED model: {led_path}")

            model_cut_LEDs = load_model(
                led_path,
                custom_objects={"rmse_loss": rmse_loss}
            )

        else:

            print(f"Training LED model: cut {i}")

            model_cut_LEDs = build_hybrid_transformer(
                img_size=(16,16,1)
            )

            model_cut_LEDs.compile(
                optimizer="adam",
                loss=rmse_loss,
                metrics=["mae"]
            )

            checkpoint_cb_cut_LEDs = ModelCheckpoint(
                filepath=led_path,
                monitor='val_loss',
                save_best_only=True
            )

            model_cut_LEDs.fit(
                X_train_cut_LEDs,
                y_train,
                validation_data=(X_val_cut_LEDs, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_LEDs]
            )

        if os.path.exists(pd_path):

            print(f"Loading PD model: {pd_path}")

            model_cut_PDs = load_model(
                pd_path,
                custom_objects={"rmse_loss": rmse_loss}
            )

        else:

            print(f"Training PD model: cut {i}")

            model_cut_PDs = build_hybrid_transformer(
                img_size=(16,16,1)
            )

            model_cut_PDs.compile(
                optimizer="adam",
                loss=rmse_loss,
                metrics=["mae"]
            )

            checkpoint_cb_cut_PDs = ModelCheckpoint(
                filepath=pd_path,
                monitor='val_loss',
                save_best_only=True
            )

            model_cut_PDs.fit(
                X_train_cut_PDs,
                y_train,
                validation_data=(X_val_cut_PDs, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_PDs]
            )

        if os.path.exists(both_path):

            print(f"Loading BOTH model: {both_path}")

            model_cut_both = load_model(
                both_path,
                custom_objects={"rmse_loss": rmse_loss}
            )

        else:

            print(f"Training BOTH model: cut {i}")

            model_cut_both = build_hybrid_transformer(
                img_size=(16,16,1)
            )

            model_cut_both.compile(
                optimizer="adam",
                loss=rmse_loss,
                metrics=["mae"]
            )

            checkpoint_cb_cut_both = ModelCheckpoint(
                filepath=both_path,
                monitor='val_loss',
                save_best_only=True
            )

            model_cut_both.fit(
                X_train_cut_both,
                y_train,
                validation_data=(X_val_cut_both, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_both]
            )

        mae_LEDs, mae_mm_LEDs, rmse_mm_LEDs = calculate_error(
            model_cut_LEDs,
            X_test_cut_LEDs,
            y_test
        )

        mae_PDs, mae_mm_PDs, rmse_mm_PDs = calculate_error(
            model_cut_PDs,
            X_test_cut_PDs,
            y_test
        )

        mae_both, mae_mm_both, rmse_mm_both = calculate_error(
            model_cut_both,
            X_test_cut_both,
            y_test
        )

        results.append([
            i,
            leds_to_remove.tolist(),
            pds_to_remove.tolist(),
            mae_LEDs, mae_mm_LEDs, rmse_mm_LEDs,
            mae_PDs, mae_mm_PDs, rmse_mm_PDs,
            mae_both, mae_mm_both, rmse_mm_both
        ])

        df_results = pd.DataFrame(
            results,
            columns=[
                "missing_sensors",
                "leds_removed",
                "pds_removed",
                "mae_LED", "mae_mm_LED", "rmse_mm_LED",
                "mae_PD", "mae_mm_PD", "rmse_mm_PD",
                "mae_both", "mae_mm_both", "rmse_mm_both"
            ]
        )

        df_results.to_csv(save_file, index=False)

    past_files.append(save_file)

plot_mae_mm_with_sensor_reduction_merged(
    past_files,
    saveplotpath="./results/surface_estimation/sensor_reduction/error_over_sensor_reduction"
)