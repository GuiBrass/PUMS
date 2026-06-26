import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_preprocessing import extract_csv, extract_ply_global_min_max, get_files, LastPCsRemover
from model_builder import build_hybrid_transformer, rmse_loss
from results_analysis import calculate_error, plot_train_val_loss, plot_training_set_reduction, plot_surfaces, plot_pca_component_removal

#Get dataset
folder_path = "./nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25"


csv_files_calib, csv_files_inputs, ply_files_targets, csv_files_test_inputs, ply_files_test_targets = get_files(folder_path)
calib = extract_csv(csv_files_calib)
calib = np.array(calib)
inputs = np.array(extract_csv(csv_files_inputs))
test_inputs = np.array(extract_csv(csv_files_test_inputs))
targets, glob_min, glob_max = extract_ply_global_min_max(ply_files_targets)
test_targets, test_glob_min, test_glob_max = extract_ply_global_min_max(ply_files_test_targets)

col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means
col_means = test_inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
test_inputs = test_inputs - col_means

N_FOLDS = 10
pcs_to_test = [
    0, 64, 128, 192, 224, 232, 240, 244, 248, 250, 252, 254
]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

past_files = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):

    save_dir = f"./figures/PCA/surface_estimation/components_removal/fold_{fold}/"
    os.makedirs(save_dir,exist_ok=True)

    save_file = save_dir + "pca_removal_results.csv"
    
    if os.path.exists(save_file):
        df_results = pd.read_csv(save_file)
        results = df_results.values.tolist()

    else:
        results = []
        df_results = pd.DataFrame(
            columns=[
                "removed_pcs",
                "mae",
                "mae_mm",
                "rmse_mm"
            ]
        )

    for n_removed_pcs in pcs_to_test:
        if not df_results[df_results["removed_pcs"] == n_removed_pcs].empty:
            print(f"Skipping {n_removed_pcs} PCs")
            continue
        print(f"\nFold {fold}| Remove {n_removed_pcs} PCs")

        X_train_full = inputs[train_idx]
        y_train_full = targets[train_idx]
        X_test = inputs[test_idx]
        y_test = targets[test_idx]

        (X_train, X_val, y_train, y_val) = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=fold
        )

        if n_removed_pcs > 0:
            remover = LastPCsRemover(n_components_to_remove=n_removed_pcs)
            remover.fit(X_train)

            X_train = remover.transform(X_train)
            X_val = remover.transform(X_val)
            X_test = remover.transform(X_test)
        
        model_dir = f'./nogit_NN/surface_estimation/pca_ablation/fold_{fold}/'
        os.makedirs(model_dir, exist_ok=True)

        model_path = model_dir + f"ViT_remove_{n_removed_pcs}_pcs.keras"

        if os.path.exists(model_path):
            print("Loading model")

            model = load_model(
                model_path,
                custom_objects={"rmse_loss": rmse_loss}
            )
        
        else:
            model = build_hybrid_transformer(
                img_size=(16,16,1)
            )

            model.compile(
                optimizer="adam",
                loss=rmse_loss,
                metrics=["mae"]
            )

            checkpoint_cb_cut_LEDs = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint_cb_cut_LEDs]
            )

        (mae, mae_mm, rmse_mm) = calculate_error(model, X_test, y_test)

        print(f"MAE(mm): {mae_mm:.4f}")

        results.append([
            n_removed_pcs,
            mae,
            mae_mm,
            rmse_mm
        ])

        df_results = pd.DataFrame(
            results,
            columns=[
                "removed_pcs",
                "mae",
                "mae_mm",
                "rmse_mm"
            ]
        )

        df_results.to_csv(
            save_file,
            index=False
        )

    past_files.append(save_file)
    plot_pca_component_removal(past_files=past_files, saveplotpath=f"./figures/PCA/surface_estimation/components_removal/")
