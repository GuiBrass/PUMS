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
from results_analysis import calculate_error, plot_train_val_loss, plot_training_set_reduction, plot_surfaces

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
N_REDUCTION = 10

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

past_files = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    results = []
    for red in range(N_FOLDS):

        X_train_full = inputs[train_idx]
        X_test = inputs[test_idx]

        y_train_full = targets[train_idx]
        y_test = targets[test_idx]

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
        
        # Make model dir
        models_folder = f'./nogit_NN/surface_estimation/training_set_reduction/fold_{fold}/'
        os.makedirs(models_folder, exist_ok=True)

        # Make results dir
        results_folder = f'./figures/figure_4_surface_estimation/training_set_reduction/fold_{fold}/'
        os.makedirs(results_folder, exist_ok=True)

        if os.path.exists(f'{models_folder}/{red}.weights.h5'):
                print("Model was trained previously")
        else:
            print("Training the model...")
            model = build_hybrid_transformer()
            model.compile(optimizer="adam", loss=rmse_loss, metrics=["mae"])
            checkpoint = ModelCheckpoint(filepath=f'{models_folder}/{red}.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
            history = model.fit(
                X_train_reduced, y_train_reduced,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[checkpoint]
            )
        
        model = build_hybrid_transformer()
        model.load_weights(f'{models_folder}/{red}.weights.h5')
        model.summary()

        mae, mae_mm, rmse_mm = calculate_error(model, X_test, y_test)

        if fold == 0 and red == 0 :
            inputs_max = X_train_full.max()
            inputs_min = X_train_full.min()

            plot_surfaces(5, model, X_test, y_test, inputs_min, inputs_max, "./figures")

        results.append([
            red,
            mae, mae_mm, rmse_mm
        ])

    df_results = pd.DataFrame(
        results,
        columns=[
            "trainset_reduced",
            "mae", "mae_mm", "rmse_mm",
        ]
    )
    save_file = f"{results_folder}/results"
    # df_results.to_csv(save_file, index=False)
    past_files.append(save_file)

plot_training_set_reduction(
    past_files,
    saveplotpath=f"./figures/figure_4_surface_estimation/training_set_reduction/training_set_reduction_error",
    show=True
)
