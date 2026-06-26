import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from data_preprocessing import extract_csv, extract_ply_global_min_max, get_files, get_SC_replace_files, PC0Remover
from model_builder import build_hybrid_transformer, rmse_loss, rmse_loss_threshold
from results_analysis import calculate_error, plot_surfaces, plot_train_val_loss

import matplotlib.pyplot as plt

#Get dataset
folder_path = "./nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25"


csv_files_calib, csv_files_inputs, ply_files_targets, csv_files_test_inputs, ply_files_test_targets = get_files(folder_path)
calib = extract_csv(csv_files_calib)
calib = np.array(calib)
inputs = np.array(extract_csv(csv_files_inputs))
test_inputs = np.array(extract_csv(csv_files_test_inputs))
targets, glob_min, glob_max = extract_ply_global_min_max(ply_files_targets)
test_targets, test_glob_min, test_glob_max = extract_ply_global_min_max(ply_files_test_targets)

ply_files_pred_cp, ply_files_tar_cp = get_SC_replace_files(folder_path)
SC_pred, SC_pred_glob_min, SC_pred_glob_max = extract_ply_global_min_max(ply_files_pred_cp)
SC_tar, SC_tar_glob_min, SC_tar_glob_max = extract_ply_global_min_max(ply_files_tar_cp)

print("inputs shape", np.shape(inputs))
print("targets shape", np.shape(targets))

col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means

indices = np.arange(len(inputs))

experiments = [
    {"name":"default", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 4, "projected_dim": 128, "pos":True},
    {"name":"no_transformer", "cnn":True, "cnn_size":[64,128], "transformer":False, "n_trans_layer": 4, "projected_dim": 128, "pos":False},
    {"name":"no_cnn", "cnn":False, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 4, "projected_dim": 128, "pos":True},
    {"name":"no_pos", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 4, "projected_dim": 128, "pos":False},
    {"name":"smaller_cnn", "cnn":True, "cnn_size":[32,64], "transformer":True, "n_trans_layer": 4, "projected_dim": 128, "pos":True},
    {"name":"bigger_cnn", "cnn":True, "cnn_size":[128,256], "transformer":True, "n_trans_layer": 4, "projected_dim": 128, "pos":True},
    {"name":"two_transformer_blocks", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 2, "projected_dim": 128, "pos":True},
    {"name":"six_transformer_blocks", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 6, "projected_dim": 128, "pos":True},
    {"name":"projected_dim_64", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 4, "projected_dim": 64, "pos":True},
    {"name":"projected_dim_256", "cnn":True, "cnn_size":[64,128], "transformer":True, "n_trans_layer": 4, "projected_dim": 256, "pos":True},
    {"name":"small_default", "cnn":True, "cnn_size":[32,64], "transformer":True, "n_trans_layer": 2, "projected_dim": 64, "pos":True},
    {"name":"big_default", "cnn":True, "cnn_size":[128,256], "transformer":True, "n_trans_layer": 6, "projected_dim": 256, "pos":True},
]


past_files = []
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):

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

    save_path = f"./figures/figure_4_surface_estimation/ablation/fold_{fold}/ablation_results.csv"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        results_df = pd.read_csv(save_path)
        done_experiments = set(results_df["experiment"].values)
        print(f"Resuming fold {fold}, found {len(done_experiments)} completed experiments.")
    else:
        results_df = pd.DataFrame()
        done_experiments = set()
    for exp in experiments:
        if exp["name"] in done_experiments:
            print(f"Skipping {exp['name']} (already done)")
            continue
        
        print("Running:", exp["name"])

        checkpoint_cb = ModelCheckpoint(
            filepath=f'./nogit_NN/surface_estimation/ablation/fold_{fold}/{exp["name"]}.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )

        model = build_hybrid_transformer(
            cnn_size = exp["cnn_size"],
            projection_dim = exp["projected_dim"],
            transformer_layers=exp["n_trans_layer"],
            use_cnn=exp["cnn"],
            use_transformer=exp["transformer"],
            use_pos_encoding=exp["pos"]
        )

        model.compile(optimizer="adam", loss=rmse_loss, metrics=["mae"])
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[checkpoint_cb]
        )

        model.summary()

        # model = load_model('./nogit_NN/surface_estimation/preprocessing_comparison/small_model.keras')

        _, mae_mm, _ = calculate_error(model, X_test, y_test)
        _, unseen_mae_mm, _ = calculate_error(model, test_inputs, test_targets)

        row = {
            "experiment": exp["name"],
            "cnn": exp["cnn"],
            "cnn_size": exp["cnn_size"],
            "transformer": exp["transformer"],
            "n_trans_layer": exp["n_trans_layer"],
            "projected_dim": exp["projected_dim"],
            "pos_encoding": exp["pos"],
            "mae_mm": mae_mm,
            "unseen_mae_mm": unseen_mae_mm,
            "model_params_count": model.count_params(),
            "model_size": os.path.getsize(f'./nogit_NN/surface_estimation/ablation/fold_{fold}/{exp["name"]}.keras') / (1024 * 1024)
        }

        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df.to_csv(save_path, index=False)

    results_df.to_csv(save_path, index=False)

    print("Saved results to:", save_path)

all_results = []

for fold in range(N_FOLDS):
    path = f"./figures/figure_4_surface_estimation/ablation/fold_{fold}/ablation_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["fold"] = fold
        all_results.append(df)

all_results_df = pd.concat(all_results, ignore_index=True)

# ==========================
# Summary statistics
# ==========================

summary = (
    all_results_df
    .groupby("experiment")
    .agg(
        mae_mean=("mae_mm", "mean"),
        mae_std=("mae_mm", "std"),
        mae_median=("mae_mm", "median"),
        n_folds=("mae_mm", "count"),
        params_mean=("model_params_count", "mean"),
        model_size_mean=("model_size", "mean")
    )
    .reset_index()
)

summary = summary.sort_values("mae_mean")

summary_path = "./figures/figure_4_surface_estimation/ablation/final_ablation_summary.csv"
summary.to_csv(summary_path, index=False)

print("Saved summary to:", summary_path)

name_map = {
    "default": "Baseline",
    "no_transformer": "No ViT",
    "no_cnn": "No CNN",
    "no_pos": "No Pos. Encoding",
    "smaller_cnn": "CNN 32-64",
    "bigger_cnn": "CNN 128-256",
    "two_transformer_blocks": "2 Trans. Layers",
    "six_transformer_blocks": "6 Trans. Layers",
    "projected_dim_64": "Emb. Dim. 64",
    "projected_dim_256": "Emb. Dim. 256",
    "small_default": "Small Model",
    "big_default": "Large Model"
}

mean_scores = all_results_df.groupby("experiment")["mae_mm"].mean().sort_values()
order = mean_scores.index

data = [
    all_results_df.loc[all_results_df["experiment"] == exp, "mae_mm"].values
    for exp in order
]

labels = [name_map.get(exp, exp) for exp in order]

default_mean = mean_scores["default"]

plt.figure(figsize=(10, 8))

plt.boxplot(
    data,
    vert=False,
    labels=labels,
    showmeans=True
)

plt.axvline(
    default_mean,
    linestyle="--",
    linewidth=1.8,
    # label=f"Baseline mean ({default_mean:.3f} mm)"
)

plt.xlabel("MAE (mm)")
# plt.title("Ablation Study Across 10 Folds")

plt.legend()
plt.tight_layout()

plt.savefig(
    "./figures/figure_4_surface_estimation/ablation/final_ablation_boxplot.png",
    dpi=300
)

plt.show()