import random
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from data_preprocessing import finger_paring
from build_model import build_CNN_model, build_CNN_encoder_decoder, build_hybrid_transformer_for_width, build_dual_finger_estimator

experiment = "reducing_dataset"

random.seed(100)
tf.random.set_seed(100)
np.random.seed(100)
tf.keras.backend.clear_session()

architectures = ["WP_one_big_body_bigger_CNN"]
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

# plt.figure(figsize=(6, 6))
# plt.imshow(inputs[0], cmap='viridis', origin='lower')
# plt.show()

inputs_reshape = inputs.reshape(np.shape(inputs)[0], -1)
targets_widths_reshape = np.array(target_widths).reshape(np.shape(target_widths)[0], -1)

scaler_folder = './nogit_NN/width_prediction/architecture_comparison/reducing_dataset'

surface_model = load_model("./nogit_NN/surface_estimation/preprocessing_comparison/SE_Transformer_CNN_hybrid_mean_PD_removed.keras", compile=False)
sc_model = load_model("./nogit_NN/surface_classification/SC_35classes_HViT_estimation_transfer_unfixed/SC_35classes_HViT_estimation_transfer_unfixed.keras", compile=False)

EPOCH_CHECKPOINTS = [10, 20, 30, 40, 50]
N_FOLDS = 10
N_REDUCTION = 9

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=100)

MAEs = {arch: np.zeros((N_FOLDS, N_REDUCTION)) for arch in architectures}
RMSEs = {arch: np.zeros((N_FOLDS, N_REDUCTION)) for arch in architectures}

val_MAE_epochs = {
    arch: np.zeros((N_FOLDS, len(EPOCH_CHECKPOINTS)))
    for arch in architectures
}

all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_reshape)):
    for architecture in architectures:

        #### MODEL NAME ####
        model_name = architecture   # model named used for training and compile the results.

        for red in range(N_REDUCTION):

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

            input_shape = (16, 32, 1)

            remove_fraction = red * 0.10
            np.random.seed(100 + fold)

            if remove_fraction > 0:
                # Calculate number to remove
                n_remove = int(remove_fraction * len(X_train))

                # Choose random indices to remove
                
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
            
            np.random.seed(100)

            # Make model dir
            models_folder = f'./nogit_NN/width_prediction/training_set_reduction/fold_{fold}/{red}'
            os.makedirs(models_folder, exist_ok=True)

            # Make results dir
            results_folder = f'./results/width_prediction/training_set_reduction/fold_{fold}/{red}'
            os.makedirs(results_folder, exist_ok=True)
            metrics_path = os.path.join(results_folder, "metrics.txt")
            if os.path.exists(metrics_path):
                
                if red == 0:
                    model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                    model.load_weights(f'{models_folder}/{model_name}.weights.h5')
                    y_pred = model.predict(X_test)
                    unscaled_y_pred = y_scaler.inverse_transform(y_pred)
                    unscaled_y_test = y_test
                    all_y_true.append(unscaled_y_test.flatten())
                    all_y_pred.append(unscaled_y_pred.flatten())

                with open(metrics_path, "r") as f:
                    lines = f.readlines()

                metrics = {}
                for line in lines:
                    key, val = line.strip().split("=")
                    metrics[key.strip()] = float(val)

                MAEs[architecture][fold, red] = metrics["MAE_mm"]
                RMSEs[architecture][fold, red] = metrics["RMSE_mm"]
                print(f'Results of fold {fold}, arch {architecture}, red {red} acquired in results file')

            else:
                checkpoint_cb = ModelCheckpoint(
                    filepath=f'{models_folder}/{model_name}.weights.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
                tf.keras.backend.clear_session()
                # # One big body for the 2 fingers, cold start
                # if model_name == "WP_one_big_body":
                #     model = build_hybrid_transformer_for_width()

                # # Two bodies, cold start
                # if model_name == "WP_two_bodies_cold_start":
                #     model = build_dual_finger_estimator(surface_model=surface_model, unlocked_weights=True, cold_start=True)

                # # Two bodies, transfered body weigths fixed
                # if model_name == "WP_two_bodies_transfered_fixed":
                #     model = build_dual_finger_estimator(surface_model=surface_model)

                # # Two bodies, transfered body weigths unfixed
                # if model_name == "WP_two_bodies_transfered_unfixed":
                #     model = build_dual_finger_estimator(surface_model=surface_model, unlocked_weights=True)

                # # Two bodies, transfered body weigths fixed from surface classification
                # if model_name == "WP_two_bodies_transfered_SC_fixed":
                #     model = build_dual_finger_estimator(surface_model=sc_model)

                # # Two bodies, transfered body weigths unfixed from surface classification
                # if model_name == "WP_two_bodies_transfered_SC_unfixed":
                #     model = build_dual_finger_estimator(surface_model=sc_model, unlocked_weights=True)

                if model_name == "WP_one_big_body_bigger_CNN":
                    model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))

                if os.path.exists(f'{models_folder}/{model_name}.weights.h5'):
                    print("Model was trained previously")
                else:
                    print("Training the model...")
                    history = model.fit(
                        X_train_reduced, y_train_reduced,
                        validation_data=(X_val, y_val),
                        epochs=150,
                        batch_size=32,
                        callbacks=[checkpoint_cb]
                    )

                    val_metrics = {
                        "epoch": [],
                        "val_loss": [],
                        "val_accuracy": []
                    }

                    for ep in EPOCH_CHECKPOINTS:
                        idx = ep - 1  # history is 0-indexed
                        val_metrics["epoch"].append(ep)
                        val_metrics["val_loss"].append(history.history["val_loss"][idx])

                    np.save(
                        os.path.join(results_folder, "val_metrics_by_epoch.npy"),
                        val_metrics
                    )
                tf.keras.backend.clear_session()
                if model_name == "WP_one_big_body":
                    model = build_hybrid_transformer_for_width()
                elif model_name == "WP_one_big_body_bigger_CNN":
                    model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                elif model_name == "WP_two_bodies_transfered_SC_fixed" or model_name == "WP_two_bodies_transfered_SC_unfixed":
                    model = build_dual_finger_estimator(surface_model=sc_model)
                else:
                    model = build_dual_finger_estimator(surface_model=surface_model)
                model.load_weights(f'{models_folder}/{model_name}.weights.h5')
                val_metrics_path = os.path.join(results_folder, "val_metrics_by_epoch.npy")

                if os.path.exists(val_metrics_path):
                    val_metrics = np.load(val_metrics_path, allow_pickle=True).item()

                    val_MAE_epochs[architecture][fold, :] = np.array(
                        val_metrics["val_loss"]
                    )

                # Predict on the test set
                if experiment == "80pct_removed" or experiment == "reducing_dataset":
                    y_pred = model.predict(X_test)
                    unscaled_y_pred = y_scaler.inverse_transform(y_pred)
                    unscaled_y_test = y_test
                    errors = unscaled_y_pred.flatten() - unscaled_y_test.flatten()
                    x = unscaled_y_test.flatten()
                    y = unscaled_y_pred.flatten()
                    coeffs = np.polyfit(x, y, 1)
                    a, b = coeffs 
                    fit_line = np.poly1d(coeffs)

                    # Generate points for the fit line
                    x_fit = np.linspace(min(unscaled_y_test), max(unscaled_y_test), 100)
                    y_fit = fit_line(x_fit)

                    mean_error = np.mean(errors)
                    std_error = np.std(errors)

                    mae_in_mm = np.mean(abs(unscaled_y_test-unscaled_y_pred))
                    print("Error in mm: ", mae_in_mm)

                    rmse = np.sqrt(np.mean(errors**2))
                    bias = mean_error
                    print(f"RMSE² = bias² + σ² = {rmse**2:.3f} ≈ {bias:.3f}**2 + {std_error:.3f}**2 = {bias**2 + std_error**2:.3f}")

                    distances = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
                    std_perp_error = np.std(distances)
                    mean_perp_error = np.mean(distances)

                    MAEs[architecture][fold, red] = mae_in_mm
                    RMSEs[architecture][fold, red] = rmse

                    if not os.path.exists(f'{models_folder}/metrics.txt'):
                        # ---- Save MAE and RMSE to a text file ----
                        with open(os.path.join(results_folder, "metrics.txt"), "w") as f:
                            f.write(f"MAE_mm = {mae_in_mm:.4f}\n")
                            f.write(f"RMSE_mm = {rmse:.4f}\n")
                            f.write(f"Bias = {bias:.4f}\n")
                            f.write(f"STD = {std_error:.4f}\n")
                            f.write(f"Perp_mean = {mean_perp_error:.4f}\n")
                            f.write(f"Perp_std = {std_perp_error:.4f}\n")

                        print(f"Metrics saved to {results_folder}/metrics.txt")

                        # ---- Re-plot and save the figure ----
                        plt.figure()
                        plt.scatter(unscaled_y_test, unscaled_y_pred, label='Width')
                        plt.plot([61.2, 93.6], [61.2, 93.6], color='red', linestyle='--', label='Ideal line')
                        plt.plot(x_fit, y_fit, color='green', label=f'Fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
                        plt.xlabel('Target Width (mm)')
                        plt.ylabel('Predicted Width (mm)')
                        plt.title(f'Target vs Predicted (MAE = {mae_in_mm:.2f} mm)')
                        plt.legend()

                        plot_path = os.path.join(results_folder, "target_vs_pred.png")
                        plt.savefig(plot_path, dpi=300)
                        plt.close()

                        print(f"Plot saved to {plot_path}")

exp_values = np.arange(N_REDUCTION) * 10   # 0%, 10%, ..., 80% removed

plt.figure(figsize=(10,6))

best_arch = "WP_one_big_body_bigger_CNN"

box_data = [MAEs[best_arch][:, r] for r in range(N_REDUCTION)]

plt.boxplot(
        box_data,
        positions=exp_values,
        widths=3,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

data_mean = np.mean(box_data, axis=1)
plt.plot(exp_values, data_mean, '^-', color='blue', label='mean')

# plt.title("Test MAE with reduced Training Set (n=10)")
plt.xlabel("Training Data Removed (%)", fontsize=14)
plt.ylabel("Test MAE (mm)", fontsize=14)
plt.xticks(exp_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "./results/width_prediction/training_set_reduction/training_set_reduction.png",
    dpi=300
)
plt.show()

all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

# Fit line
coeffs = np.polyfit(all_y_true, all_y_pred, 1)
a, b = coeffs
fit_line = np.poly1d(coeffs)

x_fit = np.linspace(min(all_y_true), max(all_y_true), 200)
y_fit = fit_line(x_fit)

# Errors
errors = all_y_pred - all_y_true
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
bias = np.mean(errors)
std = np.std(errors)

# Plot
plt.figure(figsize=(7,7))

plt.scatter(all_y_true, all_y_pred, alpha=0.5)

# # Ideal line
# plt.plot(
#     [min(all_y_true), max(all_y_true)],
#     [min(all_y_true), max(all_y_true)],
#     linestyle='--',
#     label='Ideal'
# )

# Fit line
# plt.plot(x_fit, y_fit, label=f'Fit: y={a:.3f}x+{b:.3f}')

plt.xlabel('Target Width (mm)', fontsize=14)
plt.ylabel('Predicted Width (mm)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title(f'Global Regression (MAE={mae:.2f} mm)')

# plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "./results/width_prediction/global_regression.png",
    dpi=300
)

plt.show()