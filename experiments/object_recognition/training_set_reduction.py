import numpy as np
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

from sklearn.metrics import confusion_matrix
import seaborn as sns

experiment = "reducing_dataset"

architectures = ["OR_two_bodies_cold_start"]

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

input_shape = (16, 32, 1)

EPOCH_CHECKPOINTS = [10, 20, 30, 40, 50]
N_FOLDS = 10
N_REDUCTION = 9

val_acc_epochs = {
    arch: np.zeros((N_FOLDS, len(EPOCH_CHECKPOINTS)))
    for arch in architectures
}

val_loss_epochs = {
    arch: np.zeros((N_FOLDS, len(EPOCH_CHECKPOINTS)))
    for arch in architectures
}

all_y_true = []
all_y_pred = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

accuracies = {arch: np.zeros((N_FOLDS, N_REDUCTION)) for arch in architectures}
losses = {arch: np.zeros((N_FOLDS, N_REDUCTION)) for arch in architectures}

for fold, (train_idx, test_idx) in enumerate(skf.split(inputs, targets.ravel())):
    for architecture in architectures:
        
        #### MODEL NAME ####
        model_name = architecture   # model named used for training and compile the results.
        for red in range(N_REDUCTION):

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

            remove_fraction = red * 0.10

            if remove_fraction > 0:
                # Calculate number to remove
                n_remove = int(remove_fraction * len(X_train))

                # Choose random indices to remove
                np.random.seed(100 + fold)
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
            models_folder = f'./nogit_NN/object_recognition/training_set_reduction/fold_{fold}/{red}'
            os.makedirs(models_folder, exist_ok=True)

            # Make results dir
            results_folder = f'./results/object_recognition/training_set_reduction/fold_{fold}/{red}'
            os.makedirs(results_folder, exist_ok=True)
            
            metrics_path = os.path.join(results_folder, "metrics.txt")
            if os.path.exists(metrics_path):

                print(f"Skipping fold {fold}, arch {architecture}, red {red} (metrics found)")

                test_loss = None
                test_accuracy = None

                with open(metrics_path, "r") as f:
                    for line in f:
                        if "Loss" in line:
                            test_loss = float(line.split("=")[1])
                        elif "Accuracy" in line:
                            test_accuracy = float(line.split("=")[1])

                accuracies[architecture][fold, red] = test_accuracy * 100
                losses[architecture][fold, red] = test_loss

                continue

            checkpoint_cb = ModelCheckpoint(
                filepath=f'{models_folder}/{model_name}.weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )

            surface_model = load_model("./nogit_NN/surface_estimation/SE_Transformer_CNN_hybrid_mean_PD_removed.keras", compile=False)
            # sc_model = load_model("./nogit_NN/surface_classification/SC_35classes_HViT_estimation_transfer_unfixed.keras", compile=False)

            # # One big body for the 2 fingers, cold start
            # if model_name == "OR_one_body_cold_start":
            #     model = build_transformer_classifier()

            # Two bodies, cold start
            if model_name == "OR_two_bodies_cold_start":
                model = build_dual_finger_classifier(surface_model=surface_model, unlocked_weights=True, cold_start=True)

            # # Two bodies, transfered body weigths fixed from surface estimation
            # if model_name == "OR_two_bodies_transfered_fixed":
            #     model = build_dual_finger_classifier(surface_model=surface_model)

            # # Two bodies, transfered body weigths unfixed from surface estimation
            # if model_name == "OR_two_bodies_transfered_unfixed":
            #     model = build_dual_finger_classifier(surface_model=surface_model, unlocked_weights=True)
            
            # # Two bodies, transfered body weigths fixed from surface classification
            # if model_name == "OR_two_bodies_transfered_SC_fixed":
            #     model = build_dual_finger_classifier(surface_model=sc_model)

            # # Two bodies, transfered body weigths unfixed from surface classification
            # if model_name == "OR_two_bodies_transfered_SC_unfixed":
            #     model = build_dual_finger_classifier(surface_model=sc_model, unlocked_weights=True)

            if os.path.exists(f'{models_folder}/{model_name}.weights.h5'):
                print("Model was trained previously")
            else:
                print("Training the model...")
                history = model.fit(
                    X_train_reduced, y_train_reduced,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=32,
                    callbacks=[checkpoint_cb]
                )
                # Plot training and validation loss
                plot_train_val_loss(history, saveplotpath=f"{results_folder}/train_val_loss", show=False)

                val_metrics = {
                    "epoch": [],
                    "val_loss": [],
                    "val_accuracy": []
                }

                for ep in EPOCH_CHECKPOINTS:
                    idx = ep - 1  # history is 0-indexed
                    val_metrics["epoch"].append(ep)
                    val_metrics["val_loss"].append(history.history["val_loss"][idx])
                    val_metrics["val_accuracy"].append(history.history["val_accuracy"][idx])

                np.save(
                    os.path.join(results_folder, "val_metrics_by_epoch.npy"),
                    val_metrics
                )

            if model_name == "OR_one_body_cold_start":
                model = build_transformer_classifier()
            elif model_name == "OR_two_bodies_transfered_SC_fixed" or model_name == "OR_two_bodies_transfered_SC_unfixed":
                model = build_dual_finger_classifier(surface_model=sc_model)
            else:
                model = build_dual_finger_classifier(surface_model=surface_model)
            
            val_metrics_path = os.path.join(results_folder, "val_metrics_by_epoch.npy")

            if os.path.exists(val_metrics_path):
                val_metrics = np.load(val_metrics_path, allow_pickle=True).item()

                val_acc_epochs[architecture][fold, :] = np.array(
                    val_metrics["val_accuracy"]
                ) * 100

                val_loss_epochs[architecture][fold, :] = np.array(
                    val_metrics["val_loss"]
                )
            # Make predictions
            if not os.path.exists(f'{results_folder}/metrics.txt'):
                model.load_weights(f'{models_folder}/{model_name}.weights.h5')
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                accuracies[architecture][fold, red] = test_accuracy * 100
                losses[architecture][fold, red] = test_loss
                print(f"Test Accuracy: {test_accuracy:.2f}")
                with open(os.path.join(results_folder, "metrics.txt"), "w") as f:
                    f.write(f"Accuracy = {test_accuracy:.3f}\n")
                    f.write(f"Loss = {test_loss:.6f}\n")
                # Plot confusion matrix on the test set
                plot_CM(model, X_test, y_test, saveplotpath=f"{results_folder}/confusion_matrix_object_classification", show=False)
            else:
                # Extract loss and accuracy from metrics file

                test_loss = None
                test_accuracy = None

                with open(metrics_path, "r") as f:
                    for line in f:
                        if "Loss" in line:
                            test_loss = float(line.split("=")[1])
                        elif "Accuracy" in line:
                            test_accuracy = float(line.split("=")[1])
                accuracies[architecture][fold, red] = test_accuracy * 100
                losses[architecture][fold, red] = test_loss
            

exp_values = np.arange(N_REDUCTION) * 10   # 0%, 10%, ..., 80%

plt.figure(figsize=(10,6))

best_arch = "OR_two_bodies_cold_start"

box_data = [accuracies[best_arch][:, r] for r in range(N_REDUCTION)]

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

# plt.title("Test Accuracy with reduced Training Set (k=10)")
plt.xlabel("Training Data Removed (%)", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.xticks(exp_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "./results/object_recognition/training_set_reduction/training_set_reduction.png",
    dpi=300
)
plt.show()

# Concatenate all folds
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

# Compute confusion matrix
class_names = ["empty", "cube", "screw driver", "tennis", "can",
                   "cylinder", "pyramid", "cup", "puck"]
num_classes = len(class_names)
cm = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(num_classes))
global_accuracy = np.trace(cm) / np.sum(cm)
print(f"Global Accuracy (from CM): {global_accuracy*100:.2f}%")
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
# Plot
plt.figure(figsize=(10,8))
ax = sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
        yticklabels=class_names, annot_kws={"size": 16})

# plt.title("Global Confusion Matrix")
ax.set_xlabel("Predicted Label", fontsize=18)
ax.set_ylabel("True Label", fontsize=18)
ax.tick_params(axis='both', labelsize=16)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("./results/object_recognition/global_confusion_matrix.png", dpi=300)
plt.show()