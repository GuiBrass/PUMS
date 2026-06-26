import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers

from data_preprocessing import finger_paring
from build_model import build_CNN_model, build_CNN_encoder_decoder, build_hybrid_transformer_for_width, build_dual_finger_estimator

#Get dataset
folder_path = "./nogit_datasets/dataset_width_prediction/dataset_15-07-25"

csv_files = glob.glob(os.path.join(folder_path+"/acquisition", '*.csv')) + glob.glob(os.path.join(folder_path+"/calib", '*.csv'))


#calibration
inputs, calib_pd_values, target_widths = finger_paring(csv_files)
inputs = np.array(inputs)
baseline = np.mean(calib_pd_values, axis=0)
print(np.shape(baseline))

inputs_reshape = inputs.reshape(np.shape(inputs)[0], -1)
targets_widths_reshape = np.array(target_widths).reshape(np.shape(target_widths)[0], -1)

# inputs = pd_values-baseline
col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means

def flip_fingers(X):
    """
    Swap finger 1 and finger 2.
    Assumes shape (N, 16, 32, 1)
    """
    X_flipped = X.copy()

    # Split fingers
    finger1 = X[:, :8, :, :]
    finger2 = X[:, 8:16, :, :]

    # Swap
    X_flipped[:, :8, :, :] = finger2
    X_flipped[:, 8:16, :, :] = finger1

    return X_flipped

input_shape = (16, 32, 1)
l2_reg = regularizers.l2(1e-3)

# architectures = ['cold', 'finetune', 'transfer', 'transfer_head']
# architectures = ['cold', 'finetune', 'transfer']
architectures = ['cold', 'transfer']

N_FOLDS = 10
N_SAMPLES = 13

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=100)

MAEs = {arch: np.zeros((N_FOLDS, N_SAMPLES)) for arch in architectures}
RMSEs = {arch: np.zeros((N_FOLDS, N_SAMPLES)) for arch in architectures}

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs_reshape)):

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

    # Flipped versions
    X_train_flipped = flip_fingers(X_train)
    X_val_flipped   = flip_fingers(X_val)
    X_test_flipped  = flip_fingers(X_test)

    # Targets stay the same
    y_train_flipped = y_train.copy()
    y_val_flipped   = y_val.copy()
    y_test_flipped  = y_test.copy()

    for arch in architectures:
        for samp in range(N_SAMPLES):
            results_path = f'./figures/figure_8_increase_shots_transfer_learning/flip_finger_WP/fold_{fold}/{arch}/{samp}/'
            model_path = f'./nogit_NN/width_prediction/flip_finger/fold_{fold}/{arch}/{samp}/'

            np.random.seed(100 + fold)

            if os.path.exists(f'{results_path}/metrics.txt'):
                with open(f'{results_path}/metrics.txt', "r") as f:
                    lines = f.readlines()

                metrics = {}
                for line in lines:
                    key, val = line.strip().split("=")
                    metrics[key.strip()] = float(val)

                MAEs[arch][fold, samp] = metrics["test_mae"]
                RMSEs[arch][fold, samp] = metrics["test_rmse"]
                print(f'Results of exp {fold}, arch {samp}, samp {samp} acquired in results file')
                

            else:
                weights_file = model_path + f'{arch}_{samp}.weights.h5'
                if samp == 0:
                    print("Zero-shot / pure transfer evaluation")

                    model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))

                    if arch != "cold":
                        model.load_weights(
                            './nogit_NN/width_prediction/architecture_comparison/reducing_dataset/fold_0/'
                            'WP_one_big_body_bigger_CNN/0/WP_one_big_body_bigger_CNN.weights.h5'
                        )

                    # no training at all
                    if os.path.exists(weights_file):
                        model.load_weights(weights_file)

                else:
                    if arch == "cold":
                        model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))

                    if arch == "finetune":
                        model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                        model.load_weights(
                            './nogit_NN/width_prediction/architecture_comparison/reducing_dataset/fold_0/'
                            'WP_one_big_body_bigger_CNN/0/WP_one_big_body_bigger_CNN.weights.h5'
                        )

                    if arch == "transfer":
                        model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                        model.load_weights('./nogit_NN/width_prediction/architecture_comparison/reducing_dataset/fold_0/WP_one_big_body_bigger_CNN/0/WP_one_big_body_bigger_CNN.weights.h5')

                        for layer in model.layers[-3:]:
                            if hasattr(layer, 'kernel_initializer'):
                                layer.kernel.assign(
                                    layer.kernel_initializer(
                                        shape=layer.kernel.shape
                                    )
                                )
                            if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                                layer.bias.assign(
                                    layer.bias_initializer(
                                        shape=layer.bias.shape
                                    )
                                )

                        for layer in model.layers[:-3]:
                            layer.trainable = False

                        # Keep last 2 trainable
                        for layer in model.layers[-3:]:
                            layer.trainable = True

                        model.compile(
                            optimizer=tf.keras.optimizers.Adam(1e-4),
                            loss='mse',
                            metrics=['mae']
                        )
                    
                    # if arch == "transfer_head":
                    #     model = build_hybrid_transformer_for_width(CNN_layers_size=(128,256))
                    #     model.load_weights('./nogit_NN/width_prediction/architecture_comparison/reducing_dataset/fold_0/WP_one_big_body_bigger_CNN/0/WP_one_big_body_bigger_CNN.weights.h5')

                    #     for layer in model.layers[:3]:
                    #         if hasattr(layer, 'kernel_initializer'):
                    #             layer.kernel.assign(
                    #                 layer.kernel_initializer(
                    #                     shape=layer.kernel.shape
                    #                 )
                    #             )
                    #         if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                    #             layer.bias.assign(
                    #                 layer.bias_initializer(
                    #                     shape=layer.bias.shape
                    #                 )
                    #             )

                    #     for layer in model.layers[3:]:
                    #         layer.trainable = False

                    #     # Keep last 2 trainable
                    #     for layer in model.layers[:3]:
                    #         layer.trainable = True

                    #     model.compile(
                    #         optimizer=tf.keras.optimizers.Adam(1e-4),
                    #         loss='mse',
                    #         metrics=['mae']
                    #     )

                    selected_indices = np.random.choice(
                        X_train_flipped.shape[0],
                        size=samp * 5,
                        replace=False
                    )

                    X_train_samp = X_train_flipped[selected_indices]
                    y_train_samp = y_train_flipped[selected_indices]

                    os.makedirs(model_path, exist_ok=True)

                    if not os.path.exists(weights_file):
                        checkpoint_cb = ModelCheckpoint(
                            filepath=weights_file,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True
                        )

                        model.fit(
                            X_train_samp,
                            y_train_samp,
                            validation_data=(X_val_flipped, y_val_flipped),
                            epochs=150,
                            batch_size=32,
                            callbacks=[checkpoint_cb]
                        )

                if samp > 0:
                    model.load_weights(model_path + f'{arch}_{samp}.weights.h5')
                y_pred = model.predict(X_test_flipped)
                unscaled_y_pred = y_scaler.inverse_transform(y_pred)
                unscaled_y_test = y_test_flipped
                print(f'pred:{unscaled_y_pred}   tar: {unscaled_y_test}')
                errors = unscaled_y_pred.flatten() - unscaled_y_test.flatten()
                mae_in_mm = np.mean(abs(unscaled_y_test-unscaled_y_pred))
                print("unflat:", abs(unscaled_y_test-unscaled_y_pred))
                print("flat:", abs(unscaled_y_pred.flatten() - unscaled_y_test.flatten()))
                rmse = np.sqrt(np.mean(errors**2))
                MAEs[arch][fold, samp] = mae_in_mm
                RMSEs[arch][fold, samp] = rmse
                print(f'Results of exp {fold}, arch {samp}, samp {samp}')
                if not os.path.exists(f'{results_path}/metrics.txt'):
                    os.makedirs(results_path, exist_ok=True)
                    with open(os.path.join(results_path, "metrics.txt"), "w") as f:
                        f.write(f"test_mae = {mae_in_mm:.4f}\n")
                        f.write(f"test_rmse = {rmse:.4f}\n")

exp_values = np.arange(N_SAMPLES)*5

plt.figure(figsize=(10,6))

box_width = 1.5
offset = box_width/2 + 0.1

ax = plt.gca()

colors = {
    "cold": "blue",
    # "finetune": "orange",
    "transfer": "green"
}

offsets = {
    "cold": -offset,
    # "finetune": 0,
    "transfer": offset,
}


boxplots = {}

for i, arch in enumerate(architectures):
    
    # shift positions so the boxplots sit side-by-side
    positions = exp_values + offsets[arch]

    box_data = [MAEs[arch][:, s] for s in range(N_SAMPLES)]

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color=colors[arch]),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )
    data_mean = np.mean(box_data, axis=1)
    ax.plot(positions, data_mean, '^-', color=colors[arch], label='mean')
    boxplots[arch] = bp

# legend_elements = [
#     Patch(facecolor="skyblue", edgecolor="black", label="Cold start"),
#     Patch(facecolor="orange", edgecolor="black", label="Fine Tuning"),
#     Patch(facecolor="purple", edgecolor="black", label="TL feature extractor")
# ]

# ax.legend(handles=legend_elements, loc="upper right")

# ax.set_title("Width Estimation: Test MAE While Increasing the Flipped Samples (k=10)")
ax.set_xlabel("Number of Samples", fontsize=14)
ax.set_ylabel("Test MAE", fontsize=14)

ax.set_xticks(exp_values)
ax.set_xticklabels(exp_values, fontsize=12)
plt.yticks(fontsize=12)

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("./figures/figure_8_increase_shots_transfer_learning/flip_finger_WP/MAE_summary.png", dpi=300)
plt.show()
                