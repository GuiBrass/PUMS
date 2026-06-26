import time
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from data_preprocessing import extract_csv, extract_ply_global_min_max, get_files, get_SC_replace_files, PC0Remover
from model_builder import build_hybrid_transformer, rmse_loss, rmse_loss_threshold
from results_analysis import calculate_error, plot_surfaces, plot_train_val_loss, plot_pred_surface

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

# plt.figure(figsize=(6, 6))
# plt.imshow(inputs_clean[0], cmap='viridis', origin='lower', vmin=inputs_min, vmax=inputs_max)
# plt.show()

indices = np.arange(len(inputs))

X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = train_test_split(
    inputs, targets, indices, test_size=0.15, random_state=1
)

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_train_full, y_train_full, idx_train_full, test_size=0.15, random_state=1
)

# #Removes the first component (ambient light) obtained with PCA
# remover = PC0Remover()

# #Fit on the training data only
# remover.fit(X_train_full)

# X_train = remover.transform(X_train)
# X_val = remover.transform(X_val)
# X_test = remover.transform(X_test)

inputs_max = X_train_full.max()
inputs_min = X_train_full.min()

checkpoint_cb = ModelCheckpoint(
    filepath='./nogit_NN/surface_estimation/preprocessing_comparison/SE_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

model = build_hybrid_transformer()
model.summary()

# Compile the model
model.compile(optimizer="adam", loss=rmse_loss, metrics=["mae"])

# Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=250,
#     batch_size=32,
#     callbacks=[checkpoint_cb]
# )


# model = load_model('./nogit_NN/surface_estimation/preprocessing_comparison/SE_model.keras')
model = load_model('./nogit_NN/surface_estimation/ablation/fold_0/default.keras')
model.summary()

# Plot training and validation loss
# plot_train_val_loss(history, "./figures/figure_4_surface_estimation/training_val_loss")
# plot_train_val_loss(history)


calculate_error(model, X_test, y_test)
calculate_error(model, test_inputs, test_targets)

calculate_error(model, X_test, y_test, threshold_mm=1)
calculate_error(model, test_inputs, test_targets, threshold_mm=1)

num_samples_to_show = 10

true_mm_all = y_test[0:4] * 4.5  # since z_max_mm = 4.5
pred_mm_all = np.squeeze(model.predict(X_test[0:4])) * 4.5
error_mm_all = np.abs(pred_mm_all - true_mm_all)

global_height_max = np.max(true_mm_all)
global_error_max = np.max(error_mm_all)
print(f"Global height max: {global_height_max:.3f} mm")
print(f"Global error max: {global_error_max:.3f} mm")

MAE_mm_SC = np.mean(np.abs(SC_pred*4.5-SC_tar*4.5))
print("SC MAE (mm)", MAE_mm_SC)

# plot_pred_surface(model, X_test, y_test, "./figures", "pred")
# plot_pred_surface(model, X_test, y_test, "./figures", "target")

plot_surfaces(num_samples_to_show, model, X_test, y_test, inputs_min, inputs_max, "./figures")

# plot_surfaces(num_samples_to_show, model, test_inputs, test_targets, inputs_min, inputs_max, "./figures", unseen=True)


# MODEL LATENCY TEST

# print("\nRunning inference latency test...")

# # Use one representative sample (batch size = 1)
# sample = tf.convert_to_tensor(X_test[0:1], dtype=tf.float32)

# # Wrap model in tf.function for graph execution (more realistic)
# @tf.function
# def inference_step(x):
#     return model(x, training=False)

# # Warm-up runs (important!)
# for _ in range(20):
#     _ = inference_step(sample)

# # If using GPU, force synchronization before timing
# if tf.config.list_physical_devices('GPU'):
#     tf.experimental.sync_devices()

# # Timed runs
# n_runs = 500
# times = []

# for _ in range(n_runs):
#     start = time.perf_counter()
#     _ = inference_step(sample)
    
#     end = time.perf_counter()
#     times.append(end - start)

# times = np.array(times)

# print("--------------------------------------------------")
# print(f"Hardware: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
# print(f"Input shape: {sample.shape}")
# print(f"Batch size: 1")
# print(f"Runs: {n_runs}")
# print("--------------------------------------------------")
# print(f"Mean latency: {times.mean()*1000:.3f} ms")
# print(f"Std deviation: {times.std()*1000:.3f} ms")
# print(f"Min latency: {times.min()*1000:.3f} ms")
# print(f"Max latency: {times.max()*1000:.3f} ms")
# print("--------------------------------------------------")


# # Create a concrete function
# @tf.function
# def model_forward(x):
#     return model(x)

# # Define input shape (VERY IMPORTANT)
# input_shape = (1, *model.input_shape[1:])
# dummy_input = tf.random.normal(input_shape)

# # Get concrete function
# concrete_func = model_forward.get_concrete_function(dummy_input)

# # Convert to frozen graph
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# frozen_func = convert_variables_to_constants_v2(concrete_func)
# graph_def = frozen_func.graph.as_graph_def()

# # Profile
# from tensorflow.python.profiler.model_analyzer import profile
# from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# opts = ProfileOptionBuilder.float_operation()
# flops = profile(graph=frozen_func.graph, options=opts)

# print("Total FLOPs:", flops.total_float_ops)
