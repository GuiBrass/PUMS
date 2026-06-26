import ast
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers

from data_preprocessing import extract_csv, get_files, to_onehot
from model_builder import create_cnn_enc_dec_model, build_transformer_4classifiers, ViT_on_SE_for_class, MeanValAccuracyCallback
from results_analysis import plot_conf_matrix, compute_accuracy, plot_conf_matrix_with_splits, convert_ids_to_obj_angle

#### MODEL NAME ####
model_name = "SC_35classes_HViT_cold_start"   # model named used for training and compile the results.


#Get dataset
folder_path = "./nogit_datasets/dataset_surface_estimation_classification/dataset_06-10-25"

csv_files_calib, csv_files_inputs = get_files(folder_path)
calib = extract_csv(csv_files_calib)
calib = np.array(calib)
inputs = extract_csv(csv_files_inputs)
inputs = np.array(inputs)
col_means = inputs.mean(axis=1, keepdims=True)  # shape (N, 1, PDs)
inputs = inputs - col_means

df = pd.read_csv(folder_path+"/info_placement_2025-10-06_15-05-54.csv")

y1_list, y2_list, y3_list, y4_list = [], [], [], []

dict_obj_ids = {1:{0:1, 90:2, 180:3, 270:4},
                2:{0:5, 90:6, 180:7, 270:8},
                3:{0:9, 90:10, 180:9, 270:10},
                4:{0:11, 90:12, 180:13, 270:14},
                5:{0:15, 90:16, 180:17, 270:18},
                6:{0:19, 90:19, 180:19, 270:19},
                7:{0:20, 90:21, 180:22, 270:23},
                8:{0:24, 90:25, 180:24, 270:25},
                9:{0:26, 90:26, 180:26, 270:26},
                10:{0:27, 90:27, 180:27, 270:27},
                11:{0:28, 90:29, 180:30, 270:31},
                12:{0:32, 90:33, 180:34, 270:35},
                }

inv_dict_obj_ids = {}
for obj_id, angles_dict in dict_obj_ids.items():
    for angle, class_id in angles_dict.items():
        inv_dict_obj_ids[class_id] = [obj_id, angle]


for _, row in df.iterrows():

    # Extract the 4 objects from row (each cell like "[id, angle]")
    objects = []
    for cell in row:
        obj = ast.literal_eval(cell)  # → [id, angle]   
        objects.append(dict_obj_ids[obj[0]][obj[1]])     # keep only id 

    # Safety check
    if len(objects) != 4:
        raise ValueError(f"Expected 4 objects per row, got {len(objects)}")

    # Convert each object id to one-hot
    y1_list.append(to_onehot(objects[0], num_classes=35))
    y2_list.append(to_onehot(objects[1], num_classes=35))
    y3_list.append(to_onehot(objects[2], num_classes=35))
    y4_list.append(to_onehot(objects[3], num_classes=35))

# Convert to numpy arrays
y1 = np.array(y1_list)
y2 = np.array(y2_list)
y3 = np.array(y3_list)
y4 = np.array(y4_list)

print("Shapes:")
print("y1:", y1.shape)
print("y2:", y2.shape)
print("y3:", y3.shape)
print("y4:", y4.shape)

surface_tuples = []

for i in range(len(y1_list)):
    # Convert one-hot back to class index (+1 if your classes start at 1)
    c1 = np.argmax(y1_list[i]) + 1
    c2 = np.argmax(y2_list[i]) + 1
    c3 = np.argmax(y3_list[i]) + 1
    c4 = np.argmax(y4_list[i]) + 1

    surface_tuples.append((c1, c2, c3, c4))

surface_tuples = np.array(surface_tuples)

# Count duplicates
unique_surfaces, counts = np.unique(surface_tuples, axis=0, return_counts=True)

duplicate_mask = counts > 1
num_duplicates = np.sum(counts[duplicate_mask] - 1)

print("=====================================")
print(f"Total samples: {len(surface_tuples)}")
print(f"Unique surfaces: {len(unique_surfaces)}")
print(f"Number of duplicated surfaces: {num_duplicates}")

if np.any(duplicate_mask):
    print("\nDuplicated surface configurations:")
    for surf, count in zip(unique_surfaces[duplicate_mask], counts[duplicate_mask]):
        print(f"{tuple(surf)} appears {count} times")
else:
    print("No duplicated surfaces found.")
print("=====================================")

indices = np.arange(len(inputs))
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(kf.split(inputs)):
    # Make model dir
    models_folder = f'./nogit_NN/surface_classification/fold_{fold}'
    os.makedirs(models_folder, exist_ok=True)

    # Make results dir
    results_folder = f'./results/surface_classification/fold_{fold}'
    os.makedirs(results_folder, exist_ok=True)

    X_train_full = inputs[train_idx]
    X_test = inputs[test_idx]

    y1_train_full = y1[train_idx]
    y1_test = y1[test_idx]

    y2_train_full = y2[train_idx]
    y2_test = y2[test_idx]

    y3_train_full = y3[train_idx]
    y3_test = y3[test_idx]

    y4_train_full = y4[train_idx]
    y4_test = y4[test_idx]

    (
    X_train,
    X_val,
    y1_train,
    y1_val,
    y2_train,
    y2_val,
    y3_train,
    y3_val,
    y4_train,
    y4_val,
    ) = train_test_split(
        X_train_full,
        y1_train_full, y2_train_full, y3_train_full, y4_train_full,
        test_size=0.15,
        random_state=1,
        shuffle=True
    )
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]


    input_shape = (16, 16, 1)
    num_shapes = 12  # total possible shapes

    l2_reg = regularizers.l2(1e-4)

    checkpoint_cb = ModelCheckpoint(
        filepath=f'{models_folder}/{model_name}.keras',  # You can change the path
        monitor='val_obj1_accuracy',
        mode="max",
        save_best_only=True,
        save_weights_only=False  # Set to True if you only want weights
    )

    surface_model = load_model("./nogit_NN/surface_estimation/SE_Transformer_CNN_hybrid_mean_PD_removed.keras", compile=False)

    model = ViT_on_SE_for_class(surface_model, NUM_CLASSES=35, trainable_body=True, cold_start=True)
    # model = build_transformer_4classifiers(num_classes=35)

    model.summary()

    # === Train ===
    # history = model.fit(
    #     X_train,
    #     {"obj1": y1_train, "obj2": y2_train, "obj3": y3_train, "obj4": y4_train},
    #     validation_data=(X_val, {"obj1": y1_val, "obj2": y2_val, "obj3": y3_val, "obj4": y4_val}),
    #     epochs=100,
    #     batch_size=32,
    #     verbose=0,
    #     callbacks=[
    #         checkpoint_cb,
    #         MeanValAccuracyCallback(),
    #     ]
    # )

    model = load_model(f'{models_folder}/{model_name}.keras')

    pred1, pred2, pred3, pred4 = model.predict(X_test, batch_size=32)
    preds = np.concatenate([pred1, pred2, pred3, pred4])

    y_test = np.concatenate([y1_test, y2_test, y3_test, y4_test])
    # Compute accuracy per head
    acc1 = compute_accuracy(y1_test, pred1)
    acc2 = compute_accuracy(y2_test, pred2)
    acc3 = compute_accuracy(y3_test, pred3)
    acc4 = compute_accuracy(y4_test, pred4)
    glob_acc = compute_accuracy(y_test, preds)

    print(f"Test accuracy obj1: {acc1:.4f}")
    print(f"Test accuracy obj2: {acc2:.4f}")
    print(f"Test accuracy obj3: {acc3:.4f}")
    print(f"Test accuracy obj4: {acc4:.4f}")
    print(f"Global accuracy: {glob_acc:.4f}")

    y1_true_cls = np.argmax(y1_test, axis=1)
    y2_true_cls = np.argmax(y2_test, axis=1)
    y3_true_cls = np.argmax(y3_test, axis=1)
    y4_true_cls = np.argmax(y4_test, axis=1)

    y1_pred_cls = np.argmax(pred1, axis=1)
    y2_pred_cls = np.argmax(pred2, axis=1)
    y3_pred_cls = np.argmax(pred3, axis=1)
    y4_pred_cls = np.argmax(pred4, axis=1)

    # Concatenate the 4 heads (same as your global accuracy)
    y_true_fold = np.concatenate([y1_true_cls, y2_true_cls, y3_true_cls, y4_true_cls])
    y_pred_fold = np.concatenate([y1_pred_cls, y2_pred_cls, y3_pred_cls, y4_pred_cls])

    # Store across folds
    all_y_true.append(y_true_fold)
    all_y_pred.append(y_pred_fold)

all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

# Plot confusion matrix
plot_conf_matrix_with_splits(
    all_y_true,
    all_y_pred,
    title="Global Confusion Matrixs with 35 Classes",
    dict_obj_ids=dict_obj_ids,
    saveplotpath=f'./results/surface_classification/global_confusion_matrix'
)
