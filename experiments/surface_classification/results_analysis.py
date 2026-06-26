import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_conf_matrix(true, pred, title, num_classes):
    cm = confusion_matrix(true, pred, labels=range(0,num_classes))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=range(0, num_classes),
        yticklabels=range(0, num_classes))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_conf_matrix_with_splits(
    true,
    pred,
    title,
    dict_obj_ids,
    saveplotpath,
    axis_title_size=20,
    tick_label_size=16,
    secondary_tick_size=16,
    annot_size=12,
):
    axis_angles = [0,90,180,270,
                  0,90,180,270,
                  0,90,
                  0,90,180,270,
                  0,90,180,270,
                  0,
                  0,90,180,270,
                  0,90,
                  0,
                  0,
                  0,90,180,270,
                  0,90,180,270]

    class_groups = []
    real_object_numbers = []

    for obj_id in dict_obj_ids:
        unique_classes = sorted(set(dict_obj_ids[obj_id].values()))
        class_groups.append(unique_classes)
        real_object_numbers.append(obj_id)

    boundaries = []
    c = 0
    for group in class_groups:
        c += len(group)
        boundaries.append(c)

    num_classes = boundaries[-1]

    # ---- Build confusion matrix ----
    cm = confusion_matrix(true, pred, labels=range(0, num_classes))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=axis_angles,
        yticklabels=axis_angles,
        cbar=True,
        ax=ax,
        annot_kws={"size": annot_size},
    )

    for b in boundaries[:-1]:
        ax.axvline(b, linestyle="--", color="gray", linewidth=0.8)
        ax.axhline(b, linestyle="--", color="gray", linewidth=0.8)

    # Main axis labels
    ax.set_xlabel(
        "Predicted Orientation (°)",
        fontsize=axis_title_size,
        labelpad=25,
    )
    ax.set_ylabel(
        "True Orientation (°)",
        fontsize=axis_title_size,
        labelpad=25,
    )

    # Main axis tick labels
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    secax_x = ax.secondary_xaxis("bottom")
    secax_x.spines["bottom"].set_position(("outward", 40))

    secax_y = ax.secondary_yaxis("left")
    secax_y.spines["left"].set_position(("outward", 40))

    centers = []
    prev = 0
    for b in boundaries:
        block_size = b - prev
        if block_size % 2 == 0:
            center = prev + block_size // 2
        else:
            center = prev + block_size // 2 + 0.5
        centers.append(center)
        prev = b

    secax_x.set_xticks(centers)
    secax_y.set_yticks(centers)

    secax_x.set_xticklabels(real_object_numbers)
    secax_y.set_yticklabels(real_object_numbers)

    # Secondary axis tick labels
    secax_x.tick_params(axis="x", labelsize=secondary_tick_size)
    secax_y.tick_params(axis="y", labelsize=secondary_tick_size)

    # Optional secondary axis titles
    secax_x.set_xlabel(
        "Predicted Object ID",
        fontsize=axis_title_size,
        labelpad=25,
    )
    secax_y.set_ylabel(
        "True Object ID",
        fontsize=axis_title_size,
        labelpad=20,
    )

    plt.tight_layout()

    if saveplotpath is not None:
        plt.savefig(saveplotpath, bbox_inches="tight")

    plt.show()



def compute_accuracy(y_true_onehot, y_pred_probs,):
    y_true_cls = np.argmax(y_true_onehot, axis=1)
    y_pred_cls = np.argmax(y_pred_probs, axis=1)
    acc = np.mean(y_true_cls == y_pred_cls)
    return acc

def convert_ids_to_obj_angle(obj_preds, inv_dict, filename):
    """
    obj_preds: list of arrays, e.g. [pred1_cls, pred2_cls, pred3_cls, pred4_cls]
    inv_dict: class_id -> (object_id, angle)
    """
    converted_preds = []
    for i in range(len(obj_preds[0])):  # iterate over samples
        sample = []
        for head_preds in obj_preds:   # iterate over the 4 heads
            cls_idx = head_preds[i]
            class_id = int(cls_idx) + 1  # shift from 0-based index → class_id
            sample.append(list(inv_dict[class_id]))  # convert tuple → list
        converted_preds.append(sample)
    
    # Save as CSV
    df_pred = pd.DataFrame(converted_preds, columns=["0", "1", "2", "3"])
    df_pred.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")
