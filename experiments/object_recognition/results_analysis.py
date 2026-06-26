import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_train_val_loss(history, saveplotpath=None, show=True):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if saveplotpath is not None:
        plt.savefig(saveplotpath)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_CM(model, X_test, y_test, saveplotpath=None, show=True):
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Class names
    class_names = ["empty", "cube", "screw driver", "tennis", "can",
                   "cylinder", "pyramid", "cup", "puck"]
    num_classes = len(class_names)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes, labels=np.arange(num_classes))

    # Plot heatmap (same style as second function)
    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if saveplotpath is not None:
        plt.savefig(saveplotpath)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_sensor_reduction(files, saveplotpath=None):
    dfs = [pd.read_csv(f) for f in files]

    max_sensors_removed = 15
    x_ticks = np.arange(0, max_sensors_removed + 1)  # [0, 1, 2, ..., 15]

    all_led = (1 - np.stack([df["test_accuracy_LEDs"].values for df in dfs]))[:, :max_sensors_removed + 1] * 100
    all_pd = (1 - np.stack([df["test_accuracy_PDs"].values for df in dfs]))[:, :max_sensors_removed + 1] * 100

    led_data = [all_led[:, i] for i in range(all_led.shape[1])]
    pd_data = [all_pd[:, i] for i in range(all_pd.shape[1])]

    max_both_pairs = max_sensors_removed // 2  # 15 // 2 = 7 pairs (14 sensors)
    
    all_both = (1 - np.stack([df["test_accuracy_both"].values for df in dfs]))[:, :max_both_pairs + 1] * 100
    both_data = [all_both[:, i] for i in range(all_both.shape[1])]
    
    x_both_positions = np.arange(0, max_both_pairs + 1) * 2  

    offset = 0.23
    width = 0.2

    plt.figure(figsize=(11, 6))

    plt.boxplot(
        led_data,
        positions=np.arange(len(led_data)) - offset,
        widths=width,
        showfliers=True,
        medianprops=dict(color='green'),
        boxprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

    plt.boxplot(
        pd_data,
        positions=np.arange(len(pd_data)),
        widths=width,
        showfliers=True,
        medianprops=dict(color='green'),
        boxprops=dict(color='green'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

    plt.boxplot(
        both_data,
        positions=x_both_positions + offset,
        widths=width,
        showfliers=True,
        medianprops=dict(color='purple'),
        boxprops=dict(color='purple'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=6)
    )

    led_mean = np.mean(all_led, axis=0)
    pd_mean = np.mean(all_pd, axis=0)
    both_mean = np.mean(all_both, axis=0)

    plt.plot(np.arange(len(led_data)) - offset, led_mean, '^-', color='blue', label='LED mean')
    plt.plot(np.arange(len(pd_data)), pd_mean, '^-', color='green', label='PD mean')
    plt.plot(x_both_positions + offset, both_mean, '^-', color='purple', label='Both mean (half LED / half PD)')

    plt.xlabel("Total Number of Removed Sensors", fontsize=14)
    plt.ylabel("Error Rate (%)", fontsize=14)
    plt.grid(True, axis='both', alpha=0.3)

    # Force exact integer values for the x-ticks
    plt.xticks(ticks=x_ticks, labels=[str(x) for x in x_ticks], fontsize=12)
    plt.xlim(-0.6, max_sensors_removed + 0.6)
    plt.yticks(fontsize=12)

    # plt.legend()
    plt.tight_layout()

    if saveplotpath is not None:
        plt.savefig(saveplotpath)

    plt.show()