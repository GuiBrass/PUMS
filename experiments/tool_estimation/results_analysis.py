import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_error(model, y_scaler, X_test, y_test):
    pred_norm = model.predict(X_test)
    unscaled_y_pred = y_scaler.inverse_transform(pred_norm)
    mae_real = np.mean(np.abs(unscaled_y_pred - y_test), axis=0)
    print(mae_real)
    return mae_real

def plot_mae_mm_with_sensor_reduction_merged(files, exp, saveplotpath=None):
    dfs = [pd.read_csv(f) for f in files]


    x_labels = dfs[0]["missing_sensors"].values
    x_pos = np.arange(len(x_labels)) + 1 # 1, 2, 3, ...

    if exp == "pos":
        all_led = np.stack([df["mae_pos_LED"].values for df in dfs])
        all_pd = np.stack([df["mae_pos_PD"].values for df in dfs])
    else: 
        all_led = np.stack([df["mae_force_LED"].values for df in dfs])
        all_pd = np.stack([df["mae_force_PD"].values for df in dfs])

    led_data = [all_led[:, i] for i in range(all_led.shape[1])]
    pd_data = [all_pd[:, i] for i in range(all_pd.shape[1])]


    offset = 0.15
    width = 0.3


    plt.figure(figsize=(11, 6))


    plt.boxplot(
    led_data,
    positions=x_pos - offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='blue'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='+', markersize=6)
    )


    plt.boxplot(
    pd_data,
    positions=x_pos + offset,
    widths=width,
    showfliers=True,
    medianprops=dict(color='red'),
    boxprops=dict(color='green'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(marker='+', markersize=6)
    )


    led_mean = np.mean(all_led, axis=0)
    pd_mean = np.mean(all_pd, axis=0)


    plt.plot(x_pos - offset, led_mean, 'o-', color='blue', label='LED mean')
    plt.plot(x_pos + offset, pd_mean, 'o-', color='green', label='PD mean')


    plt.xlabel("Number of removed sensors")
    if exp == "pos":
        plt.ylabel("Position MAE (mm)")
        plt.title("Tool Estimation: Position MAE Distribution with Sensor Reduction")
    else:
        plt.ylabel("Force MAE (N)")
        plt.title("Tool Estimation: Force MAE Distribution with Sensor Reduction")
    
    plt.ylim(0.25, 0.8)
    plt.grid(True, axis='y')


    plt.xticks(x_pos, x_labels) # ← labels are decoupled
    plt.legend()
    plt.tight_layout()


    if saveplotpath is not None:
        plt.savefig(saveplotpath)

    plt.close()