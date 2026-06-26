import numpy as np
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import open3d as o3d
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def plot_train_val_loss(history, saveplotpath=None, show=True):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if saveplotpath is not None:
        plt.savefig(saveplotpath)
    if show == False:
        plt.close()
    else:
        plt.show()

def calculate_error(model, x, y_t, threshold_mm=0.0, z_max_mm=4.5):
    # Predict
    y_p = model.predict(x)
    y_p = np.squeeze(y_p)
    y_t = np.squeeze(y_t)

    # Convert to mm
    y_p_mm = y_p * z_max_mm
    y_t_mm = y_t * z_max_mm

    # Create mask (keep only values above threshold)
    mask = y_t_mm >= threshold_mm

    # Safety check
    if np.sum(mask) == 0:
        print("No values above threshold. Returning None.")
        return None, None, None

    # Apply mask
    y_p_mm_masked = y_p_mm[mask]
    y_t_mm_masked = y_t_mm[mask]

    # Compute errors in mm directly
    mae_mm = np.mean(np.abs(y_p_mm_masked - y_t_mm_masked))
    rmse_mm = np.sqrt(np.mean((y_p_mm_masked - y_t_mm_masked) ** 2))

    # Convert back to normalized if needed
    mae_normalized = mae_mm / z_max_mm

    print(f"Threshold applied: {threshold_mm} mm")
    print(f"Valid points kept: {np.sum(mask)} / {mask.size}")
    print(f"Mean Absolute Error (mm): {mae_mm:.6f} mm")
    print(f"Root Mean Square Error (mm): {rmse_mm:.6f} mm")

    return mae_normalized, mae_mm, rmse_mm

def upsample_heightmap(Z, scale=4, method="linear"):
    """
    Upsample a heightmap using interpolation.
    scale=4 turns 16x16 -> 64x64
    """
    H, W = Z.shape
    y = np.linspace(0, 1, H)
    x = np.linspace(0, 1, W)

    interp = RegularGridInterpolator((y, x), Z, method=method)

    y_new = np.linspace(0, 1, H * scale)
    x_new = np.linspace(0, 1, W * scale)
    Y_new, X_new = np.meshgrid(y_new, x_new, indexing="ij")

    Z_up = interp(np.stack([Y_new.flatten(), X_new.flatten()], axis=-1))
    return Z_up.reshape(H * scale, W * scale)

def smooth_heightmap(Z, sigma=1.0):
    """
    Smooth heightmap to remove grid artifacts.
    sigma in pixels (0.8–1.5 works best)
    """
    return gaussian_filter(Z, sigma=sigma)

def add_text_to_png(
    image_path,
    text,
    position=(20, 20),
    font_size=48
):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Optional background box for readability
    bbox = draw.textbbox((0, 0), text, font=font)
    x, y = position
    pad = 10

    draw.rectangle(
        [x - pad, y - pad, x + bbox[2] + pad, y + bbox[3] + pad],
        fill=(255, 255, 255)
    )

    draw.text(position, text, fill=(0, 0, 0), font=font)

    img.save(image_path)

def colormap_error_surface(error, global_max_error):
    # Normalize 0 → 1
    norm = error*4.5 / (global_max_error + 1e-8)

    # Use a matplotlib colormap (e.g., "inferno", "viridis", "jet", etc.)
    cmap = cm.get_cmap("plasma")

    colors = cmap(norm.flatten())[:, :3]  # remove alpha
    return colors

def heightmap_to_mesh(Z, x_scale=1.0, y_scale=1.0, z_scale=1.0, z_offset=0.0, color=None, per_vertex_colors=None):
    H, W = Z.shape

    Z = Z * z_scale + z_offset

    # Grille XY régulière
    xs = np.linspace(0, x_scale, W)
    ys = np.linspace(0, y_scale, H)
    X, Y = np.meshgrid(xs, ys)
    
    # Points 3D
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    
    # Triangulation : deux triangles par carré
    triangles = []
    for r in range(H - 1):
        for c in range(W - 1):
            i = r * W + c
            triangles.append([i, i+1, i+W])
            triangles.append([i+1, i+W+1, i+W])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

    mesh.compute_vertex_normals()

    if color is not None:
        mesh.paint_uniform_color(color)

    if per_vertex_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(per_vertex_colors)

    return mesh

def heightmap_to_solid_mesh(
    Z,
    x_scale=1.0,
    y_scale=1.0,
    z_scale=1.0,
    z_offset=0.0,
    z_base=0.0,
    color=None,
    per_vertex_colors=None
):
    """
    Convert a heightmap (H×W) into a solid, watertight 3D mesh.
    The side walls are now vertically subdivided to allow sharp height-based features.
    """
    H, W = Z.shape

    Z_top = Z * z_scale + z_offset
    Z_bottom = np.full_like(Z_top, z_base)

    xs = np.linspace(0, x_scale, W)
    ys = np.linspace(0, y_scale, H)
    X, Y = np.meshgrid(xs, ys)

    # ---- VERTICES ----
    top_vertices = np.stack((X, Y, Z_top), axis=-1).reshape(-1, 3)
    bottom_vertices = np.stack((X, Y, Z_bottom), axis=-1).reshape(-1, 3)

    all_vertices = list(np.vstack((top_vertices, bottom_vertices)))
    top_offset = 0
    bottom_offset = H * W

    triangles = []

    # ---- TOP SURFACE ----
    for r in range(H - 1):
        for c in range(W - 1):
            i = r * W + c
            triangles.append([top_offset + i, top_offset + i + 1, top_offset + i + W])
            triangles.append([top_offset + i + 1, top_offset + i + W + 1, top_offset + i + W])

    # ---- BOTTOM SURFACE ----
    for r in range(H - 1):
        for c in range(W - 1):
            i = r * W + c
            triangles.append([bottom_offset + i + W, bottom_offset + i + 1, bottom_offset + i])
            triangles.append([bottom_offset + i + W + 1, bottom_offset + i + 1, bottom_offset + i + W])


    def add_subdivided_wall(idx_top_a, idx_top_b, steps=20):
        v_top_a = all_vertices[idx_top_a]
        v_top_b = all_vertices[idx_top_b]
        v_bot_a = all_vertices[idx_top_a + bottom_offset]
        v_bot_b = all_vertices[idx_top_b + bottom_offset]
        
        # Create intermediate coordinate steps running vertically from top to bottom
        z_steps_a = np.linspace(v_top_a[2], v_bot_a[2], steps + 1)
        z_steps_b = np.linspace(v_top_b[2], v_bot_b[2], steps + 1)
        
        # Generate new wall vertices and track their indices
        wall_indices_a = [idx_top_a]
        wall_indices_b = [idx_top_b]
        
        # Add the middle vertices
        for step in range(1, steps):
            # Column A vertex
            all_vertices.append(np.array([v_top_a[0], v_top_a[1], z_steps_a[step]]))
            wall_indices_a.append(len(all_vertices) - 1)
            # Column B vertex
            all_vertices.append(np.array([v_top_b[0], v_top_b[1], z_steps_b[step]]))
            wall_indices_b.append(len(all_vertices) - 1)
            
        wall_indices_a.append(idx_top_a + bottom_offset)
        wall_indices_b.append(idx_top_b + bottom_offset)
        
        # Stitch the column arrays together with small paired triangles
        for step in range(steps):
            ta = wall_indices_a[step]
            tb = wall_indices_b[step]
            ba = wall_indices_a[step + 1]
            bb = wall_indices_b[step + 1]
            
            triangles.append([ta, tb, ba])
            triangles.append([tb, bb, ba])

    # Front (y = 0)
    for c in range(W - 1):
        add_subdivided_wall(c, c + 1)

    # Back (y = max)
    for c in range(W - 1):
        add_subdivided_wall((H - 1) * W + c, (H - 1) * W + c + 1)

    # Left (x = 0)
    for r in range(H - 1):
        add_subdivided_wall(r * W, (r + 1) * W)

    # Right (x = max)
    for r in range(H - 1):
        add_subdivided_wall(r * W + (W - 1), (r + 1) * W + (W - 1))

    # ---- BUILD MESH ----
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))

    # ---- NORMALS ----
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # ---- COLORS ----
    if color is not None:
        mesh.paint_uniform_color(color)

    if per_vertex_colors is not None:
        colors = np.vstack([per_vertex_colors, per_vertex_colors])
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh

def add_overlay_to_png(image_path, error_max):

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    width, height = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Side labels
    draw.text((170, height * 0.48), "Estimated", fill="black", font=font)
    draw.text((265, height * 0.65), "Target", fill="black", font=font)
    draw.text((340, height * 0.78), "Error", fill="black", font=font)

    # Color bar
    bar_width = 60
    bar_height = int(height * 0.5)

    x0 = width - 450
    y0 = int(height * 0.25)

    # Create vertical gradient
    gradient = np.linspace(0, 1, bar_height).reshape(-1, 1)
    cmap = plt.get_cmap("plasma_r")
    gradient_rgb = cmap(gradient)[:, :, :3]
    gradient_rgb = (gradient_rgb * 255).astype(np.uint8)

    gradient_img = Image.fromarray(gradient_rgb)
    gradient_img = gradient_img.resize((bar_width, bar_height))

    img.paste(gradient_img, (x0, y0))

    # Border
    draw.rectangle(
        [x0, y0, x0 + bar_width, y0 + bar_height],
        outline="black",
        width=3
    )

    # Labels
    draw.text((x0 + bar_width + 10, y0 - 10),
              f"{error_max:.2f} mm",
              fill="black",
              font=font_small)

    draw.text((x0 + bar_width + 10, y0 + bar_height - 20),
              "0 mm",
              fill="black",
              font=font_small)

    img.save(image_path)


def render_pred_screenshot(pred, save_path, close_images):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=not close_images)

    vis.add_geometry(pred)

    render_option = vis.get_render_option()
    render_option.light_on = True

    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic.intrinsic_matrix.copy()
    intrinsic[0, 0] = intrinsic[1, 1] = 1e6

    params.intrinsic.intrinsic_matrix = intrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

    ctr.set_lookat([40.0, 52.0, 10.0])
    # ctr.set_front([-1.9, 1.0, 1.1])
    ctr.set_front([-1, 0.02, 0.3])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

def render_screenshot(geometries, save_path, close_images, error_max=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=not close_images)

    for g in geometries:
        vis.add_geometry(g)

    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = params.intrinsic.intrinsic_matrix.copy()
    intrinsic[0, 0] = intrinsic[1, 1] = 1e6

    params.intrinsic.intrinsic_matrix = intrinsic
    ctr.convert_from_pinhole_camera_parameters(params)

    ctr.set_lookat([40.0, 50.0, 10.0])
    ctr.set_front([-1.9, 1.0, 1.1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()

    if error_max is not None:
        add_overlay_to_png(save_path, error_max)

def plot_surfaces(num_samples_to_show, model, X_test, y_test, inputs_min, inputs_max, saveplotpath=None, unseen=False, close_images=True, fstl=None):
    true_mm_all = y_test[0:4] * 4.5  # since z_max_mm = 4.5
    pred_mm_all = np.squeeze(model.predict(X_test[0:4])) * 4.5
    error_mm_all = np.abs(pred_mm_all - true_mm_all)

    global_height_max = np.max(true_mm_all)
    global_error_max = np.max(error_mm_all)
    for i in range(num_samples_to_show):
        pred = np.squeeze(model.predict(X_test[i:i+1])[0, :, :])
        true = y_test[i, :, :]
        inp = X_test[i, :, :]

        plt.figure(figsize=(6, 6))
        plt.imshow(inp, cmap='viridis', origin='lower', vmin=inputs_min, vmax=inputs_max)
        # plt.title("Input (LED × Photodetectors)")

        # Axis labels and ticks
        plt.xlabel("Photodetectors", fontsize=20)
        plt.ylabel("LEDs", fontsize=20)

        # Tick positions and labels
        # plt.xticks(np.arange(16), [f"PD{i+1}" for i in range(16)], rotation=45, ha='right', fontsize=10)
        # plt.yticks(np.arange(16), [f"LED{i+1}" for i in range(16)], fontsize=10)
        plt.xticks([])
        plt.yticks([])

        # cbar = plt.colorbar(label="Signal Intensity", shrink=0.75, pad=0.05)
        # cbar.ax.tick_params(labelsize=10)
        # cbar.set_label("Signal Intensity", fontsize=10)
        plt.tight_layout()
        if saveplotpath is not None and i==1 and unseen==False and fstl==None:
            plt.savefig(saveplotpath+"/figure_3_input_matrix_PDs_LEDS/input_matrix")
        if close_images == True:
            plt.close()
        else:
            plt.show()
        

        # Physical dimensions (in mm)
        x_len_mm = 30.0
        y_len_mm = 120.0
        z_max_mm = 4.5

        # Scale axes to real dimensions
        x = np.linspace(0, x_len_mm, pred.shape[1])
        y = np.linspace(0, y_len_mm, pred.shape[0])
        X, Y = np.meshgrid(x, y)

        # Convert normalized Z values to mm
        pred_mm = pred * z_max_mm
        true_mm = true * z_max_mm
        error_mm = np.abs(pred_mm - true_mm)

        # Compute global Z range (used for 3D spacing only)
        z_min = min(pred_mm.min(), true_mm.min(), error_mm.min())
        z_max = max(pred_mm.max(), true_mm.max(), error_mm.max())
        z_range = z_max - z_min
        offset_step = 0.15 * z_range

        z_offset_pred = 40
        z_offset_true = 18
        z_offset_error = 0

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # ---- SURFACES ----
        # Make sure they use global vmin/vmax
        surf_pred = ax.plot_surface(
            X, Y, pred_mm + z_offset_pred,
            cmap='viridis', linewidth=0, antialiased=False, alpha=0.9
        )

        surf_true = ax.plot_surface(
            X, Y, true_mm + z_offset_true,
            cmap='viridis', linewidth=0, antialiased=False, alpha=0.9
        )

        Z_flat = np.full_like(error_mm, z_offset_error)

        surf_err = ax.plot_surface(
            X, Y, Z_flat,
            facecolors=plt.cm.coolwarm(error_mm / (global_error_max + 1e-8)),
            linewidth=0,
            antialiased=False,
            shade=False
        )

        # ---- AXIS SETTINGS ----
        ax.set_zlim(z_min, z_max + 3 * offset_step)
        ax.set_box_aspect((x_len_mm, y_len_mm, z_max_mm))
        ax.set_xlabel("X (mm)", labelpad=15)
        ax.set_ylabel("Y (mm)", labelpad=60)
        ax.set_zlabel("Height (mm)")
        ax.set_zticklabels([])

        ax.text(45, 135, z_offset_pred, "Predicted", color='black', fontsize=15)
        ax.text(40, 130, z_offset_true, "Target", color='black', fontsize=15)
        ax.text(38, 128, 0, "Error", color='black', fontsize=15)

        # ---- COLORBARS ----
        mappable_pred = plt.cm.ScalarMappable(cmap='viridis')
        mappable_pred.set_clim(0, global_height_max)
        cbar1 = fig.colorbar(mappable_pred, ax=ax, shrink=0.5, aspect=10, pad=0.01)
        cbar1.set_label('Height (mm)', fontsize=12)

        mappable_err = plt.cm.ScalarMappable(cmap='coolwarm')
        mappable_err.set_clim(0, global_error_max)
        cbar2 = fig.colorbar(mappable_err, ax=ax, shrink=0.5, aspect=10, pad=0.02)
        cbar2.set_label('Error (mm)', fontsize=12)

        ax.view_init(elev=20, azim=140)
        if saveplotpath is not None:
            if unseen==False:
                plt.savefig(saveplotpath+f"/figure_4_surface_estimation/surf_test_{i}")
            elif fstl is not None:
                plt.savefig(saveplotpath+f"/figure_8_increase_shots_transfer_learning/surf_unseen_test_n{fstl}_{i}")
            else:
                plt.savefig(saveplotpath+f"/figure_4_surface_estimation/surf_unseen_{i}")
        if close_images == True:
            plt.close()
        else:
            plt.show()

        error = np.abs(true - pred)
        error_colors = colormap_error_surface(error, global_error_max)

        MAE_in_mm = np.mean(error)*4.5
        print("MAE: ", MAE_in_mm, "ex:", i)

        UPSCALE = 4

        true_hi  = smooth_heightmap(upsample_heightmap(true, UPSCALE),  sigma=1.0)
        pred_hi  = smooth_heightmap(upsample_heightmap(pred, UPSCALE),  sigma=1.0)
        error_hi = smooth_heightmap(upsample_heightmap(error, UPSCALE), sigma=0.8)

        error_colors_hi = colormap_error_surface(error_hi, global_error_max)

        mesh_target = heightmap_to_solid_mesh(true_hi, x_scale=30, y_scale=120, z_scale=4.5, z_offset=20, z_base=20, color=[0.72, 0.78, 0.85])

        mesh_pred = heightmap_to_solid_mesh(
            pred_hi, x_scale=30, y_scale=120, z_scale=4.5,
            z_offset=40, z_base=40, color=[0.85, 0.65, 0.40]
        )

        flat_error = np.zeros_like(error_hi)

        mesh_error = heightmap_to_mesh(
            flat_error, x_scale=30, y_scale=120, z_scale=4.5,
            z_offset=0, per_vertex_colors=error_colors_hi
        )
        mesh_target.compute_triangle_normals()
        mesh_target.compute_vertex_normals()
        mesh_pred.compute_triangle_normals()
        mesh_pred.compute_vertex_normals()
        mesh_error.compute_triangle_normals()
        mesh_error.compute_vertex_normals()

        if saveplotpath is not None:
            if unseen==False:
                render_screenshot(
                    [mesh_target, mesh_pred, mesh_error],
                    saveplotpath+f"/figure_4_surface_estimation/o3d_test_surfs_{i}.png",
                    close_images,
                    error_max=global_error_max
                )
            elif fstl is not None:
                render_screenshot(
                    [mesh_target, mesh_pred, mesh_error],
                    saveplotpath+f"/figure_8_increase_shots_transfer_learning/o3d_unseen_surfs_n{fstl}_{i}.png",
                    close_images,
                    error_max=global_error_max
                )
                add_text_to_png(
                    saveplotpath+f"/figure_8_increase_shots_transfer_learning/o3d_unseen_surfs_n{fstl}_{i}.png",
                    f"MAE = {MAE_in_mm:.2f} mm"
                )
            else:
                render_screenshot(
                    [mesh_target, mesh_pred, mesh_error],
                    saveplotpath+f"/figure_4_surface_estimation/o3d_unseen_surfs_{i}.png",
                    close_images,
                    error_max=global_error_max
                )


def plot_MAE_mm_over_nbsamples(csvfile, save_path):

    # Load CSV file
    df = pd.read_csv(csvfile)

    # Plot nb_samples vs MAE_mm
    plt.figure(figsize=(8, 5))
    plt.scatter(df["nb_samples"], df["mae_mm"])
    plt.xlabel("Number of Samples")
    plt.ylabel("MAE (mm)")
    plt.title("Number of Samples vs MAE (mm)")
    plt.savefig(save_path)
    plt.show()

def plot_mae_mm_with_sensor_reduction_merged(files, saveplotpath=None):

    dfs = [pd.read_csv(f) for f in files]

    max_sensors_removed = 15
    x_ticks = np.arange(0, max_sensors_removed + 1)  # 0, 1, 2, ..., 15

    all_led = np.stack([df["mae_mm_LED"].values for df in dfs])[:, :max_sensors_removed + 1]
    all_pd = np.stack([df["mae_mm_PD"].values for df in dfs])[:, :max_sensors_removed + 1]
    
    led_data = [all_led[:, i] for i in range(all_led.shape[1])]
    pd_data = [all_pd[:, i] for i in range(all_pd.shape[1])]

    max_both_pairs = max_sensors_removed // 2  # 15 // 2 = 7 pairs (14 sensors)
    
    all_both = np.stack([df["mae_mm_both"].values for df in dfs])[:, :max_both_pairs + 1]
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
        medianprops=dict(color='red'),
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
        medianprops=dict(color='red'),
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
        medianprops=dict(color='red'),
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
    
    # Both line only connects the points at 0, 2, 4, ..., 14
    plt.plot(x_both_positions + offset, both_mean, '^-', color='purple', label='Both mean (half LED / half PD)')

    plt.xlabel("Total Number of Removed Sensors", fontsize=14)
    plt.ylabel("MAE (mm)", fontsize=14)
    plt.grid(True, axis='both', alpha=0.3)

    plt.xticks(ticks=x_ticks, labels=[str(x) for x in x_ticks], fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-0.6, max_sensors_removed + 0.6)

    # plt.legend()
    plt.tight_layout()

    if saveplotpath is not None:
        plt.savefig(saveplotpath)

    plt.show()


def plot_training_set_reduction(past_files, saveplotpath, show=False):

    N_REPEATS = len(past_files)

    all_mae_mm = []

    for file in past_files:
        df = pd.read_csv(file)
        all_mae_mm.append(df["mae_mm"].values)

    all_mae_mm = np.array(all_mae_mm)  

    N_REDUCTION = all_mae_mm.shape[1]
    exp_values = np.arange(N_REDUCTION) * 10

    plt.figure(figsize=(10,6))

    box_data = [all_mae_mm[:, r] for r in range(N_REDUCTION)]

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

    # plt.title("Test Error with Reduced Training Set (k=10)")
    plt.xlabel("Training Data Removed (%)", fontsize=14)
    plt.ylabel("Test MAE (mm)", fontsize=14)
    plt.xticks(exp_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{saveplotpath}_mae_mm_boxplot.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

def apply_height_bands_by_face(mesh, band_height=0.4):
    """
    Duplicates vertices so each face has its own independent vertices.
    This allows us to use vertex colors to get perfectly sharp face-colored bands.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    unrolled_vertices = vertices[triangles.flatten()]

    num_triangles = len(triangles)
    unrolled_triangles = np.arange(num_triangles * 3).reshape(-1, 3)
    
    face_z_centers = unrolled_vertices[:, 2].reshape(-1, 3).mean(axis=1)
    
    colors = [
        [0.3, 0.3, 0.3],  # Color A: Dark Grey
        [0.8, 0.8, 0.8],  # Color B: Light Grey
    ]
    
    band_indices = np.floor((face_z_centers + 1e-5) / band_height).astype(int)
    
    face_colors = np.zeros((num_triangles, 3))
    for i in range(len(colors)):
        face_colors[band_indices % len(colors) == i] = colors[i]
        
    vertex_colors = np.repeat(face_colors, 3, axis=0)
    
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(unrolled_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(unrolled_triangles)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return new_mesh

def save_as_onshape_step_compatible(mesh, base_path, band_height=0.4):
    """
    Saves a mesh by separating it into two distinct geometry collections 
    (Dark and Light layers) and saving them as a multi-body OBJ.
    When imported into Onshape as a composite part, they form distinct selectable bodies.
    """
    obj_path = base_path + ".obj"
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    
    # Identify which faces belong to the dark bands using vertex colors
    face_colors = vertex_colors[triangles[:, 0]]
    is_dark_face = face_colors[:, 0] < 0.5
    
    # Separate the triangles into two groups
    dark_triangles = triangles[is_dark_face]
    light_triangles = triangles[~is_dark_face]
    
    # Write as a multi-object OBJ file (Onshape translates separate 'o' tags as distinct parts)
    with open(obj_path, "w") as f:
        f.write("# Multi-body mesh for Onshape import\n")
        
        # Write all shared vertices first
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Group 1: Dark Bands
        f.write("\no Dark_Bands_Body\n")
        for t in dark_triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
            
        # Group 2: Light Bands
        f.write("\no Light_Bands_Body\n")
        for t in light_triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
            
    print(f"Generated Onshape-ready multi-body mesh: {obj_path}")

def plot_pred_surface(model, X_test, y_test, saveplotpath=None, surf="pred", close_images=True):
    UPSCALE = 4
    if surf == 'pred':
        pred = np.squeeze(model.predict(X_test[1:2])[0])
        pred_hi  = smooth_heightmap(upsample_heightmap(pred, UPSCALE),  sigma=1.0)
        mesh_pred = heightmap_to_solid_mesh(
                pred_hi, x_scale=30, y_scale=120, z_scale=4.5,
                z_offset=0, z_base=0, color=[0.85, 0.65, 0.40]
            )
        mesh_pred = apply_height_bands_by_face(mesh_pred, band_height=0.4)
        mesh_pred.compute_triangle_normals()
        mesh_pred.compute_vertex_normals()

        if saveplotpath is not None:
            cad_output_dir = os.path.join(saveplotpath, "figure_4_surface_estimation")
            os.makedirs(cad_output_dir, exist_ok=True)
            
            # Use the new explicit companion exporter
            base_filename = os.path.join(cad_output_dir, "mesh_pred_sample")
            save_as_onshape_step_compatible(mesh_pred, base_filename)

        render_pred_screenshot(
                    mesh_pred,
                    saveplotpath+f"/figure_4_surface_estimation/mesh_pred_sample.png",
                    close_images
                )
    elif surf == 'target':
        true = y_test[1, :, :]
        true_hi  = smooth_heightmap(upsample_heightmap(true, UPSCALE),  sigma=1.0)
        mesh_target = heightmap_to_solid_mesh(
                true_hi, x_scale=30, y_scale=120, z_scale=4.5,
                z_offset=0, z_base=0, color=[0.85, 0.65, 0.40]
            )
        mesh_target = apply_height_bands_by_face(mesh_target, band_height=0.4)
        mesh_target.compute_triangle_normals()
        mesh_target.compute_vertex_normals()

        if saveplotpath is not None:
            cad_output_dir = os.path.join(saveplotpath, "figure_4_surface_estimation")
            os.makedirs(cad_output_dir, exist_ok=True)
            
            # Use the new explicit companion exporter
            base_filename = os.path.join(cad_output_dir, "mesh_target_sample")
            save_as_onshape_step_compatible(mesh_target, base_filename)

        render_pred_screenshot(
                    mesh_target,
                    saveplotpath+f"/figure_4_surface_estimation/mesh_target_sample.png",
                    close_images
                )


def plot_pca_component_removal(past_files, saveplotpath, show=False):
    dfs = [pd.read_csv(f) for f in past_files]

    removed_pcs = dfs[0]["removed_pcs"].values

    box_data = []
    for n_pc in removed_pcs:
        values = []
        for df in dfs:
            row = df[df["removed_pcs"] == n_pc]
            if not row.empty:
                values.append(row["mae_mm"].iloc[0])
        box_data.append(values)

    # 1. Calculate remaining PCs
    remaining_pcs = 256 - removed_pcs
    
    # 2. Use the index/order of the elements as the X-positions to keep spacing uniform
    positions = np.arange(len(remaining_pcs))
    
    plt.figure(figsize=(10, 6))

    plt.boxplot(
        box_data,
        positions=positions,  # Uniform structural spacing
        widths=0.35,
        showfliers=True,
        medianprops=dict(color='red'),
        boxprops=dict(color='blue'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(
            marker='o',
            markersize=6
        )
    )

    data_mean = [np.mean(v) for v in box_data]

    plt.plot(
        positions,
        data_mean,
        '^-',
        color='blue',
        # label='Mean'
    )

    # 3. FIX: Match the positions exactly with your desired string labels
    plt.xlabel("Number of Principal Components", fontsize=14)
    plt.ylabel("Test MAE (mm)", fontsize=14)
    
    # Using remaining_pcs as labels so the user sees "256, 192, ..., 2"
    plt.xticks(ticks=positions, labels=remaining_pcs.astype(int), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{saveplotpath}_mae_mm_boxplot.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()