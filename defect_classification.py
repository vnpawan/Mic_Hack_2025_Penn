#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hyperspy.api as hs
import cv2
from scipy import ndimage

import time

# Record the start time
start_time = time.perf_counter()

# ----------------------------
# Paths
# ----------------------------
csv_root = Path("/Users/cychen/Downloads/atom-analysis_Thursday")
dm3_root = Path("/Users/cychen/Documents/QuaternaryAlloyMoWSSe")
output_root = Path("/Users/cychen/Documents/classification_Thursday_25-40-40")
output_root.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Parameters (adjust if needed)
# ----------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
GAUSSIAN_BLUR_SIGMA = 1.0

# Edge filtering (tune if needed)
EDGE_MARGIN_MULT_A = 1.0   # margin = 3 * a_px (in pixels)
EDGE_MARGIN_MIN_PX = 10    # at least 10 px

# Manual Calibration Overrides (filename without extension : calibration in nm/px)
MANUAL_CALIBRATIONS = {
    '3D Stack1': 0.03125,
    '3D Stack2': 0.0156,
    '3D Stack3': 0.0156,
    '3D Stack4': 0.0156,
    '3D Stack5': 0.0156,
    '3D Stack6': 0.0156,
    '3D Stack7': 0.0156,
    '3D Stack8': 0.0156,
    '3D Stack9': 0.0156,
    '3D Stack11': 0.0156,
    '3D Stack12': 0.0156,
    '3D Stack13': 0.0156
}

def enhance_image(image):
    """Apply CLAHE and Gaussian blur to enhance image (returns float image in [0,1])."""
    image = np.nan_to_num(image)

    img_min = float(np.min(image))
    img_max = float(np.max(image))
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)

    img_norm = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )
    img_clahe = clahe.apply(img_norm)
    img_blur = ndimage.gaussian_filter(img_clahe.astype(np.float32), GAUSSIAN_BLUR_SIGMA)

    bmin = float(np.min(img_blur))
    bmax = float(np.max(img_blur))
    if bmax == bmin:
        return np.zeros_like(img_blur, dtype=np.float32)

    return (img_blur - bmin) / (bmax - bmin)

def get_calibration_nm_per_px(data_obj, filename_no_ext: str) -> float:
    # Try HyperSpy’s signal axis first (more robust)
    try:
        cal = data_obj.axes_manager.signal_axes[0].scale
    except Exception:
        # Fallback to your original indexing logic
        if data_obj.data.ndim > 2:
            cal = data_obj.axes_manager[2].scale
        else:
            cal = data_obj.axes_manager[1].scale

    # Manual override
    if filename_no_ext in MANUAL_CALIBRATIONS:
        cal_original = cal
        cal = MANUAL_CALIBRATIONS[filename_no_ext]
        print(f"  > Calibration OVERRIDDEN: {cal_original:.4f} -> {cal:.4f} nm/px")
    else:
        print(f"  > Calibration: {cal:.4f} nm/px")

    print(f"  > {cal*1000:.2f} pm/px")
    return float(cal)

def load_and_get_enhanced_sum_image(dm3_path: Path):
    filename_no_ext = dm3_path.stem
    print(f"\n[{dm3_path.name}] Loading...")
    data_obj = hs.load(str(dm3_path))

    cal_nm_per_px = get_calibration_nm_per_px(data_obj, filename_no_ext)

    if data_obj.data.ndim == 3:
        print(f"  > Summing stack ({data_obj.data.shape[0]} images)")
        image_sum = np.sum(data_obj.data, axis=0)
    else:
        image_sum = data_obj.data

    print("  > Enhancing image...")
    image_enhanced = enhance_image(image_sum)
    return image_enhanced, cal_nm_per_px

def save_single_class_overlay(image_enhanced, x, y, mask, color, title, out_path,
                              s=8, lw=0.7, origin="upper"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image_enhanced, cmap="gray", origin=origin)
    plt.scatter(x[mask], y[mask], s=s, facecolors="none", edgecolors=color, linewidths=lw)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_counts_bar_chart(class_names, class_counts, total_atoms, title, out_path):
    plt.figure(figsize=(7, 4))
    plt.bar(class_names, class_counts)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title(f"{title} (total atoms = {total_atoms})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_class_histograms(mean_distance_pm, class_items, bins, title, out_path):
    """
    class_items: list of (name, mask, color) in desired order
    """
    plt.figure(figsize=(7, 4))
    ax = plt.gca()

    for name, mask, color in class_items:
        data = mean_distance_pm[mask]
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue

        ax.hist(
            data,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            color=color,
            label=f"{name} (N={data.size})"
        )

    ax.set_xlabel("Mean_Distance_pm")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ----------------------------
# Main loop
# ----------------------------
for subdir in (p for p in csv_root.rglob("*") if p.is_dir()):
    expected_csv = subdir / f"{subdir.name}_atom_data.csv"
    if expected_csv.exists():
        csv_path = expected_csv
    else:
        matches = list(subdir.glob("*_atom_data.csv"))
        if not matches:
            continue
        csv_path = matches[0]

    dm3_path = dm3_root / f"{subdir.name}.dm3"
    if not dm3_path.exists():
        print(f"\n[{subdir.name}] DM3 missing: {dm3_path.name} (skipping)")
        continue

    # CSV read (kept, even if unused for now)
    try:
        df = pd.read_csv(csv_path)
        print(f"\n[{subdir.name}] CSV loaded: {csv_path.name} | shape={df.shape}")
    except Exception as e:
        print(f"\n[{subdir.name}] CSV FAIL: {csv_path} | {e}")
        continue

    try:
        image_enhanced, cal_nm_per_px = load_and_get_enhanced_sum_image(dm3_path)
        
    except Exception as e:
        print(f"  > DM3 FAIL: {dm3_path} | {e}")
        # del df
        continue

    out_dir = output_root / subdir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{subdir.name}_enhanced_sum.png"
    plt.imsave(out_path, image_enhanced, cmap="gray")
    print(f"  > Saved enhanced image: {out_path}")

   # ---- REQUIRED: thresholds (in PIXELS) ----
    a_px = 0.183 / cal_nm_per_px
    eps_px = 0.25 * a_px
    delta_px = 0.4 * a_px
    vac_delta_px = 0.4 * a_px
    
    # ---- Pull columns as numpy arrays ----
    x_all = df["X_Position_px"].to_numpy(float)
    y_all = df["Y_Position_px"].to_numpy(float)
    
    l1_all = df["Distance_NN1_px"].to_numpy(float)
    l2_all = df["Distance_NN2_px"].to_numpy(float)
    l3_all = df["Distance_NN3_px"].to_numpy(float)
    
    # =========================================================
    # Edge filtering (BEFORE classification)
    # =========================================================
    H, W = image_enhanced.shape[:2]
    edge_margin_px = int(round(max(EDGE_MARGIN_MULT_A * a_px, EDGE_MARGIN_MIN_PX)))
    
    edge_keep = (
        np.isfinite(x_all) & np.isfinite(y_all) &
        (x_all >= edge_margin_px) & (x_all < (W - edge_margin_px)) &
        (y_all >= edge_margin_px) & (y_all < (H - edge_margin_px))
    )
    
    keep_idx = np.flatnonzero(edge_keep)
    n_all = len(df)
    n_keep = int(keep_idx.size)
    
    print(f"  > Edge filter: margin={edge_margin_px}px | kept {n_keep}/{n_all}")
    
    if n_keep == 0:
        print("  > WARNING: no atoms left after edge filtering, skipping dataset.")
        continue
    
    # Work only on kept atoms
    x = x_all[keep_idx]
    y = y_all[keep_idx]
    l1 = l1_all[keep_idx]
    l2 = l2_all[keep_idx]
    l3 = l3_all[keep_idx]
    
    # ---- Helper deviations ----
    dev1 = np.abs(l1 - a_px)
    dev2 = np.abs(l2 - a_px)
    dev3 = np.abs(l3 - a_px)
    
    # =========================================================
    # Class 1: No Defect (distance-only)
    # =========================================================
    no_defect = (np.maximum.reduce([dev1, dev2, dev3]) < eps_px)
    
    # =========================================================
    # Class 2: Interstitial (too short), excludes No Defect
    # =========================================================
    too_short = np.minimum.reduce([l1, l2, l3]) < (a_px - delta_px)
    interstitial = (~no_defect) & too_short
    
    # =========================================================
    # Class 3: 1 adjacent vacancy, excludes previous classes
    # l1 ~ a AND l2 ~ a AND l3 >> a
    # =========================================================
    near_a_12 = (dev1 < eps_px) & (dev2 < eps_px)
    l3_far = (l3 > (a_px + vac_delta_px))
    one_adjacent_vacancy = (~no_defect) & (~interstitial) & near_a_12 & l3_far
    
    # =========================================================
    # Class 4: 2 adjacent vacancies
    # l1 ~ a AND l2 >> a AND l3 >> a
    # =========================================================
    near_a_1 = (dev1 < eps_px)
    l2_l3_far = (l2 > (a_px + vac_delta_px)) & (l3 > (a_px + vac_delta_px))
    two_adjacent_vacancy = (~no_defect) & (~interstitial) & (~one_adjacent_vacancy) & near_a_1 & l2_l3_far
    
    # =========================================================
    # Class 5: 3 adjacent vacancies
    # l1 >> a AND l2 >> a AND l3 >> a
    # =========================================================
    l1_l2_l3_far = (l1 > (a_px + vac_delta_px)) & (l2 > (a_px + vac_delta_px)) & (l3 > (a_px + vac_delta_px))
    three_adjacent_vacancy = (~no_defect) & (~interstitial) & (~one_adjacent_vacancy) & (~two_adjacent_vacancy) & l1_l2_l3_far
    
    # =========================================================
    # Other: remaining (kept atoms not in classes 1-5)
    # =========================================================
    other = ~(no_defect | interstitial | one_adjacent_vacancy | two_adjacent_vacancy | three_adjacent_vacancy)
    
    # ---- Build class_items using FILTERED masks + FILTERED x/y ----
    class_items = [
        ("No Defect", no_defect, "green"),
        ("Interstitial", interstitial, "blue"),
        ("1 Adjacent Vacancy", one_adjacent_vacancy, "yellow"),
        ("2 Adjacent Vacancies", two_adjacent_vacancy, "orange"),
        ("3 Adjacent Vacancies", three_adjacent_vacancy, "red"),
        ("Others", other, "purple"),
]
    
    # ---- Plot overlay ----
    out_overlay = out_dir / f"{subdir.name}_no_defect_interstitial_overlay.png"
    plt.figure(figsize=(6, 6))
    plt.imshow(image_enhanced, cmap="gray", origin="upper")
    plt.scatter(x[no_defect], y[no_defect], s=8, facecolors='none', edgecolors='green', label='No Defect', linewidths=0.7)
    plt.scatter(x[interstitial], y[interstitial], s=8,  facecolors='none', edgecolors='blue', label='Interstitial', linewidths=0.7)
    plt.scatter(x[one_adjacent_vacancy], y[one_adjacent_vacancy], s=8,  facecolors='none', edgecolors='yellow', label='1 Adjacent Vacancy', linewidths=0.7)
    plt.scatter(x[two_adjacent_vacancy], y[two_adjacent_vacancy], s=8,  facecolors='none', edgecolors='orange', label='2 Adjacent Vacancies', linewidths=0.7)
    plt.scatter(x[three_adjacent_vacancy], y[three_adjacent_vacancy], s=8,  facecolors='none', edgecolors='red', label='3 Adjacent Vacancies', linewidths=0.7)
    plt.scatter(x[other], y[other], s=8,  facecolors='none', edgecolors='purple', label='Others', linewidths=0.7)
    plt.axis("off")
    plt.legend(loc="upper right")
    plt.tight_layout(pad=0)
    plt.savefig(out_overlay, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  > Saved overlay: {out_overlay}")

    # ----------------------------
    # New outputs per dataset
    # ----------------------------
    overlay_dir = out_dir / "class_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Use same class colors as your main overlay :contentReference[oaicite:5]{index=5}
    class_items = [
        ("No Defect", no_defect, "green"),
        ("Interstitial", interstitial, "blue"),
        ("1 Adjacent Vacancy", one_adjacent_vacancy, "yellow"),
        ("2 Adjacent Vacancies", two_adjacent_vacancy, "orange"),
        ("3 Adjacent Vacancies", three_adjacent_vacancy, "red"),
        ("Others", other, "purple"),
    ]
    
    # 1) Save one labeled image per class (only that class plotted)
    for name, mask, color in class_items:
        out_img = overlay_dir / f"{subdir.name}_{name.replace(' ', '_')}.png"
        save_single_class_overlay(
            image_enhanced=image_enhanced,
            x=x, y=y,
            mask=mask,
            color=color,
            title=f"{subdir.name} | {name} (N={int(mask.sum())})",
            out_path=out_img
        )
    
    # 2) Save bar chart of counts per class (title includes total atoms)
    counts = [int(mask.sum()) for _, mask, _ in class_items]
    names  = [name for name, _, _ in class_items]
    total_atoms = n_keep
    
    out_bar = out_dir / f"{subdir.name}_class_counts_bar.png"
    save_counts_bar_chart(
        class_names=names,
        class_counts=counts,
        total_atoms=total_atoms,
        title=f"{subdir.name} class counts",
        out_path=out_bar
    )
    
    # 3) Histogram of Mean_Distance_pm with same colors as scatter
    mean_distance_pm_all = df["Mean_Distance_pm"].to_numpy(float)
    mean_distance_pm = mean_distance_pm_all[keep_idx]  # IMPORTANT: align with masks
    
    # shared bins across classes for this dataset (use kept range)
    bins = np.linspace(np.nanmin(mean_distance_pm), np.nanmax(mean_distance_pm), 60)
    
    out_hist = out_dir / f"{subdir.name}_Mean_Distance_pm_hist_by_class.png"
    save_class_histograms(
        mean_distance_pm=mean_distance_pm,
        class_items=class_items,
        bins=bins,
        title=f"{subdir.name} Mean_Distance_pm by class (total atoms = {n_keep})",
        out_path=out_hist
    )
    
    print(f"  > Saved per-class overlays to: {overlay_dir}")
    print(f"  > Saved bar chart: {out_bar}")
    print(f"  > Saved histogram: {out_hist}")

    # ----------------------------
    # Save per-atom classification CSV (6 classes as float)
    # ----------------------------
    df_kept = df.iloc[keep_idx].copy()

    defect_class = np.full(n_keep, np.nan, dtype=float)
    defect_class[no_defect] = 1.0
    defect_class[interstitial] = 2.0
    defect_class[one_adjacent_vacancy] = 3.0
    defect_class[two_adjacent_vacancy] = 4.0
    defect_class[three_adjacent_vacancy] = 5.0
    defect_class[other] = 6.0
    
    out_class_df = pd.DataFrame({
        "Atom_Index": df_kept["Atom_Index"].to_numpy(),
        "Y_Position_px": df_kept["Y_Position_px"].to_numpy(float),
        "X_Position_px": df_kept["X_Position_px"].to_numpy(float),
        "Defect_Class": defect_class,
    })
    
    out_class_csv = out_dir / f"{subdir.name}_atom_defect_class.csv"
    out_class_df.to_csv(out_class_csv, index=False)
    print(f"  > Saved atom class CSV: {out_class_csv}")

    del df
    del image_enhanced

# Record the end time
end_time = time.perf_counter()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
