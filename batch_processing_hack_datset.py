"""
Batch Atom Distance Analysis (No Peak Assignment)
- Processes ALL .dm3/.dm4 files in a folder
- Sums stacks automatically
- Detects atoms and calculates nearest neighbor statistics
- SAVES ALL DIAGNOSTIC PLOTS
- Exports comprehensive CSV with distances, angles, and intensities
- NO peak assignment or Gaussian fitting
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from skimage.feature import peak_local_max, blob_dog
from skimage.filters import gaussian
from scipy.ndimage import generic_filter
from scipy import ndimage
from scipy.spatial import distance_matrix
from scipy.spatial import KDTree
from TEMDepthEstimator import TEMDepthEstimator
depth_estimator = TEMDepthEstimator()

import hyperspy.api as hs
import cv2
import csv
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# I/O Settings
# INPUT_FOLDER = '/Users/George/Desktop/PhD/ML/Microscopy Hackathon 2025/QuaternaryAlloyMoWSSe-20251216T170946Z-1-001/testfile'  # Folder containing .dm3 files
# OUTPUT_ROOT = '/Users/George/Desktop/PhD/ML/Microscopy Hackathon 2025/QuaternaryAlloyMoWSSe-20251216T170946Z-1-001/atom-analysis-test'
INPUT_FOLDER = '/Users/George/Desktop/PhD/ML/Microscopy Hackathon 2025/QuaternaryAlloyMoWSSe-20251216T170946Z-1-001/QuaternaryAlloyMoWSSe'  # Folder containing .dm3 files
OUTPUT_ROOT = '/Users/George/Desktop/PhD/ML/Microscopy Hackathon 2025/QuaternaryAlloyMoWSSe-20251216T170946Z-1-001/atom-analysis'

# Diagnostic plots
DIAGNOSTIC_PLOTS = True

# Merge Stacks
MERGE_STACKS = True

# Image Enhancement Parameters
CLAHE_CLIP_LIMIT = 0.03
CLAHE_TILE_SIZE = 8
GAUSSIAN_BLUR_SIGMA = 1
BKGD_NORMALIZATION_SIZE = 480 #picometers

# Atom Detection Parameters
PEAK_RADIUS = 2
PEAK_THRESHOLD = 0.04
MIN_SIGMA = 32 #min stdev of a gaussian to fit on an atom, picometers
MAX_SIGMA = 50 #max stdev of a gaussian to fit on an atom, picometers

# Nearest Neighbor Parameters
N_NEAREST_NEIGHBORS = 3

# Visualization Parameters
FIGURE_DPI = 150
INTENSITY_RADIUS = 3

MIN_AREA_PERCENTAGE = 2.5  # Adjust this value as needed
DEPTH_THRESHOLD = 0.85  # Depth threshold for contour detection


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
# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def enhance_image(image):
    """Apply CLAHE and Gaussian blur to enhance image"""
    image = np.nan_to_num(image)
    img_min, img_max = image.min(), image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)

    img_norm = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    img_clahe = clahe.apply(img_norm)
    img_blur = ndimage.gaussian_filter(img_clahe, GAUSSIAN_BLUR_SIGMA)
    return (img_blur - img_blur.min()) / (img_blur.max() - img_blur.min())

def normalize_bkgd(image, size):
    eps = 1e-6
    local_std = generic_filter(image, np.std, size=size)
    normalized = image / (local_std + eps)
    return normalized

# def detect_atoms(image, radius, threshold):
#     """Detect atom positions using peak detection"""
#     out = peak_local_max(image, min_distance=radius,
#                           threshold_rel=threshold, exclude_border=True)
#     print(out)
#     return out


def detect_atoms(image, min_sigma, max_sigma, threshold):
    """Detect atom positions using peak detection"""
    out = blob_dog(image, min_sigma=min_sigma, max_sigma=max_sigma,
                          threshold=threshold, exclude_border=True)
    return out



def find_k_nearest_neighbors(atoms, k=3):
    """
    Find k nearest neighbors using KD-tree.
    
    Parameters
    ----------
    atoms : ndarray, shape (N, 2)
    k : int
        Number of neighbors (excluding self)
    
    Returns
    -------
    dict with:
        neighbor_indices : ndarray (N, k)
        neighbor_distances : ndarray (N, k)
        avg_distances : ndarray (N,)
    """
    if len(atoms) < k + 1:
        return None
    
    tree = KDTree(atoms)
    
    # Query k+1 because first result is the point itself (distance=0)
    distances, indices = tree.query(atoms, k=k+1, workers=-1)
    
    # Strip self-matches (first column)
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    
    return {
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'avg_distances': neighbor_distances.mean(axis=1)
    }


def calculate_neighbor_angles(atoms, neighbor_indices_list):
    """
    Calculate angles between atom and its 3 nearest neighbors
    Uses first neighbor as reference (0 degrees), calculates angles to other 2

    Returns:
        angles_list: List of [angle_to_neighbor2, angle_to_neighbor3] for each atom
    """
    angles_list = []

    for i, neighbor_indices in enumerate(neighbor_indices_list):
        if len(neighbor_indices) < 3:
            angles_list.append([np.nan, np.nan])
            continue

        atom_pos = atoms[i]

        # Get positions of 3 nearest neighbors
        neighbor1_pos = atoms[neighbor_indices[0]]
        neighbor2_pos = atoms[neighbor_indices[1]]
        neighbor3_pos = atoms[neighbor_indices[2]]

        # Calculate vectors from atom to each neighbor
        vec1 = neighbor1_pos - atom_pos
        vec2 = neighbor2_pos - atom_pos
        vec3 = neighbor3_pos - atom_pos

        # Calculate angles relative to first neighbor
        def angle_between_vectors(v1, v2):
            """Calculate signed angle from v1 to v2 in degrees"""
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            # Convert to degrees and normalize to [0, 360)
            angle_deg = np.degrees(angle)
            if angle_deg < 0:
                # angle_deg += 360
                angle_deg = -1 * angle_deg
            if angle_deg > 180:
                angle_deg = 360-angle_deg
            return angle_deg

        angle2 = angle_between_vectors(vec1, vec2)
        angle3 = angle_between_vectors(vec1, vec3)

        # remainderangle = 360 - angle2 - angle3 
        # sortedangles = np.sort(np.array([angle2, angle3, remainderangle]))
        # angles_list.append([sortedangles[0], sortedangles[1]])
        angles_list.append([angle2, angle3])


    return angles_list


def calculate_atom_intensity(image, atom_pos, radius=3):
    """
    Calculate mean intensity around an atom position
    """
    y, x = int(atom_pos[0]), int(atom_pos[1])

    y_min = max(0, y - radius)
    y_max = min(image.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(image.shape[1], x + radius + 1)

    region = image[y_min:y_max, x_min:x_max]
    return np.mean(region) if region.size > 0 else 0.0


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def save_all_plots(output_folder, filename_base, image, image_enhanced, atoms,
                   avg_distances, cal_pm):
    """Generates and saves all diagnostic plots to the specific image folder"""

    print(f"  > Generating diagnostic plots in {output_folder}...")

    # 1. Enhancement Check
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original")
    ax[0].axis('off')
    ax[1].imshow(image_enhanced, cmap='gray')
    ax[1].set_title("Enhanced")
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '01_enhancement' + filename_base + '.png'))
    plt.close()

    # 2. Detected Atoms
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=FIGURE_DPI)
    ax.imshow(image_enhanced, cmap='gray')  

    # Create all patches at once
    circles = [Circle((a[1], a[0]), radius=2*a[2]) for a in atoms]
    collection = PatchCollection(circles, facecolors='none', edgecolors='red', linewidths=1)
    ax.add_collection(collection)

    ax.set_title(f"Detected {len(atoms)} Atoms")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_atoms' + filename_base + '.png'))
    plt.close()

    # 3. Distance Map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=FIGURE_DPI)
    sc = ax.scatter(atoms[:, 1], atoms[:, 0], c=avg_distances, cmap='viridis', s=30)
    ax.imshow(image_enhanced, cmap='gray', alpha=0.5)
    plt.colorbar(sc, label='Avg Distance (px)', ax=ax)
    ax.set_title("Distance Map")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_distance_map' + filename_base + '.png'))
    plt.close()

    # 4. Distance Histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=FIGURE_DPI)
    ax.hist(avg_distances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(avg_distances.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {avg_distances.mean():.2f} px ({avg_distances.mean()*cal_pm:.1f} pm)')
    ax.set_xlabel('Average Distance to 3 Nearest Neighbors (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distance Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_histogram' + filename_base + '.png'))
    plt.close()

    # 5. Summary Composite
    fig = plt.figure(figsize=(15, 10), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Input")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image_enhanced, cmap='gray')

    circles = [Circle((a[1], a[0]), radius=2*a[2]) for a in atoms]
    collection = PatchCollection(circles, facecolors='none', edgecolors='red', 
                                linewidths=1, alpha=0.5)
    ax2.add_collection(collection)

    ax2.set_title(f"Detected Atoms (n={len(atoms)})")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(atoms[:, 1], atoms[:, 0], c=avg_distances, cmap='viridis', s=20)
    ax3.imshow(image_enhanced, cmap='gray', alpha=0.3)
    ax3.set_title("Distance Map")
    ax3.axis('off')
    plt.colorbar(sc, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[1, 1])
    text = f"File: {filename_base}\n"
    text += f"Atoms: {len(atoms)}\n"
    text += f"Mean Distance: {avg_distances.mean():.2f} px ({avg_distances.mean()*cal_pm:.1f} pm)\n"
    text += f"Std Distance: {avg_distances.std():.2f} px\n"
    text += f"Min Distance: {avg_distances.min():.2f} px ({avg_distances.min()*cal_pm:.1f} pm)\n"
    text += f"Max Distance: {avg_distances.max():.2f} px ({avg_distances.max()*cal_pm:.1f} pm)\n"

    ax4.text(0.1, 0.5, text, fontsize=12, fontfamily='monospace',
             verticalalignment='center')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '05_summary' + filename_base + '.png'))
    plt.close()

def save_depth_contour_visualization(output_folder, filename_base, image_enhanced,
                                     depth_map, binary_mask, all_contours,
                                     qualified_contours, threshold, min_area_percentage):
    """
    Visualize the depth map and detected contours for debugging
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=FIGURE_DPI)

    # 1. Original enhanced image
    axes[0, 0].imshow(image_enhanced, cmap='gray')
    axes[0, 0].set_title('Enhanced Image')
    axes[0, 0].axis('off')

    # 2. Depth map
    im = axes[0, 1].imshow(depth_map, cmap='plasma')
    plt.colorbar(im, ax=axes[0, 1], label='Depth')
    axes[0, 1].set_title(f'Depth Map (Threshold: {threshold})')
    axes[0, 1].axis('off')

    # 3. Binary threshold image with ALL contours
    axes[1, 0].imshow(image_enhanced, cmap='gray', alpha=0.5)
    # Draw all contours in red
    contour_img_all = np.zeros_like(image_enhanced)
    cv2.drawContours(contour_img_all, all_contours, -1, 1, 2)
    axes[1, 0].imshow(contour_img_all, cmap='Reds', alpha=0.7)
    axes[1, 0].set_title(f'All Contours (n={len(all_contours)})')
    axes[1, 0].axis('off')

    # 4. Qualified contours only (after area filtering)
    axes[1, 1].imshow(image_enhanced, cmap='gray', alpha=0.5)
    # Draw qualified contours in green, filled
    contour_img_qualified = np.zeros_like(image_enhanced)
    cv2.drawContours(contour_img_qualified, qualified_contours, -1, 1, -1)
    axes[1, 1].imshow(contour_img_qualified, cmap='Greens', alpha=0.7)
    axes[1, 1].set_title(f'Qualified Contours (>={min_area_percentage}% area, n={len(qualified_contours)})')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_folder, f'{filename_base}_depth_contours.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  > Saved depth contour visualization: {filename_base}_depth_contours.png")

def save_csv_output(output_folder, filename_base, atoms, nn_results,
                    angles_list, intensities, cal, cal_pm):
    """
    Save comprehensive CSV file with atom data (matching second code format)
    """
    csv_path = os.path.join(output_folder, f'{filename_base}_atom_data.csv')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header matching the second code
        writer.writerow([
            'Atom_Index',
            'Y_Position_px',
            'X_Position_px',
            'Distance_NN1_px',
            'Distance_NN2_px',
            'Distance_NN3_px',
            'Mean_Distance_px',
            'Mean_Distance_pm',
            'Angle_to_NN2_deg',
            'Angle_to_NN3_deg',
            'Mean_Intensity'
        ])

        # Data rows
        for i in range(len(atoms)):
            atom_pos = atoms[i]
            distances = nn_results['neighbor_distances'][i]
            avg_dist = nn_results['avg_distances'][i]
            angles = angles_list[i]
            intensity = intensities[i]

            writer.writerow([
                i,  # Atom index
                f'{atom_pos[0]:.2f}',  # Y position
                f'{atom_pos[1]:.2f}',  # X position
                f'{distances[0]:.4f}',  # Distance to NN1
                f'{distances[1]:.4f}',  # Distance to NN2
                f'{distances[2]:.4f}',  # Distance to NN3
                f'{avg_dist:.4f}',  # Mean distance in pixels
                f'{avg_dist * cal_pm:.2f}',  # Mean distance in pm
                f'{angles[0]:.2f}',  # Angle to NN2
                f'{angles[1]:.2f}',  # Angle to NN3
                f'{intensity:.4f}'  # Mean intensity
            ])

    print(f"  > Saved: {filename_base}_atom_data.csv")
    return csv_path


# ============================================================================
# MAIN PROCESS
# ============================================================================

def process_image(image, cal, file_output_folder, filename_no_ext):
    # Save original image at full resolution
    original_path = os.path.join(file_output_folder, f'{filename_no_ext}_original.png')
    plt.imsave(original_path, image, cmap='gray')
    print(f"  > Saved original image: {filename_no_ext}_original.png")

    cal_pm = cal*1e3

    # Pipeline
    image_enhanced = image
    # print(f"  > Enhancing image...")
    # image_enhanced = enhance_image(image)

    # Found DoG seems to work better w/out normalization 
    # print(f"  > Normalizing image")
    # image_enhanced = normalize_bkgd(image_enhanced, round(BKGD_NORMALIZATION_SIZE / cal_pm))

    print(f"  > Detecting atoms...")
    atoms = detect_atoms(image_enhanced, MIN_SIGMA / cal_pm, MAX_SIGMA / cal_pm, PEAK_THRESHOLD)
    print(f"  > Detected {len(atoms)} atoms")

    # Filter atoms by depth contours
    print(f"  > Filtering atoms by depth contours...")

    filtered_atoms, num_contours, contour_area, depth_map, binary_mask, all_contours, qualified_contours = depth_estimator.filter_atoms_by_depth_contours(
        image_enhanced=image_enhanced,
        atoms=atoms,
        threshold=DEPTH_THRESHOLD,
        min_area_percentage=MIN_AREA_PERCENTAGE
    )
    
    # Save visualization of detected contours
    if DIAGNOSTIC_PLOTS:
        save_depth_contour_visualization(
            file_output_folder,
            filename_no_ext,
            image_enhanced,
            depth_map,
            binary_mask,
            all_contours,
            qualified_contours,
            DEPTH_THRESHOLD,
            MIN_AREA_PERCENTAGE
        )

    # Update atoms with filtered results
    atoms = filtered_atoms

    if len(atoms) < 5:
        print(f"  > Too few atoms ({len(atoms)}). Skipping.")
        return

    # Calculate nearest neighbors
    print(f"  > Finding {N_NEAREST_NEIGHBORS} nearest neighbors...")
    nn_res = find_k_nearest_neighbors(atoms, k=N_NEAREST_NEIGHBORS)

    if nn_res is None:
        print(f"  > Not enough atoms for neighbor analysis. Skipping.")
        return

    # Calculate angles
    print(f"  > Calculating angles...")
    angles_list = calculate_neighbor_angles(atoms, nn_res['neighbor_indices'])

    # Calculate intensities
    print(f"  > Calculating intensities...")
    intensities = []
    for atom in atoms:
        intensity = calculate_atom_intensity(image, atom, radius=INTENSITY_RADIUS)
        intensities.append(intensity)
    intensities = np.array(intensities)

    # Print statistics
    avg_distances = nn_res['avg_distances']
    print(f"\n  > Distance Statistics:")
    print(f"    Mean: {avg_distances.mean():.2f} px ({avg_distances.mean() * cal_pm:.1f} pm)")
    print(f"    Std:  {avg_distances.std():.2f} px")
    print(f"    Min:  {avg_distances.min():.2f} px ({avg_distances.min() * cal_pm:.1f} pm)")
    print(f"    Max:  {avg_distances.max():.2f} px ({avg_distances.max() * cal_pm:.1f} pm)")

    # Save CSV
    save_csv_output(file_output_folder, filename_no_ext, atoms, nn_res,
                               angles_list, intensities, cal, cal_pm)

    # Save all diagnostic plots
    if DIAGNOSTIC_PLOTS:
        save_all_plots(file_output_folder, filename_no_ext, image, image_enhanced,
                        atoms, avg_distances, cal_pm)

def process_single_file(file_path, output_base_folder):
    """Process a single .dm3/.dm4 file"""
    filename = os.path.basename(file_path)
    filename_no_ext = os.path.splitext(filename)[0]

    # Output folder for THIS file
    file_output_folder = os.path.join(output_base_folder, filename_no_ext)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n[{filename}] Loading...")
    try:
        data_obj = hs.load(file_path)
    except Exception as e:
        print(f"  > Error loading {filename}: {str(e)}")
        return

    # Get calibration
    if data_obj.data.ndim > 2:
        cal = data_obj.axes_manager[2].scale
    else:
        cal = data_obj.axes_manager[1].scale

    # Check for manual calibration override
    if filename_no_ext in MANUAL_CALIBRATIONS:
        cal_original = cal
        cal = MANUAL_CALIBRATIONS[filename_no_ext]
        print(f"  > Calibration OVERRIDDEN: {cal_original:.4f} -> {cal:.4f} nm/px")
    else:
        print(f"  > Calibration: {cal:.4f} nm/px", end="")

    cal_pm = cal * 1000
    print(f" = {cal_pm:.2f} pm/px")

    # Handle stack vs single image

    if data_obj.data.ndim == 3:
        if MERGE_STACKS:
            print(f"  > Summing stack ({data_obj.data.shape[0]} images)")
            image = np.sum(data_obj.data, axis=0)
            process_image(image, cal, file_output_folder, filename_no_ext)
        else:
            for i in range(data_obj.data.shape[0]):
                image = data_obj.data[i,:,:]
                filename_numbered = filename_no_ext + f"_{i}"
                process_image(image, cal, file_output_folder, filename_numbered)
    else:
        image = data_obj.data
        process_image(image, cal, file_output_folder, filename_no_ext)
    
        

    print(f"  > Processing complete for {filename}")


def main():
    """Main batch processing function"""
    print("="*60)
    print("BATCH ATOM ANALYSIS - NEAREST NEIGHBOR STATISTICS")
    print("="*60)

    files = glob.glob(os.path.join(INPUT_FOLDER, '*.dm3')) + \
            glob.glob(os.path.join(INPUT_FOLDER, '*.dm4'))

    if not files:
        print("No .dm3 or .dm4 files found.")
        return

    print(f"\nFound {len(files)} files")
    print(f"Output directory: {OUTPUT_ROOT}")

    for i, f in enumerate(files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(files)}: {os.path.basename(f)}")
        print(f"{'='*60}")
        process_single_file(f, OUTPUT_ROOT)

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()