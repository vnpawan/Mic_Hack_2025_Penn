
"""
Atom Analysis Pipeline - Refactored
Process multiple image files, detect atoms, and classify elements

FOLDER STRUCTURE (from batch processing):
=========================================
INPUT_FOLDER/
├── 3D Stack align4/              # Each file gets its own folder
│   ├── 3D Stack align4_original.png
│   ├── 3D Stack align4_atom_data.csv
│   ├── 01_enhancement3D Stack align4.png
│   ├── 02_atoms3D Stack align4.png
│   └── ... (other diagnostic plots)
├── 3D Stack align5/
│   ├── 3D Stack align5_original.png
│   ├── 3D Stack align5_atom_data.csv
│   └── ...
└── ... (more folders)

USAGE:
======
1. Update the configuration section below:
   - INPUT_FOLDER: Root folder containing the batch processing output
   - FILES_TO_PROCESS: List of folder names to process (just the names!)
   - CSV_FILES_FOR_FITTING: (Optional) List of folder names for global fitting
     Leave empty [] to use all files for fitting

2. Run the script:
   python Hackathon_2025_refactored.py

EXAMPLE CONFIGURATION:
======================
FILES_TO_PROCESS = [
    "3D Stack align4",
    "3D Stack align5",
    "3D Stack align6",
]

CSV_FILES_FOR_FITTING = []  # Use all files for fitting
# OR
CSV_FILES_FOR_FITTING = ["3D Stack align4", "3D Stack align5"]  # Use subset

WORKFLOW:
=========
Step 1: Aggregate data from selected CSV files
Step 2: Fit global A-site and B-site histograms
Step 3: Process all files using global fit parameters
Step 4: Save aggregated results with file_id tracking

OUTPUT:
=======
- Updates each CSV with Element and File_ID columns
- Creates all_atoms_classified.csv with combined data
- Each atom has: [y, x, intensity_scaled, element, file_id]

CALIBRATION:
============
Automatically extracted from .dm3/.dm4 files in the folder or parent folder.
Falls back to default (0.0156 nm/pixel) if metadata unavailable.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
import hyperspy.api as hs
import cv2
import csv
import pandas as pd
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Folders - batch processing creates a folder for each file
# Structure: INPUT_FOLDER/filename_no_ext/filename_no_ext_atom_data.csv
INPUT_FOLDER = "/Users/vnpawan/Downloads/atom-analysis_Thursday"
OUTPUT_FOLDER = "/Users/vnpawan/Downloads/atom-analysis_Thursday_1"

# Calibration - will be automatically extracted from image metadata
CAL = None  # Will be set automatically
CAL_PM = None  # Will be set automatically

# Figure settings
FIGURE_DPI = 100

# Atom sorting parameters
T = 0.25  # Threshold for atom sorting

# Files to process - MODIFY THIS LIST
# Just provide the folder names! CSV files will be found automatically
# Each folder should contain: foldername_original.png and foldername_atom_data.csv
# Example folder structure:
#   INPUT_FOLDER/3D Stack align4/3D Stack align4_original.png
#   INPUT_FOLDER/3D Stack align4/3D Stack align4_atom_data.csv
FILES_TO_PROCESS = [
    "3D Stack align4",
    "3D Stack align5",
    "3D Stack align6",
    "3D Stack align1",
    "3D Stack align2",
    "3D Stack align3",
    "3D Stack align7",
    "3D Stack align8",
    "3D Stack align9",
    "3D Stack align10",
    "3D Stack align11",
    "3D Stack align12",
    "3D Stack align13",
    "HAADF-88055",
    "HAADF-88147",
    "HAADF-88165",
    "HAADF-88282",
    "HAADF-88507",
    "HAADF-89245"
]


# Files to use for global fitting - MODIFY THIS LIST (OPTIONAL)
# If empty or None, all files in FILES_TO_PROCESS will be used for fitting
# Otherwise, specify which subset to use for global fitting
CSV_FILES_FOR_FITTING = []  # Use all files


# CSV_FILES_FOR_FITTING = ["3D Stack align4", "3D Stack align5"]  # Use only these for fitting

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_calibration_from_image(folder_path, item):
    """
    Extract calibration from image file metadata using hyperspy

    Args:
        folder_path: path to folder containing the image (e.g., INPUT_FOLDER/item/)
        item: item name (without extension)

    Returns:
        cal: calibration in nm/pixel
        cal_pm: calibration in pm/pixel
    """
    # The batch processing script creates folders: OUTPUT_ROOT/filename_no_ext/
    # And saves files as: filename_no_ext_original.png, filename_no_ext_atom_data.csv

    # First, try to find the original .dm3/.dm4 file in the parent INPUT_FOLDER
    # The batch script processes files from INPUT_FOLDER and creates output in subfolders
    parent_folder = os.path.dirname(folder_path) if os.path.basename(folder_path) == item else folder_path

    for ext in ['.dm3', '.dm4', '.dm']:
        try:
            # Try in the subfolder first
            file_path = os.path.join(folder_path, item + ext)
            if os.path.exists(file_path):
                print(f"  > Loading calibration from: {file_path}")
                data = hs.load(file_path)

                # Try to get calibration from axes_manager
                if hasattr(data, 'axes_manager'):
                    # For stacks, use the spatial axes (usually the last two)
                    if len(data.data.shape) == 3:
                        # Stack: use axis 2 (last spatial axis)
                        cal = data.axes_manager[2].scale
                    else:
                        # Single image: use axis 1 (x-axis, to match batch script)
                        cal = data.axes_manager[1].scale

                    cal_pm = cal * 1000  # Convert nm to pm
                    print(f"  > Calibration extracted: {cal:.4f} nm/pixel = {cal_pm:.2f} pm/pixel")
                    return cal, cal_pm

            # Try in the parent folder (where original .dm3 files are)
            file_path = os.path.join(parent_folder, item + ext)
            if os.path.exists(file_path):
                print(f"  > Loading calibration from: {file_path}")
                data = hs.load(file_path)

                # Try to get calibration from axes_manager
                if hasattr(data, 'axes_manager'):
                    if len(data.data.shape) == 3:
                        cal = data.axes_manager[2].scale
                    else:
                        cal = data.axes_manager[1].scale

                    cal_pm = cal * 1000
                    print(f"  > Calibration extracted: {cal:.4f} nm/pixel = {cal_pm:.2f} pm/pixel")
                    return cal, cal_pm

        except Exception as e:
            continue

    # Default calibration if no metadata file found
    print("  > Warning: Could not extract calibration from metadata, using default")
    cal = 0.0156  # Default value (matches batch script default)
    cal_pm = cal * 1000
    print(f"  > Using default calibration: {cal:.4f} nm/pixel = {cal_pm:.2f} pm/pixel")
    return cal, cal_pm


def dist(a, b):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def gaussian(x, amplitude, mean, sigma):
    """Single Gaussian function"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def multi_gaussian(x, *params):
    """Sum of multiple Gaussians"""
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i + 1], params[i + 2])
    return y


def bi_gaussian(x, a1, m1, s1, a2, m2, s2):
    """Sum of two Gaussians for A-site fitting"""
    return gaussian(x, a1, m1, s1) + gaussian(x, a2, m2, s2)


def bi_gaussian_fixed_means(x, a1, s1, a2, s2):
    """Sum of two Gaussians with FIXED means at 0.3 and 0.8"""
    m1 = 0.3  # Fixed mean for Peak 1
    m2 = 0.8  # Fixed mean for Peak 2
    return gaussian(x, a1, m1, s1) + gaussian(x, a2, m2, s2)


def tri_gaussian(x, a1, m1, s1, a2, m2, s2, a3, m3, s3):
    """Sum of three Gaussians for B-site fitting"""
    return (gaussian(x, a1, m1, s1) +
            gaussian(x, a2, m2, s2) +
            gaussian(x, a3, m3, s3))


def sort_atoms_into_sublattices(atoms, threshold=0.25):
    """
    Sort atoms into A and B sublattices based on nearest neighbor distances
    Returns: (Aatoms, Batoms, Aint, Bint)
    """
    natoms = len(atoms)
    d = []
    idx = []

    # Step 1: Find Nearest Neighbors for all atoms
    for i in range(natoms):
        dmin = 100000
        jmin = 0
        for j in range(natoms):
            if i != j:
                dij = dist(atoms[i], atoms[j])
                if dij < dmin:
                    dmin = dij
                    jmin = j
        d.append(dmin)
        idx.append(jmin)

    # Step 2: Sort into A and B Sites
    Aatoms = []
    Batoms = []
    for i in range(natoms):
        j = idx[i]
        if dist(atoms[i], atoms[idx[j]]) > threshold:
            Aatoms.append(atoms[i])
        else:
            Batoms.append(atoms[i])

    return Aatoms, Batoms


def normalize_image_locally(image, mask_size=64):
    """Apply local normalization to image"""
    normalized = image.copy()

    for i in range(0, image.shape[0], mask_size):
        for j in range(0, image.shape[1], mask_size):
            top, bottom = j, j + mask_size
            left, right = i, i + mask_size

            region_data = image[top:bottom, left:right]
            mean = np.mean(region_data)
            normalize = region_data - mean

            normalized[top:bottom, left:right] = normalize

    normalized = normalized - normalized.min()
    return normalized


def extract_intensities(normalized, Aatoms, Batoms):
    """Extract intensity values for A and B atoms"""
    Aint = [normalized[int(atom[0]), int(atom[1])] for atom in Aatoms]
    Bint = [normalized[int(atom[0]), int(atom[1])] for atom in Batoms]
    return Aint, Bint


def fit_histogram(intensities, num_peaks, initial_guess, fix_means=False):
    """
    Fit histogram with multiple Gaussian peaks

    Args:
        intensities: list of intensity values
        num_peaks: number of Gaussian peaks to fit (2 or 3)
        initial_guess: list of initial parameters [amp, mean, sigma] for each peak
        fix_means: if True and num_peaks==2, fix A-site means at 0.3 and 0.8

    Returns:
        popt: optimized parameters
        pcov: covariance matrix
    """
    # Create histogram
    counts, bin_edges = np.histogram(intensities, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts = counts / np.sum(counts)  # Normalize

    # Choose the appropriate function
    if num_peaks == 2:
        if fix_means:
            # Use fixed means version - only fit amplitudes and sigmas
            fit_func = bi_gaussian_fixed_means
            # Initial guess: [a1, s1, a2, s2] (no means)
            initial_guess_fixed = [initial_guess[0], initial_guess[2],
                                   initial_guess[3], initial_guess[5]]
            lower_bounds = [0, 0.01, 0, 0.01]  # amp >= 0, sigma >= 0.01
            upper_bounds = [np.inf, 0.3, np.inf, 0.3]  # sigma <= 0.3

            try:
                popt, pcov = curve_fit(fit_func, bin_centers, counts,
                                       p0=initial_guess_fixed,
                                       bounds=(lower_bounds, upper_bounds),
                                       maxfev=10000)
            except RuntimeError as e:
                print(f"  Warning: Fit failed - {e}")
                popt = initial_guess_fixed
                pcov = None

            # Expand popt to include fixed means for consistency
            # [a1, m1, s1, a2, m2, s2]
            popt_full = [popt[0], 0.3, popt[1], popt[2], 0.8, popt[3]]
            return popt_full, pcov, bin_centers, counts
        else:
            fit_func = bi_gaussian
    elif num_peaks == 3:
        fit_func = tri_gaussian
    else:
        raise ValueError("num_peaks must be 2 or 3")

    # Set up bounds to ensure peak separation (for non-fixed case)
    lower_bounds = []
    upper_bounds = []

    if num_peaks == 2 and not fix_means:
        # For 2 peaks: ensure they're separated, Peak 2 around 0.8
        lower_bounds = [
            0, 0.0, 0.01,  # Peak 1: amp >= 0, mean >= 0, sigma >= 0.01
            0, 0.5, 0.01  # Peak 2: amp >= 0, mean >= 0.5, sigma >= 0.01
        ]
        upper_bounds = [
            np.inf, 0.6, 0.3,  # Peak 1: mean <= 0.6, sigma <= 0.3
            np.inf, 1.0, 0.3  # Peak 2: mean <= 1.0, sigma <= 0.3
        ]
    elif num_peaks == 3:
        # For 3 peaks: ensure progressive separation
        lower_bounds = [
            0, 0.0, 0.01,  # Peak 1: lowest intensity
            0, 0.2, 0.01,  # Peak 2: middle intensity
            0, 0.4, 0.01  # Peak 3: highest intensity
        ]
        upper_bounds = [
            np.inf, 0.3, 0.3,  # Peak 1: mean <= 0.3
            np.inf, 0.5, 0.3,  # Peak 2: mean <= 0.5
            np.inf, 1.0, 0.3  # Peak 3: mean <= 1.0
        ]

    # Perform fit
    try:
        popt, pcov = curve_fit(fit_func, bin_centers, counts,
                               p0=initial_guess, bounds=(lower_bounds, upper_bounds),
                               maxfev=10000)
    except RuntimeError as e:
        print(f"  Warning: Fit failed - {e}")
        popt = initial_guess
        pcov = None

    return popt, pcov, bin_centers, counts


def plot_fit_results(bin_centers, counts, popt, title, num_peaks):
    """Plot histogram with fitted Gaussians"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate bin width
    bin_edges = np.concatenate([[bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
                                (bin_centers[:-1] + bin_centers[1:]) / 2,
                                [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2]])
    width = bin_edges[1] - bin_edges[0]

    # Top plot: Original data with fit
    ax1.bar(bin_centers, counts, width=width, alpha=0.5,
            label='Histogram data', color='gray')
    ax1.plot(bin_centers, multi_gaussian(bin_centers, *popt),
             'r-', linewidth=2, label='Combined fit')

    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
    for i in range(num_peaks):
        gauss = gaussian(bin_centers, popt[i * 3], popt[i * 3 + 1], popt[i * 3 + 2])
        ax1.plot(bin_centers, gauss, '--', linewidth=2,
                 color=colors[i % len(colors)],
                 label=f'Peak {i + 1}')

    ax1.set_xlabel('Normalized Intensity', fontsize=12)
    ax1.set_ylabel('Normalized Counts', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Residuals
    fitted = multi_gaussian(bin_centers, *popt)
    residuals = counts - fitted
    ax2.bar(bin_centers, residuals, width=width, color='red', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Normalized Intensity', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save instead of show - add timestamp to avoid overwriting
    import time
    timestamp = int(time.time())
    save_path = f"fit_results_{title.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved fit plot: {save_path}")

    # Print fitted parameters
    print("\nFitted Parameters:")
    print("-" * 60)
    for i in range(num_peaks):
        peak_num = i + 1
        amp, mean, sigma = popt[i * 3], popt[i * 3 + 1], popt[i * 3 + 2]
        print(f"Peak {peak_num}:")
        print(f"  Amplitude: {amp:.4f}")
        print(f"  Mean:      {mean:.4f}")
        print(f"  Sigma:     {sigma:.4f}")
        area = amp * sigma * np.sqrt(2 * np.pi)
        print(f"  Area:      {area:.4f}\n")


def create_element_overlay(output_folder, item, image, atoms, element_list, file_id):
    """
    Create overlay visualization showing element assignments on original image

    Args:
        output_folder: path to save the image
        item: file name
        image: original image
        atoms: list of atom positions
        element_list: list of element classifications (1-5)
        file_id: unique file identifier
    """
    print(f"  > Creating element overlay visualization...")

    # Define colors for each element (1-5)
    element_colors = {
        1: 'red',  # Element 1
        2: 'orange',  # Element 2
        3: 'yellow',  # Element 3
        4: 'green',  # Element 4
        5: 'blue'  # Element 5
    }

    element_names = {
        1: 'Element 1',
        2: 'Element 2',
        3: 'Element 3',
        4: 'Element 4',
        5: 'Element 5'
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=FIGURE_DPI)

    # Show original image
    ax.imshow(image, cmap='gray')

    # Plot atoms colored by element
    for atom, element in zip(atoms, element_list):
        color = element_colors.get(element, 'white')
        ax.add_patch(Circle((atom[1], atom[0]), radius=4,
                            alpha=0.7, fc=color, ec='black', lw=1))

    # Count atoms per element
    element_counts = {i: element_list.count(i) for i in range(1, 6)}

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = []
    for elem_num in range(1, 6):
        count = element_counts[elem_num]
        color = element_colors[elem_num]
        label = f"{element_names[elem_num]}: {count} atoms"
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.9, title='Element Classification')

    # Title
    total_atoms = len(atoms)
    ax.set_title(f"{item} - Element Classification\n{total_atoms} atoms detected",
                 fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()

    # Save to subfolder
    save_path = os.path.join(output_folder, f'{item}_element_overlay.png')
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"  > Saved: {item}_element_overlay.png")


def create_master_csv(output_folder, all_dat_elem, file_names_map):
    """
    Create master CSV with all classified atoms

    Args:
        output_folder: path to save CSV
        all_dat_elem: list of [y, x, intensity_scaled, element, file_id]
        file_names_map: dict mapping file_id to file_name

    Returns:
        path to saved CSV
    """
    print(f"\n  > Creating master CSV...")

    csv_path = os.path.join(output_folder, "all_atoms_classified.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow([
            'File_Name',
            'Atom_ID',
            'Y_Position_px',
            'X_Position_px',
            'Element_Number'
        ])

        # Group data by file_id to assign atom IDs
        from collections import defaultdict
        atoms_by_file = defaultdict(list)

        for atom_data in all_dat_elem:
            y, x, intensity_scaled, element, file_id = atom_data
            atoms_by_file[file_id].append([y, x, element])

        # Write data rows
        for file_id in sorted(atoms_by_file.keys()):
            file_name = file_names_map.get(file_id, f"Unknown_{file_id}")
            atoms = atoms_by_file[file_id]

            for atom_id, (y, x, element) in enumerate(atoms, start=1):
                writer.writerow([
                    file_name,
                    atom_id,
                    f'{y:.2f}',
                    f'{x:.2f}',
                    element
                ])

    print(f"  > Saved: all_atoms_classified.csv")
    return csv_path


def classify_elements(atoms, normalized, Aatoms, Aint, Bint, poptA, poptB, file_id):
    """
    Classify atoms into elements based on intensity

    Returns:
        element_list: list of element classifications
        dat_elem: list of [y, x, int_sc, element, file_id]
    """
    element_list = []
    dat_elem = []

    for atom in atoms:
        pos = [atom[0], atom[1]]
        intensity = normalized[int(pos[0]), int(pos[1])]

        if pos in Aatoms:
            # A-site classification (2 peaks)
            int_sc = (intensity - min(Aint)) / (max(Aint) - min(Aint))
            z_score = [
                abs(int_sc - poptA[0]) / poptA[1],
                abs(int_sc - poptA[2]) / poptA[3]
            ]
            min_index = z_score.index(min(z_score))
            element = 2 if min_index == 0 else 1
            element_list.append(element)
            dat_elem.append([int(pos[0]), int(pos[1]), int_sc, element, file_id])
        else:
            # B-site classification (3 peaks)
            int_sc = (intensity - min(Bint)) / (max(Bint) - min(Bint))
            z_score = [
                abs(int_sc - poptB[0]) / poptB[1],
                abs(int_sc - poptB[2]) / poptB[3],
                abs(int_sc - poptB[4]) / poptB[5]
            ]
            min_index = z_score.index(min(z_score))
            if min_index == 0:
                element = 5
            elif min_index == 1:
                element = 4
            else:
                element = 3
            element_list.append(element)
            dat_elem.append([int(pos[0]), int(pos[1]), int_sc, element, file_id])

    return element_list, dat_elem


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_file(item, input_folder, output_folder, file_id,
                        poptA=None, poptB=None, use_global_fit=False):
    """
    Process a single file

    Args:
        item: filename (without extension)
        input_folder: path to input folder
        output_folder: path to output folder
        file_id: unique identifier for this file
        poptA: fitted parameters for A-site (if using global fit)
        poptB: fitted parameters for B-site (if using global fit)
        use_global_fit: whether to use global fit parameters

    Returns:
        dat_elem: list of atom data with classifications
        Aint_scaled: scaled A-site intensities
        Bint_scaled: scaled B-site intensities
        cal: calibration in nm/pixel
        cal_pm: calibration in pm/pixel
        original_image: original grayscale image for overlay
        atoms: list of atom positions
        element_list: list of element classifications
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {item} (file_id: {file_id})")
    print(f"{'=' * 70}")

    folder_path = os.path.join(input_folder, item)

    # Get calibration from image metadata
    cal, cal_pm = get_calibration_from_image(folder_path, item)

    # Load image
    file_path = os.path.join(folder_path, item + "_original.png")
    print(f"  > Loading image: {file_path}")
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store original for overlay
    original_image = gray.copy()

    # Normalize image
    print(f"  > Normalizing image...")
    img_min, img_max = gray.min(), gray.max()
    img_norm = ((gray - img_min) / (img_max - img_min))

    # Apply local normalization
    normalized = normalize_image_locally(img_norm, mask_size=64)

    # Load atom positions from CSV
    csv_path = os.path.join(folder_path, item + "_atom_data.csv")
    print(f"  > Loading atom positions from: {csv_path}")
    df = pd.read_csv(csv_path)

    atoms = []
    for i in range(df.shape[0]):
        pos = [df.iloc[i, 1], df.iloc[i, 2]]
        atoms.append(pos)

    print(f"  > Loaded {len(atoms)} atoms")

    # Sort atoms into A and B sublattices
    print(f"  > Sorting atoms into sublattices...")
    Aatoms, Batoms = sort_atoms_into_sublattices(atoms, threshold=T)
    print(f"  > A-site atoms: {len(Aatoms)}, B-site atoms: {len(Batoms)}")

    # Extract intensities
    Aint, Bint = extract_intensities(normalized, Aatoms, Batoms)

    # Scale intensities
    Aint_scaled = [(x - min(Aint)) / (max(Aint) - min(Aint)) for x in Aint]
    Bint_scaled = [(x - min(Bint)) / (max(Bint) - min(Bint)) for x in Bint]

    # Fit histograms if not using global fit
    if not use_global_fit:
        print(f"  > Fitting A-site histogram (2 peaks with FIXED means)...")
        p0_A = [
            0.1, 0.3, 0.1,  # Peak 1: lower intensity (fixed at 0.3)
            0.05, 0.8, 0.1  # Peak 2: higher intensity (fixed at 0.8)
        ]
        poptA, pcovA, bin_centers_A, counts_A = fit_histogram(
            Aint_scaled, 2, p0_A, fix_means=True
        )

        print(f"  > Fitting B-site histogram (3 peaks)...")
        p0_B = [
            0.1, 0.2, 0.1,  # Peak 1: low intensity
            0.1, 0.4, 0.1,  # Peak 2: mid intensity
            0.05, 0.6, 0.1  # Peak 3: high intensity
        ]
        poptB, pcovB, bin_centers_B, counts_B = fit_histogram(
            Bint_scaled, 3, p0_B
        )

        # Plot results
        plot_fit_results(bin_centers_A, counts_A, poptA,
                         f'A-site Histogram - {item}', 2)
        plot_fit_results(bin_centers_B, counts_B, poptB,
                         f'B-site Histogram - {item}', 3)
    else:
        print(f"  > Using global fit parameters")

    # Extract parameters for classification
    poptA_params = [poptA[1], poptA[2], poptA[4], poptA[5]]
    poptB_params = [poptB[1], poptB[2], poptB[4], poptB[5], poptB[7], poptB[8]]

    # Classify elements
    print(f"  > Classifying elements...")
    element_list, dat_elem = classify_elements(
        atoms, normalized, Aatoms, Aint, Bint,
        poptA_params, poptB_params, file_id
    )

    # Print element counts
    print(f"\n  Element counts:")
    for elem_num in range(1, 6):
        count = element_list.count(elem_num)
        print(f"    Element {elem_num}: {count}")

    # Save results to CSV
    df['Element'] = element_list
    df['File_ID'] = file_id
    df.to_csv(csv_path, index=False)
    print(f"  > Saved updated CSV to: {csv_path}")

    # Create element overlay visualization
    create_element_overlay(folder_path, item, original_image, atoms, element_list, file_id)

    # Visualize detected atoms (save instead of show)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=FIGURE_DPI)
    ax.imshow(normalized, cmap='gray')
    for a in atoms:
        ax.add_patch(Circle((a[1], a[0]), radius=4, color='red', fill=False))
    ax.set_title(f"{item} - {len(atoms)} Atoms Detected")
    ax.axis('off')
    plt.tight_layout()
    detection_path = os.path.join(folder_path, f'{item}_atoms_detected.png')
    plt.savefig(detection_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  > Saved atoms detection plot: {item}_atoms_detected.png")

    return dat_elem, Aint_scaled, Bint_scaled, cal, cal_pm, original_image, atoms, element_list


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "=" * 70)
    print("ATOM ANALYSIS PIPELINE - MULTI-FILE PROCESSING")
    print("=" * 70)

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")

    # Determine which files to use for fitting
    if not CSV_FILES_FOR_FITTING or len(CSV_FILES_FOR_FITTING) == 0:
        # Use all files for fitting
        files_for_fitting = FILES_TO_PROCESS
        print(f"\nUsing all {len(files_for_fitting)} files for global fitting")
    else:
        # Use specified subset
        files_for_fitting = CSV_FILES_FOR_FITTING
        print(f"\nUsing {len(files_for_fitting)} specified files for global fitting")

    # Step 1: Aggregate data from selected CSV files for global fitting
    print("\n" + "=" * 70)
    print("STEP 1: Aggregating data for global fitting")
    print("=" * 70)

    all_Aint = []
    all_Bint = []

    for item in files_for_fitting:
        # Construct CSV path automatically
        csv_path = os.path.join(INPUT_FOLDER, item, f"{item}_atom_data.csv")
        print(f"\n  > Reading: {csv_path}")

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"    WARNING: File not found, skipping")
            continue

        # Read CSV and get atom positions
        df = pd.read_csv(csv_path)
        atoms = [[df.iloc[i, 1], df.iloc[i, 2]] for i in range(df.shape[0])]

        # Load corresponding image
        folder_path = os.path.join(INPUT_FOLDER, item)

        # Get calibration automatically
        cal, cal_pm = get_calibration_from_image(folder_path, item)

        # Path to the original PNG saved by batch processing
        file_path = os.path.join(folder_path, f"{item}_original.png")

        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_min, img_max = gray.min(), gray.max()
        img_norm = ((gray - img_min) / (img_max - img_min))
        normalized = normalize_image_locally(img_norm, mask_size=64)

        # Sort atoms and extract intensities
        Aatoms, Batoms = sort_atoms_into_sublattices(atoms, threshold=T)
        Aint, Bint = extract_intensities(normalized, Aatoms, Batoms)

        # Scale and aggregate
        Aint_scaled = [(x - min(Aint)) / (max(Aint) - min(Aint)) for x in Aint]
        Bint_scaled = [(x - min(Bint)) / (max(Bint) - min(Bint)) for x in Bint]

        all_Aint.extend(Aint_scaled)
        all_Bint.extend(Bint_scaled)

        print(f"    Added {len(Aint_scaled)} A-site and {len(Bint_scaled)} B-site intensities")

    print(f"\n  > Total aggregated: {len(all_Aint)} A-site, {len(all_Bint)} B-site intensities")

    # Step 2: Fit global histograms
    print("\n" + "=" * 70)
    print("STEP 2: Fitting global histograms")
    print("=" * 70)

    print(f"\n  > Fitting global A-site histogram (2 peaks)...")
    # A-site with FIXED means at 0.3 and 0.8
    p0_A = [
        0.1, 0.3, 0.1,  # Peak 1: amp, mean (will be fixed at 0.3), sigma
        0.05, 0.8, 0.1  # Peak 2: amp, mean (will be fixed at 0.8), sigma
    ]
    poptA_global, pcovA_global, bin_centers_A, counts_A = fit_histogram(
        all_Aint, 2, p0_A, fix_means=True
    )
    print(f"    A-site peaks FIXED at means: 0.3 and 0.8")

    print(f"\n  > Fitting global B-site histogram (3 peaks)...")
    # Better initial guesses for B-site: three well-separated peaks
    p0_B = [
        0.1, 0.2, 0.1,  # Peak 1: low intensity
        0.1, 0.4, 0.1,  # Peak 2: mid intensity
        0.05, 0.6, 0.1  # Peak 3: high intensity
    ]
    poptB_global, pcovB_global, bin_centers_B, counts_B = fit_histogram(
        all_Bint, 3, p0_B
    )

    # Plot global fits (save instead of show)
    print(f"\n  > Saving global fit visualizations...")

    # Save A-site fit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    bin_edges_A = np.concatenate([[bin_centers_A[0] - (bin_centers_A[1] - bin_centers_A[0]) / 2],
                                  (bin_centers_A[:-1] + bin_centers_A[1:]) / 2,
                                  [bin_centers_A[-1] + (bin_centers_A[-1] - bin_centers_A[-2]) / 2]])
    width_A = bin_edges_A[1] - bin_edges_A[0]

    ax1.bar(bin_centers_A, counts_A, width=width_A, alpha=0.5,
            label='Histogram data', color='gray')
    ax1.plot(bin_centers_A, multi_gaussian(bin_centers_A, *poptA_global),
             'r-', linewidth=2, label='Combined fit')

    colors = ['blue', 'green', 'orange', 'purple', 'cyan']
    for i in range(2):
        gauss = gaussian(bin_centers_A, poptA_global[i * 3], poptA_global[i * 3 + 1], poptA_global[i * 3 + 2])
        ax1.plot(bin_centers_A, gauss, '--', linewidth=2,
                 color=colors[i % len(colors)], label=f'Peak {i + 1}')

    ax1.set_xlabel('Normalized Intensity', fontsize=12)
    ax1.set_ylabel('Normalized Counts', fontsize=12)
    ax1.set_title('Global A-site Histogram Fit', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fitted = multi_gaussian(bin_centers_A, *poptA_global)
    residuals = counts_A - fitted
    ax2.bar(bin_centers_A, residuals, width=width_A, color='red', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Normalized Intensity', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    asite_path = os.path.join(OUTPUT_FOLDER, 'global_Asite_fit.png')
    plt.savefig(asite_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: global_Asite_fit.png")

    # Save B-site fit
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    bin_edges_B = np.concatenate([[bin_centers_B[0] - (bin_centers_B[1] - bin_centers_B[0]) / 2],
                                  (bin_centers_B[:-1] + bin_centers_B[1:]) / 2,
                                  [bin_centers_B[-1] + (bin_centers_B[-1] - bin_centers_B[-2]) / 2]])
    width_B = bin_edges_B[1] - bin_edges_B[0]

    ax1.bar(bin_centers_B, counts_B, width=width_B, alpha=0.5,
            label='Histogram data', color='gray')
    ax1.plot(bin_centers_B, multi_gaussian(bin_centers_B, *poptB_global),
             'r-', linewidth=2, label='Combined fit')

    for i in range(3):
        gauss = gaussian(bin_centers_B, poptB_global[i * 3], poptB_global[i * 3 + 1], poptB_global[i * 3 + 2])
        ax1.plot(bin_centers_B, gauss, '--', linewidth=2,
                 color=colors[i % len(colors)], label=f'Peak {i + 1}')

    ax1.set_xlabel('Normalized Intensity', fontsize=12)
    ax1.set_ylabel('Normalized Counts', fontsize=12)
    ax1.set_title('Global B-site Histogram Fit', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fitted = multi_gaussian(bin_centers_B, *poptB_global)
    residuals = counts_B - fitted
    ax2.bar(bin_centers_B, residuals, width=width_B, color='red', alpha=0.6)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Normalized Intensity', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    bsite_path = os.path.join(OUTPUT_FOLDER, 'global_Bsite_fit.png')
    plt.savefig(bsite_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"    Saved: global_Bsite_fit.png")

    # Step 3: Process all files using global fit
    print("\n" + "=" * 70)
    print("STEP 3: Processing individual files with global fit")
    print("=" * 70)

    all_dat_elem = []
    file_names_map = {}  # Map file_id to file_name

    for file_id, item in enumerate(FILES_TO_PROCESS, start=1):
        file_names_map[file_id] = item
        dat_elem, _, _, cal, cal_pm, _, _, _ = process_single_file(
            item, INPUT_FOLDER, OUTPUT_FOLDER, file_id,
            poptA=poptA_global, poptB=poptB_global, use_global_fit=True
        )
        all_dat_elem.extend(dat_elem)

    # Step 4: Save aggregated results
    print("\n" + "=" * 70)
    print("STEP 4: Saving aggregated results")
    print("=" * 70)

    # Create master CSV with file names
    master_csv_path = create_master_csv(OUTPUT_FOLDER, all_dat_elem, file_names_map)

    print(f"  > Saved master CSV to: {master_csv_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal atoms processed: {len(all_dat_elem)}")
    print(f"\nElement distribution:")

    # Count elements from all_dat_elem
    element_counts = {i: 0 for i in range(1, 6)}
    for atom_data in all_dat_elem:
        element = atom_data[3]  # Element is at index 3
        element_counts[element] = element_counts.get(element, 0) + 1

    for elem_num in range(1, 6):
        count = element_counts[elem_num]
        percentage = (count / len(all_dat_elem)) * 100 if len(all_dat_elem) > 0 else 0
        print(f"  Element {elem_num}: {count} ({percentage:.2f}%)")

    print(f"\nFiles processed:")
    for file_id, file_name in sorted(file_names_map.items()):
        # Count atoms in this file
        file_atoms = [d for d in all_dat_elem if d[4] == file_id]
        print(f"  [{file_id}] {file_name}: {len(file_atoms)} atoms")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - Master CSV: {master_csv_path}")
    print(f"  - Individual element overlays saved in each subfolder")
    print(f"  - Updated CSVs with Element and File_ID columns in each subfolder")


if __name__ == "__main__":
    main()