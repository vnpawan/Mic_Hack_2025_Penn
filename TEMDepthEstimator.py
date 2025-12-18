import numpy as np
from scipy import ndimage
import cv2
# import torch

class TEMDepthEstimator:
    def __init__(self):
        """
        Initialize the TEM image depth estimator with specialized processing
        for electron microscopy images.
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_tem_image(self, image):
        """
        Preprocess TEM image for better depth estimation.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 2:
            image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = clahe.apply(image) / 255.0

        return image

    def guided_filter(self, guide, src, radius, eps):
        """Implementation of guided filter without requiring OpenCV contrib"""
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)

        kernel = np.ones((2 * radius + 1, 2 * radius + 1)) / ((2 * radius + 1) ** 2)
        mean_guide = ndimage.convolve(guide, kernel)
        mean_src = ndimage.convolve(src, kernel)
        mean_guide_src = ndimage.convolve(guide * src, kernel)
        mean_guide_guide = ndimage.convolve(guide * guide, kernel)

        var_guide = mean_guide_guide - mean_guide * mean_guide
        cov_guide_src = mean_guide_src - mean_guide * mean_src

        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        mean_a = ndimage.convolve(a, kernel)
        mean_b = ndimage.convolve(b, kernel)

        return mean_a * guide + mean_b

    def estimate_depth_from_intensity(self, image):
        """
        Edge-preserving smoothing approach using guided filter.
        Good for maintaining subtle details while reducing artifacts.
        """
        depth_map = 1.0 - image

        # Multi-scale guided filtering
        r_values = [2, 4, 8]
        eps_values = [0.1 ** 2, 0.2 ** 2, 0.4 ** 2]

        filtered_depth = np.zeros_like(depth_map)
        for r, eps in zip(r_values, eps_values):
            filtered = self.guided_filter(
                guide=depth_map,
                src=depth_map,
                radius=r,
                eps=eps
            )
            filtered_depth += filtered

        filtered_depth /= len(r_values)

        # Calculate gradient component
        gx = ndimage.gaussian_filter(filtered_depth, sigma=1.0, order=(1, 0))
        gy = ndimage.gaussian_filter(filtered_depth, sigma=1.0, order=(0, 1))
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        gradient = np.tanh(gradient)

        depth_map = 0.95 * filtered_depth + 0.05 * gradient

        return cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    def filter_atoms_by_depth_contours(self, image_enhanced, atoms, threshold=0.85, min_area_percentage=2.0):
        """
        Filter atoms by removing those that fall within depth-qualified contour regions.

        Args:
            image_enhanced: Preprocessed TEM image (grayscale, float32, 0-1 range)
            atoms: List/array of atom coordinates [(x1, y1), (x2, y2), ...]
            threshold: Depth threshold value (default 0.85)
            min_area_percentage: Minimum contour area as percentage of total image area (default 2.0)

        Returns:
            filtered_atoms: List of atom coordinates that survived filtering
            num_qualified_contours: Number of contours used for filtering
            total_qualified_area: Total pixel area of qualified contours
        """
        print(f"    > Generating depth map for atom filtering...")

        # Step 1: Generate depth map
        depth_map = self.estimate_depth_from_intensity(image_enhanced)

        # Step 2: Apply threshold to create binary image
        binary = (depth_map > threshold).astype(np.uint8) * 255

        # Step 3: Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Calculate minimum area threshold
        total_area = image_enhanced.shape[0] * image_enhanced.shape[1]
        min_area_pixels = (min_area_percentage / 100.0) * total_area

        print(f"    > Total image area: {total_area} pixels")
        print(f"    > Minimum contour area: {min_area_pixels:.0f} pixels ({min_area_percentage}%)")

        # Step 5: Filter contours by area
        qualified_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area_pixels]

        print(f"    > Found {len(contours)} total contours")
        print(f"    > Qualified contours (>= min area): {len(qualified_contours)}")

        # Step 6: Create binary mask (white background, black contours)
        mask = np.ones(image_enhanced.shape[:2], dtype=np.uint8) * 255
        if len(qualified_contours) > 0:
            cv2.drawContours(mask, qualified_contours, -1, 0, -1)  # Fill contours with black

        # Calculate total qualified area
        total_qualified_area = sum(cv2.contourArea(cnt) for cnt in qualified_contours)

        # Step 7: Filter atoms based on mask
        filtered_atoms = []
        removed_count = 0

        for atom in atoms:
            y, x, sigma = atom  # blob_dog returns [y, x, sigma]

            # Convert to integer coordinates and ensure within bounds
            ix, iy = int(round(x)), int(round(y))

            # Check bounds
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                if mask[iy, ix] == 255:  # Keep if on white area
                    filtered_atoms.append(atom)  # Keep the full [y, x, sigma] format
                else:
                    removed_count += 1
            else:
                # Keep atoms that fall outside image bounds (edge case)
                filtered_atoms.append(atom)

        print(f"    > Atoms before filtering: {len(atoms)}")
        print(f"    > Atoms removed: {removed_count}")
        print(f"    > Atoms after filtering: {len(filtered_atoms)}")

        # Step 8: Return filtered atoms and stats
        filtered_atoms_array = np.array(filtered_atoms) if len(filtered_atoms) > 0 else np.array([]).reshape(0, 3)

        return filtered_atoms_array, len(
            qualified_contours), total_qualified_area, depth_map, binary, contours, qualified_contours