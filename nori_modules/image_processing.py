import numpy as np
import cv2 as cv
from typing import List, Dict


# Constants
BACKGROUND_INTENSITY_THRESHOLD = 50
AREA_THRESHOLD_PERCENTAGE = 0.3

def filter_tubule_masks(image: np.ndarray, masks: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Filters tubule masks based on area and average intensity constraints.

    Parameters:
    - image (np.ndarray): The input image array.
    - masks (List[Dict]): List of dictionaries containing 'area' and 'segmentation'.

    Returns:
    - List[Dict]: Filtered list of mask dictionaries.
    """
    h, w, _ = image.shape
    th_area_top = h * w * AREA_THRESHOLD_PERCENTAGE  # Upper area threshold
    th_area_bot = 0  # Lower area threshold

    filtered_mask = []
    for mask in masks:
        if th_area_bot < mask['area'] < th_area_top and image[mask['segmentation']].mean() > BACKGROUND_INTENSITY_THRESHOLD:
            filtered_mask.append(mask)
    
    return filtered_mask

def combine_masks(masks: List[Dict[str, any]], image: np.ndarray) -> np.ndarray:
    """
    Combines multiple masks into a single mask, applying erosion to each.

    Parameters:
    - masks (List[Dict]): List of mask dictionaries.
    - image (np.ndarray): The input image array for shape reference.

    Returns:
    - np.ndarray: Combined binary mask.
    """
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        segmentation = (255 * mask['segmentation'].astype(np.uint8))
        eroded = erode_tubule(segmentation)
        combined = np.logical_or(combined, eroded)

    return combined.astype(np.uint8)

def erode_tubule(segmentation: np.ndarray) -> np.ndarray:
    """
    Applies erosion to a binary mask segmentation.

    Parameters:
    - segmentation (np.ndarray): Binary mask segmentation.

    Returns:
    - np.ndarray: Eroded binary mask.
    """
    se = create_circular_se(5)
    return cv.erode(segmentation, se)

def create_circular_se(radius: int) -> np.ndarray:
    """
    Creates a circular structuring element.

    Parameters:
    - radius (int): The radius of the circular structuring element.

    Returns:
    - np.ndarray: The structuring element.
    """
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def combine_input_channels(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Preprocess two images by element-wise multiplication, normalization, and histogram equalization.

    Parameters:
    - image1 (np.ndarray): First input image.
    - image2 (np.ndarray): Second input image.

    Returns:
    - np.ndarray: Preprocessed image.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Multiply images element-wise in float32 type for precision
    result = np.multiply(image1, image2, dtype='float32')

    # Normalize result to scale between 0-255
    scaled = normalize_intensity_levels(result)

    # Apply histogram equalization to improve image contrast
    equalized = cv.equalizeHist(scaled)

    return equalized

def normalize_intensity_levels(image: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity levels of an image to fit within the 0-255 range.

    Parameters:
    - image (np.ndarray): Image to normalize.

    Returns:
    - np.ndarray: Normalized image with intensity scaled to 8-bit.
    """
    min_val, max_val = image.min(), image.max()
    if max_val - min_val == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)

