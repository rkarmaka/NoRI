import numpy as np
import cv2 as cv
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def validate_mask(mask: np.ndarray) -> np.ndarray:
    """
    Checks if the mask is boolian or not and then converts it to boolean if not.
    Parametes:
    - mask: np.ndarray - Binary mask where non-zero values indicate the region of interest.

    Return:
    - mask: bool - Return the same mask as boolean
    """
    if mask.dtype != bool:
        mask = mask > 0

    return mask

def measure_intensity(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Measure the total, mean, and standard deviation of intensity in the masked area of the image.

    Parameters:
    - image: np.ndarray - Input image.
    - mask: np.ndarray - Binary mask where non-zero values indicate the region of interest.

    Returns:
    - total_intensity: float - Sum of the pixel intensities in the masked area.
    - mean_intensity: float - Mean of the pixel intensities in the masked area.
    - std_intensity: float - Standard deviation of the pixel intensities in the masked area.
    """
    if mask.sum() != 0:
        mask = validate_mask(mask)
        
        total_intensity = float(image[mask].sum())
        mean_intensity = float(image[mask].mean())
        std_intensity = float(image[mask].std())
    else:
        total_intensity = 0.0
        mean_intensity = 0.0
        std_intensity = 0.0

    # logging.info(f"Measured intensity - Total: {total_intensity}, Mean: {mean_intensity}, Std Dev: {std_intensity}")
    return total_intensity, np.round(mean_intensity, 4), np.round(std_intensity, 4)

def measure_content(image: np.ndarray, mask: np.ndarray, constant: float) -> Tuple[float, float]:
    """
    Measure the total and mean content based on a provided constant in the masked area of the image.

    Parameters:
    - image: np.ndarray - Input image.
    - mask: np.ndarray - Binary mask where non-zero values indicate the region of interest.
    - constant: float - A constant used in the content calculation.

    Returns:
    - total_content: float - Total content in the masked area.
    - mean_content: float - Mean content in the masked area.
    """
    if mask.sum() != 0:
        mask = validate_mask(mask)
        
        total_content = float(((image[mask] * constant * 1000) / 8192).sum())
        mean_content = float(((image[mask] * constant * 1000) / 8192).mean())
    else:
        total_content = 0.0
        mean_content = 0.0

    # logging.info(f"Measured content - Total: {total_content}, Mean: {mean_content}")
    return total_content, mean_content

def measure_nuclei_intensity(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, int]:
    """
    Measure the total, mean, and standard deviation of intensity in the masked area of the image,
    and count the number of nuclei.

    Parameters:
    - image: np.ndarray - Input image.
    - mask: np.ndarray - Binary mask where non-zero values indicate the region of interest.

    Returns:
    - total_intensity: float - Sum of the pixel intensities in the masked area.
    - mean_intensity: float - Mean of the pixel intensities in the masked area.
    - std_intensity: float - Standard deviation of the pixel intensities in the masked area.
    - count: int - Number of nuclei detected in the masked area.
    """
    mask = validate_mask(mask)
    
    contours, _ = cv.findContours((255 * mask.astype('uint8')), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    
    if count != 0:
        # total_intensity = float(image[mask].sum())
        mean_intensity = float(image[mask].mean())
        std_intensity = float(image[mask].std() / np.sqrt(count))
    else:
        # total_intensity = 0.0
        mean_intensity = 0.0
        std_intensity = 0.0

    # logging.info(f"Measured nuclei intensity - Mean: {mean_intensity}, Std Dev: {std_intensity}, Count: {count}")
    return np.round(mean_intensity, 4), np.round(std_intensity, 4), count
