import cv2 as cv
import numpy as np



def preprocess_images(image1, image2):
    """
    Preprocess two images by element-wise multiplication, normalization, and histogram equalization.

    Parameters:
    - image1 (numpy.ndarray): First input image.
    - image2 (numpy.ndarray): Second input image.

    Returns:
    - numpy.ndarray: Preprocessed image.
    """
    # Ensure the input images have the same size
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Convert the input images to uint32 format by multiplying them
    result = np.multiply(image1, image2, dtype='float32')

    # Scale the image to 8-bit
    scaled = normalize_intensity_levels(result)

    # Histogram equalization
    equalized = cv.equalizeHist(scaled)

    return cv.equalizeHist(equalized)

def normalize_intensity_levels(image, max=255):
    """
    Normalize the intensity levels of an image to the range [0, max].

    Parameters:
    - image (numpy.ndarray): Input image.
    - max (int): Maximum intensity value (default is 255).

    Returns:
    - numpy.ndarray: Normalized image.
    """
    return (((image - image.min()) / (image.max() - image.min())) * max).astype(np.uint8)

