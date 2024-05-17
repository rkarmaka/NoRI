import os
import numpy as np
import cv2 as cv
import pandas as pd
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PADDING = 20


def create_directory(directory_path: str) -> None:
    """
    Creates a directory if it does not exist.

    Args:
        directory_path (str): Path of the directory to be created.
    """
    try:
        os.makedirs(directory_path)
        logging.info(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        logging.warning(f"Directory '{directory_path}' already exists.")
    except OSError as e:
        logging.error(f"Error creating directory {directory_path}: {e}")

def read_file_names(root_folder: str, file_type=0) -> List[str]:
    """
    Returns a list of full paths for TIFF files within a given directory.

    Args:
        root_folder (str): Root directory to search for TIFF files.

    Returns:
        List[str]: List of file paths.
    """
    if file_type==0:
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_folder)
            for file in files if file.lower().endswith(('.tif', '.tiff'))
        ]
        logging.info(f"Found {len(image_files)} TIFF files in {root_folder}.")
    elif file_type==1:
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_folder)
            for file in files if file.lower().endswith(('.png'))
        ]
        logging.info(f"Found {len(image_files)} PNG files in {root_folder}.")
    return image_files

def transpose_input_image(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Transpose the image if its height is greater than its width.

    Args:
        image (np.ndarray): The image array.

    Returns:
        Tuple[np.ndarray, bool]: (Transposed image, True if transposed, otherwise False).
    """
    h, w, _ = image.shape
    if h > w:
        logging.debug("Transposing image because height > width.")
        return np.transpose(image, (1, 0, 2)), True
    else:
        return image, False

def pad_image(image: np.ndarray, tile_shape: Tuple[int, int], stride: int) -> np.ndarray:
    """
    Pads the image to fit the tile shape based on the stride.

    Args:
        image (np.ndarray): Image to pad.
        tile_shape (Tuple[int, int]): Height and width of the tile.
        stride (int): Stride for tiling.

    Returns:
        np.ndarray: Padded image.
    """
    image_h, image_w, image_c = image.shape
    tile_h, tile_w = tile_shape
    w_new = int(stride * np.ceil((image_w - tile_w) / stride)) + tile_w
    new_image = np.zeros((image_h, w_new, image_c), dtype=image.dtype)
    new_image[:image_h, :image_w, :] = image
    logging.debug(f"Padded image to new width {w_new}.")
    return new_image







def extract_tubule(original_image, contour, binary_mask, model=None, save_tubule=False):
    """
    Extracts a tubule from the original image based on the provided contour and binary mask.

    Args:
        original_image (numpy.ndarray): Original image.
        contour: Contour of the tubule.
        binary_mask: Binary mask of the tubule.
        model: Model for additional processing (optional).
        save_tubule (bool): Whether to save the extracted tubule.

    Returns:
        tuple: Tuple containing the cropped tubule, mask, and bounding box (x, y, w, h).
    """
    cell_mask = np.zeros_like(binary_mask)
    cv.drawContours(cell_mask, [contour], 0, 255, thickness=cv.FILLED)
    k = create_circular_se(5)
    cell_mask = cv.dilate(cell_mask, kernel=k)

    # Crop the region from the original image using the mask
    cell_cropped = cv.bitwise_and(original_image, original_image, mask=cell_mask)

    # Get the bounding box of the contour to determine the filename
    x, y, w, h = cv.boundingRect(contour)

    x = np.max((0,x+PADDING))
    y = np.max((0,y+PADDING))
    w+=2*PADDING
    h+=2*PADDING
    
    return cell_cropped[y:y + h, x:x + w], cell_mask[y:y + h, x:x + w], x, y, w, h



def get_centroid(contour):
    """
    Calculates the centroid of a contour.

    Args:
        contour: Contour of an object.

    Returns:
        tuple: Centroid coordinates (cx, cy).
    """
    moment = cv.moments(contour)

    # Calculate centroid coordinates
    if moment['m00']!=0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
    else:
        cx, cy = 0, 0

    return cx, cy



def extract_cyto_only_mask(main_mask, nuclei_mask, lumen_mask, bb_mask):
    """
    Extract the cytoplasm-only mask by combining and subtracting object masks.

    Parameters:
    - main_mask (numpy.ndarray): Main segmentation mask.
    - nuclei_mask (numpy.ndarray): Nuclei segmentation mask.
    - lumen_mask (numpy.ndarray): Lumen segmentation mask.
    - bb_mask (numpy.ndarray): Brushed border segmentation mask.

    Returns:
    - numpy.ndarray: Cytoplasm-only segmentation mask.
    """

    # Combine inverted object masks
    combined_obj_mask = 255*(np.logical_or.reduce([nuclei_mask, lumen_mask, bb_mask]).astype(np.uint8))

    # Remove areas from the main mask where there are objects in the other three masks
    final_mask = np.subtract(main_mask, combined_obj_mask)>1


    return final_mask


def create_circular_se(radius):
    """
    Create a circular kernel for morphological operations.

    Parameters:
    - radius (int): Radius of the circular kernel.

    Returns:
    - numpy.ndarray: Circular kernel.
    """
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))