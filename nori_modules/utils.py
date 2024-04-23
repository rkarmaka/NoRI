import os
import numpy as np
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def read_tiff_file_names(root_folder: str) -> List[str]:
    """
    Returns a list of full paths for TIFF files within a given directory.

    Args:
        root_folder (str): Root directory to search for TIFF files.

    Returns:
        List[str]: List of file paths.
    """
    tiff_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(root_folder)
        for file in files if file.lower().endswith(('.tif', '.tiff'))
    ]
    logging.info(f"Found {len(tiff_files)} TIFF files in {root_folder}.")
    return tiff_files

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