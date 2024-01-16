import os
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2 as cv

def create_directory(directory_path):
    """
    Creates a directory if it does not exist.

    Args:
        directory_path (str): Path of the directory to be created.

    Returns:
        None
    """
    try:
        # Attempt to create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")


def read_tiff_and_extract_channels(file_path, separate_channels=True):
    """
    Reads a TIFF file, extracts protein and lipid channels, and returns 2 image stacks with image patches.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        list: List containing two image channels.
    """
    try:
        # Read TIFF file. Image is read in (c x h x w) format
        image = imread(file_path)


        if separate_channels:
            # Check the number of channels in the image
            if image.shape[0] == 7:
                print("Image has 7 channels")
                protein_channel = image[1, :, :]
                lipid_channel = image[2, :, :]
                brush_border_marker = image[5, :, :]

                return [protein_channel, lipid_channel, brush_border_marker]
            elif image.shape[0] == 6:
                print("Image has 6 channels")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]
                brush_border_marker = image[4, :, :]

                return [protein_channel, lipid_channel, brush_border_marker]
            elif image.shape[0] == 3:
                print("Image has 3 channels. Assigning Ch1 as Protein and Ch2 as Lipid")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]

                return [protein_channel, lipid_channel]
            else:
                print(f"Image has {image.shape[0]} channels")
                return None
        else:
            return image

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None



def display_image(image, fig_size=(20, 10), overlay=False, alpha=0.2):
    """
    Displays the given image.

    Args:
        img (numpy.ndarray): Image data.
        fig_size (tuple): Figure size.
        overlay (bool): Whether to overlay images.
        alpha (float): Transparency for overlay.

    Returns:
        None
    """
    if overlay:
        plt.figure(figsize=fig_size)
        plt.imshow(image[0], cmap='gray')
        plt.imshow(image[1], cmap='jet', alpha=alpha)
        plt.axis('off')
    else:
        plt.figure(figsize=fig_size)
        plt.imshow(image, cmap='gray')
        plt.axis('off')


def image_scaling(image, max=255):
    """
    Scale the intensity levels of an image to the range [0, max].

    Parameters:
    - image (numpy.ndarray): Input image.
    - max (int): Maximum intensity value (default is 255).

    Returns:
    - numpy.ndarray: Scaled image.
    """
    if image.size == 0:
        print("Error: Empty image provided.")
        return None  # You can customize this return value based on your needs
    else:
        return (((image - image.min()) / (image.max() - image.min())) * max).astype(np.uint8)

def gray_to_rgb(image):
    """
    Convert a grayscale image to an RGB image.

    Parameters:
    - image (numpy.ndarray): Input grayscale image.

    Returns:
    - numpy.ndarray: RGB image.
    """
    if len(image.shape) == 2:
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            rgb_image[:, :, i] = image_scaling(image)
    else:
        rgb_image = image

    return rgb_image


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

    # Crop the region from the original image using the mask
    cell_cropped = cv.bitwise_and(original_image, original_image, mask=cell_mask)

    # Get the bounding box of the contour to determine the filename
    x, y, w, h = cv.boundingRect(contour)

    return cell_cropped[y:y + h, x:x + w], cell_mask[y:y + h, x:x + w], x, y, w, h


def create_rgb_image(image1, image2, image3):
    """
    Creates an RGB image from three input images.

    Args:
        image1, image2, image3 (numpy.ndarray): Input images.

    Returns:
        numpy.ndarray: RGB image.
    """
    rgb_image = np.stack([image1, image2, image3], axis=-1)

    return image_scaling(rgb_image)


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


def euclidean_distance(centroid1, centroid2):
    """
    Calculates the Euclidean distance between two centroids.

    Args:
        centroid1, centroid2 (tuple): Centroid coordinates.

    Returns:
        float: Euclidean distance.
    """
    return distance.euclidean(centroid1, centroid2)


def tile_image(image, tile_shape, stride):
    """
    Tile an input image with a given shape and stride.

    Parameters:
    - image (numpy.ndarray): Input image to be tiled.
    - tile_shape (tuple): Shape of the tiles (height, width).
    - stride (int): Stride for tiling.

    Returns:
    - numpy.ndarray: Array of tiled images with shape (tile_height, tile_width, num_tiles).
    """

    # Pad the input image
    padded = pad_image(image, tile_shape, stride)

    # Calculate the number of rows and columns for the tiles
    rows = (padded.shape[0] - tile_shape[0]) // stride + 1
    cols = (padded.shape[1] - tile_shape[1]) // stride + 1

    # Initialize an empty array to store the tiles
    tiles = np.zeros((tile_shape[0], tile_shape[1], (rows*cols)))

    # Extract the tiles from the padded image
    for i in range(0, rows):
        for j in range(0, cols):
            start_x = j * stride
            start_y = i * stride
            end_x = start_x + tile_shape[1]
            end_y = start_y + tile_shape[0]

            ch = (i * rows) + j
            tiles[:, :, ch] = padded[start_y:end_y, start_x:end_x]

    return tiles.astype(np.uint8)


def pad_image(image, tile_shape, stride):
    """
    Pad an image to make its dimensions multiples of the tile shape.

    Parameters:
    - image (numpy.ndarray): Input image to be padded.
    - tile_shape (tuple): Shape of the tiles used for processing, e.g., (tile_height, tile_width).
    - stride (int): Stride used for processing (not directly used in this function).

    Returns:
    - numpy.ndarray: Padded image with dimensions multiples of the tile shape.
    """

    # Get the dimensions of the original image
    height, width = image.shape

    # Calculate the padding required to make the image dimensions multiples of the tile shape
    if (height % tile_shape[0]) != 0:
        pad_height = tile_shape[0] - (height % tile_shape[0])
    else:
        pad_height = 0

    if (width % tile_shape[1]) != 0:
        pad_width = tile_shape[1] - (width % tile_shape[1])
    else:
        pad_width = 0

    # Pad the image with zeros
    image_padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')

    return image_padded


def label_tubules(mask, contour, idx):
    """
    Label a tubule in the provided mask with its index and centroid coordinates.

    Parameters:
    - mask (numpy.ndarray): Binary mask where the tubule is labeled.
    - contour (numpy.ndarray): Contour of the tubule.
    - idx (int): Index of the tubule.

    Returns:
    - numpy.ndarray: Updated mask with the tubule labeled.
    """

    # Get centroid coordinates of the tubule contour
    cx, cy = get_centroid(contour=contour)
    
    # Write label at the centroid
    label = f"{idx}"
    cv.putText(mask, label, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv.LINE_AA)
    
    return mask

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
    combined_obj_mask = np.logical_or.reduce([nuclei_mask, lumen_mask, bb_mask])

    # Remove areas from the main mask where there are objects in the other three masks
    final_mask = np.logical_xor(main_mask, combined_obj_mask)

    return final_mask
