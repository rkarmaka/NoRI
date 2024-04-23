import os
import numpy as np



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



def read_tiff_file_names(root_folder):
    tiff_files = []

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                file_path = os.path.join(root, file)
                tiff_files.append(file_path)

    return tiff_files




def transpose_input_image(image):
    h,w,_ = image.shape

    if h>w:
        return np.transpose(image, (1,0,2)), True
    else:
        return image, False
    



def pad_image(image, tile_shape, stride):
    image_h, image_w, image_c = image.shape
    tile_h, tile_w = tile_shape
    w_new = int(stride*np.ceil((image_w - tile_w) / stride)) + tile_w
    new_image = np.zeros((image_h, w_new, image_c), dtype='uint8')
    
    new_image[:image_h, :image_w,:] = image

    return new_image



def intensity_scaling(image, max=255):
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
