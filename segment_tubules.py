import os
import numpy as np
import cv2 as cv
from typing import List, Tuple
from nori_modules.image_processing import combine_input_channels, normalize_intensity_levels
from nori_modules.segmentation import sam_segmentation_tiled
from nori_modules.utils import read_tiff_file_names, transpose_input_image, pad_image

import time

# Constants
ROOT_FOLDER: str = "/Users/ranit/IAC/Project/Will Trim/analysis/all_images(processed)/test/v3"
OUT_FOLDER: str = os.path.join(ROOT_FOLDER, "out")
CHECKPOINT_PATH: str = "/Users/ranit/IAC/Project/Will Trim/code/v2/sam_model/sam_vit_h_4b8939.pth"
TIFF_FILES: List[str] = read_tiff_file_names(root_folder=ROOT_FOLDER)
TILE_SHAPE: Tuple[int, int] = (1970, 2000)
STRIDE: int = 500
DEVICE: str = 'cpu'  # 'cuda' for NVIDIA GPU

def main() -> None:
    for tiff_file in TIFF_FILES:
        st_time = time.time()
        image_name: str = os.path.basename(tiff_file).split('.')[0]
        print(f'Loading image {image_name}')
        
        try:
            image = load_and_prepare_image(tiff_file)
            image, image_transposed = transpose_input_image(image)
            
            # SAM segmentation
            print('Tubule segmentation...')
            process_and_save_image(image, image_name, image_transposed)
            en_time = time.time()
            print(en_time - st_time)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

def load_and_prepare_image(tiff_file: str) -> np.ndarray:
    """Load and preprocess the TIFF file."""
    image = cv.imread(tiff_file)
    if image is None:
        raise ValueError(f"Failed to load image {tiff_file}")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image[:, :, 2] = combine_input_channels(image[:, :, 0], image[:, :, 1])
    return image

def process_and_save_image(image: np.ndarray, image_name: str, image_transposed: bool) -> None:
    """Process the image and save the segmentation result."""
    padded_image = pad_image(image, tile_shape=TILE_SHAPE, stride=STRIDE)
    
    n_tiles: int = int(np.ceil((padded_image.shape[1] - TILE_SHAPE[1]) / STRIDE)) + 1
    mask: np.ndarray = np.zeros((padded_image.shape[0], padded_image.shape[1]), dtype='uint8')

    tiles = np.zeros((*TILE_SHAPE, 3, n_tiles), dtype='uint8')
    for i in range(n_tiles):
        s_start = i * STRIDE
        s_end = s_start + TILE_SHAPE[1]
        tiles[:, :, :, i] = padded_image[:, s_start:s_end, :]

    segmented_tile = sam_segmentation_tiled(tiles=tiles, checkpoint_path=CHECKPOINT_PATH, points_per_side=64, box_nms_thresh=0.3, device=DEVICE)

    for i in range(n_tiles):
        s_start = i * STRIDE
        s_end = s_start + TILE_SHAPE[1]
        mask[:, s_start:s_end] = np.logical_or(mask[:, s_start:s_end], segmented_tile[:, :, i])

    final_mask = mask[:image.shape[0], :image.shape[1]]
    if image_transposed:
        final_mask = np.transpose(final_mask, (1, 0))

    print("Saving tubule mask...")
    cv.imwrite(os.path.join(OUT_FOLDER, f"{image_name}.tif"), normalize_intensity_levels(final_mask))

if __name__ == '__main__':
    main()
