import os

import numpy as np
from nori_modules.preprocessing import preprocess_images
from nori_modules.segmentation import sam_segmentation_tiled
from nori_modules.utils import read_tiff_file_names, pad_image, transpose_input_image, intensity_scaling

import cv2 as cv


################################ PATH #######################################################################
# Define image path and name
root_folder = "/Users/ranit/IAC/Project/Will Trim/analysis/all_images(processed)/test"
out_folder = "/Users/ranit/IAC/Project/Will Trim/analysis/all_images(processed)/test/tubules/stride_250"

tiff_files = read_tiff_file_names(root_folder=root_folder)

################################ READ IMAGE #################################################################
# Define tile parameters
tile_shape = (1970, 2000)
stride = 250

for tiff_file in tiff_files:
    # Read image and extract lipid and protein channel
    image_name = tiff_file.split('/')[-1].split('.')[0]

    # # Create a directory to save processed images
    # create_directory(f"{root_folder}/analyzed/tubules/")
    
    print(f'Loading image {image_name}')
    image = cv.imread(tiff_file)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image[:,:,2] = preprocess_images(image[:,:,0], image[:,:,1])

    image, image_transposed = transpose_input_image(image)

    ################################ TUBULE SEGMENTATION ############################################################
    # Break into tiles and stack them
    print('Breaking image into tiles...')
    padded_image = pad_image(image, tile_shape=tile_shape, stride=stride)

    # SAM segmentation
    print('Tubule segmentation...')

    # Perform SAM segmentation on the entire stack
    n_tiles = int(np.ceil((padded_image.shape[1] - tile_shape[1])/stride)) + 1
    
    mask = np.zeros((padded_image.shape[0], padded_image.shape[1]), dtype='uint8')

    tiles = np.zeros((tile_shape[0], tile_shape[1], 3, n_tiles), dtype='uint8')
    for i in range(n_tiles):
        s_start = i*stride
        s_end = s_start+tile_shape[1]
        tiles[:,:,:,i] = padded_image[:, s_start:s_end, :]

    segmented_tile = sam_segmentation_tiled(tiles = tiles,
                                               points_per_side=64,
                                               box_nms_thresh=0.3,
                                               device='cpu')

    for i in range(n_tiles):
        s_start = i*stride
        s_end = s_start+tile_shape[1]
        mask[:,s_start:s_end] = np.logical_or(mask[:,s_start:s_end], segmented_tile[:,:,i])

    final_mask = mask[:image.shape[0], :image.shape[1]]

    if image_transposed:
        final_mask = np.transpose(final_mask, (1,0))

    # final_mask = watershed_segment(final_mask)

    # Save tubule mask
    print("Saving tubule mask...")
    cv.imwrite(f"{root_folder}/tubules/{image_name}.png", intensity_scaling(final_mask))