import os

import numpy as np
from my_module.preprocessing import preprocess_images
from my_module.segmentation import sam_segmentation, nuclei_and_brushed_border_seg, lumen_seg
from my_module.postprocessing import stitch_masks, watershed_segment, remove_border_tubules, get_labeled_mask
from my_module.utils import create_directory, read_tiff_and_extract_channels, extract_tubule, create_rgb_image, image_scaling, tile_image, label_tubules, extract_cyto_only_mask

import cv2 as cv


################################ PATH #######################################################################
# Define image path and name
# Record start time
#st_time = time.time()

image_path = "/Users/ranit/IAC/Project/Will Trim/_DATA/Batch_2/Batch 1 (healthy kidneys)"
image_names = ["S1_fullpanel_translated", "S1(2)_fullpanel_translated",
               "S2_fullpanel_translated", "S2(2)_fullpanel_translated",
               "S3_fullpanel_translated", "S3(2)_fullpanel_translated",
               "S4_fullpanel_translated", "S4(2)_fullpanel_translated",
               "S5_fullpanel_translated", "S5(2)_fullpanel_translated",
               "S6_fullpanel_translated", "S6(2)_fullpanel_translated",
               "S7_fullpanel_translated", "S7(2)_fullpanel_translated",
               "S8_fullpanel_translated", "S8(2)_fullpanel_translated",
               "S9_fullpanel_translated", "S9(2)_fullpanel_translated",
               "S10_fullpanel_translated", "S10(2)_fullpanel_translated"]

# Create a directory to save processed images
create_directory(f"{image_path}/Analysis/Tubules_only")

################################ READ IMAGE #################################################################
# Define tile parameters
tile_shape = (1970, 2000)
stride = 1500

for image_name in image_names:
    # Read image and extract lipid and protein channel
    print(f'Loading image {image_name}')
    ch1, ch2 = read_tiff_and_extract_channels(f"{image_path}/{image_name}.tif")

    # Pre-process for contrast enhancement
    print('Preprocessing...')
    preprocessed = preprocess_images(ch1, ch2)

    # Record end time
    #end_time = time.time()

    # Save processed image
    cv.imwrite(f"{image_path}/Analysis/Tubules_only/{image_name.split('.')[0]}_processed.tif", image_scaling(preprocessed))

    # Display total time taken
    #print(f'Reading time: {(end_time - st_time) / 60} seconds')
    ################################ TUBULE SEGMENTATION ############################################################
    # Break into tiles and stack them
    print('Breaking image into tiles...')
    # Record start time
    #st_time = time.time()

    image_stack = tile_image(preprocessed, tile_shape=tile_shape, stride=stride)

    # SAM segmentation
    print('Tubule segmentation...')

    # Perform SAM segmentation on the entire stack
    segment_stack = sam_segmentation(image_stack, device='cpu')  # For Apple

    # Post-process SAM segmentation
    print('Postprocessing...')
    reconstructed_mask = stitch_masks(segment_stack, ch1.shape, tile_shape, stride)


    # Save tubule mask
    print("Saving tubule mask...")
    cv.imwrite(f"{image_path}/Analysis/Tubules_only/{image_name.split('.')[0]}_tubule.tif", image_scaling(reconstructed_mask))

    # Watershed segmentation and processing
    labels = watershed_segment(reconstructed_mask)
    labels_bw = (labels > 1).astype(np.uint8)

    # Remove border tubules
    mask_no_border_component = remove_border_tubules(labels_bw, same_mask=False)
    # Record end time
    #end_time = time.time()

    print("Saving mask without border objects...")
    cv.imwrite(f"{image_path}/Analysis/Tubules_only/{image_name.split('.')[0]}_no_border_tubule.tif", image_scaling(mask_no_border_component))

    contours, _ = cv.findContours(mask_no_border_component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    print("##"*50)