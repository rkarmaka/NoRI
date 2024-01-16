import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from my_module.postprocessing import combine_masks, gray_to_rgb, create_circular_kernel, image_scaling, filter_brushed_border, filter_nuclei, filter_lumen

import cv2 as cv

def sam_segmentation(image_stack, device="cuda"):
    """
    Perform segmentation on a stack of images using the SAM (Segment Anything) model.

    Parameters:
    - image_stack (numpy.ndarray): Input image stack with shape (height, width, num_tiles).
    - device (str): Device on which to run the segmentation (default is "cuda" for GPU).

    Returns:
    - numpy.ndarray: Segmented image stack with the same shape as the input.
    """

    # Load SAM model checkpoint
    sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Initialize SAM model and move it to the specified device
    print('Loading SAM model...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Initialize SAM mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Initialize an array to store segmented images
    segment_stack = np.zeros(image_stack.shape, dtype='uint8')

    # Iterate through image stack and segment each tile
    for i in range(image_stack.shape[2]):
        print(f'Segmenting tile {i+1} on GPU {device}')

        # Convert grayscale tile to RGB
        image = np.stack((image_stack[:,:,i], image_stack[:,:,i], image_stack[:,:,i]), axis=-1).astype(np.uint8)

        # Generate masks using SAM mask generator and combine them
        masks = mask_generator.generate(image)
        segment_stack[:,:,i] = combine_masks(masks, image)

    return segment_stack

def sam_segmentation_single_tile(image, device="cuda"):
    """
    Perform segmentation on a single tile using the SAM (Segment Anything) model.

    Parameters:
    - image (numpy.ndarray): Input grayscale image.
    - device (str): Device on which to run the segmentation (default is "cuda" for GPU).

    Returns:
    - numpy.ndarray: Segmented mask for the input image.
    """

    # Load SAM model checkpoint
    sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Initialize SAM model and move it to the specified device
    print('Loading SAM model...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Initialize SAM mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)

    print(f'Running SAM segmentation on device {device}...')

    # Convert grayscale image to RGB and generate masks using SAM mask generator
    image = gray_to_rgb(image)
    masks = mask_generator.generate(image)

    # Combine masks to get the final segmentation mask
    mask = combine_masks(masks, image)

    return mask

def nuclei_and_brushed_border_seg(tubule, mask, model=None):

    if model==None:
        tubule = cv.equalizeHist(image_scaling(tubule))
        blurred = cv.medianBlur(tubule, 3)


        # Nuclei Segmentation
        _, thresh = cv.threshold(blurred, blurred.max()*0.15, blurred.max(), cv.THRESH_BINARY_INV)
        opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel=create_circular_kernel(2), iterations=1)
        nuclei = np.logical_and(opened, mask)
        opened = cv.morphologyEx(nuclei.astype(np.uint8), cv.MORPH_OPEN, kernel=create_circular_kernel(1), iterations=2)
        filtered_nuclei = filter_nuclei(opened, mask, circularity_threshold=0.75, area_threshold_bot=50, area_threshold_top=500)

        c, _ = cv.findContours(filtered_nuclei.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        nuclei_count = len(c)


        # Brushed Border Segmentation
        _, thresh = cv.threshold(blurred, blurred.max()*0.88, blurred.max(), cv.THRESH_BINARY)
        opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel=np.ones((1,1)), iterations=1)
        nuclei = np.logical_and(opened, mask)
        opened = cv.morphologyEx(nuclei.astype(np.uint8), cv.MORPH_OPEN, kernel=np.ones((1,1)), iterations=2)
        filtered_brushed_border = filter_brushed_border(opened, mask)

        c, _ = cv.findContours(filtered_brushed_border.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(c)==0:
            bb_count = False
        else:
            bb_count = True



    return filtered_nuclei, filtered_brushed_border, nuclei_count, bb_count

def lumen_seg(tubule, mask, model=None):
    """
    Perform lumen segmentation on a tubule image.

    Parameters:
    - tubule (numpy.ndarray): Input tubule image.
    - mask (numpy.ndarray): Segmentation mask for the tubule image.
    - model: Not used in this function.

    Returns:
    - numpy.ndarray: Segmented lumen mask.
    """

    # Check if a specific model is provided, if not, apply default segmentation pipeline
    if model==None:
        blurred = cv.medianBlur(tubule, 3)
        _, thresh = cv.threshold(blurred, blurred.max()*0.32, blurred.max(), cv.THRESH_BINARY_INV)
        thresh = image_scaling(thresh)

        opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel=np.ones((1,1)), iterations=1)
        bb = np.logical_and(opened, mask)

        opened = cv.morphologyEx(bb.astype(np.uint8), cv.MORPH_OPEN, kernel=create_circular_kernel(1), iterations=1)

        if opened.max()!=0:
            filtered = filter_lumen(mask, opened)
            c, _ = cv.findContours(filtered.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(c)==0:
                lumen_count=False
            else:
                lumen_count=True

        else:
            filtered=opened
            lumen_count = False

    return filtered, lumen_count

    return filtered
