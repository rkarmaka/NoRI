import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from nori_modules.postprocessing import combine_masks

def sam_segmentation_tiled(tiles, device="cuda", points_per_side=32, box_nms_thresh=0.3):
    """
    Perform segmentation on a stack of images using the SAM (Segment Anything) model.

    Parameters:
    - image_stack (numpy.ndarray): Input image stack with shape (height, width, num_tiles).
    - device (str): Device on which to run the segmentation (default is "cuda" for GPU).

    Returns:
    - numpy.ndarray: Segmented image stack with the same shape as the input.
    """

    # Load SAM model checkpoint
    sam_checkpoint = "/Users/ranit/IAC/Project/Will Trim/code/v2/sam_model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Initialize SAM model and move it to the specified device
    print('Loading SAM model...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Initialize SAM mask generator
    mask_generator = SamAutomaticMaskGenerator(sam, box_nms_thresh=box_nms_thresh, points_per_side=points_per_side)

    print(f'Segmenting image on {device.upper()}')

    n_tiles = tiles.shape[-1]
    
    segmentation = np.zeros((tiles.shape[0], tiles.shape[1], n_tiles), dtype='uint8')

    for i in range(n_tiles):
        # Generate masks using SAM mask generator and combine them
        print(f"Tile {i+1}")
        image = tiles[:,:,:,i]
        masks = mask_generator.generate(image)
        
        # print(len(masks))
        # areas = [mask['area'] for mask in masks]
        masks = filter_tubules(image, masks)

        segmentation[:,:,i] = combine_masks(masks, image)

    return segmentation