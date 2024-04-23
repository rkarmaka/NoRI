import numpy as np
from typing import Any
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from nori_modules.image_processing import combine_masks, filter_tubule_masks

def sam_segmentation_tiled(tiles: np.ndarray, 
                           checkpoint_path: str, 
                           model_type: str = "vit_h", 
                           device: str = "cuda", 
                           points_per_side: int = 32, 
                           box_nms_thresh: float = 0.3) -> np.ndarray:
    """
    Perform segmentation on a stack of images using the SAM (Segment Anything) model.

    Parameters:
    - tiles (np.ndarray): Input image stack with shape (height, width, channels, num_tiles).
    - checkpoint_path (str): Path to the model checkpoint.
    - model_type (str): Type of the model (vit_h, vit_b, vit_l) with 'vit_h' as the default.
    - device (str): Device on which to run the segmentation (default is "cuda" for GPU).
    - points_per_side (int): Number of points per side for masks.
    - box_nms_thresh (float): Box non-maximum suppression threshold.

    Returns:
    - np.ndarray: Segmented image stack with the same shape as the input (height, width, num_tiles).
    """

    # Validate model type
    valid_model_types = ['vit_h', 'vit_b', 'vit_l']
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model type provided. Choose from {valid_model_types}")

    print('Loading SAM model...')
    sam: Any = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(
        sam, box_nms_thresh=box_nms_thresh, points_per_side=points_per_side)

    print(f'Segmenting image on {device.upper()}')

    n_tiles: int = tiles.shape[-1]
    segmentation: np.ndarray = np.zeros((tiles.shape[0], tiles.shape[1], n_tiles), dtype='uint8')

    for i in tqdm(range(n_tiles)):
        image: np.ndarray = tiles[:,:,:,i]
        masks: list = mask_generator.generate(image)
        masks = filter_tubule_masks(image, masks)
        segmentation[:,:,i] = combine_masks(masks, image)

    return segmentation
