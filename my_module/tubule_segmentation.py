import numpy as np
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def sam_segmentation(image_stack):
    sam_checkpoint = "sam_model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)


    segment_stack = np.zeros(image_stack.shape, dtype='uint8')

    for i in range(image_stack.shape[2]):
        print(i)
        image = np.stack((image_stack[:,:,i],image_stack[:,:,i],image_stack[:,:,i]), axis=-1).astype(np.uint8)
        masks = mask_generator.generate(image)
        segment_stack[:,:,i] = combine_masks(masks, image)


    return segment_stack
                                           


def combine_masks(masks,image):
    mask = np.zeros(masks[0]['segmentation'].shape)
    th_area_top = (1970*2000*0.05) # if the area of each tubule is greater than 5%
    th_area_bot = 1000
    th_int = 30 # an arbitrary threshold. Basically trying to remove background and areas with very low intensity
    for i in range(len(masks)):
        if ((masks[i]['area']>th_area_bot) & (masks[i]['area']<th_area_top) & (image[masks[i]['segmentation']].mean()>th_int)):
          mask = np.logical_or(mask,masks[i]['segmentation'])

    return mask.astype(np.uint8)