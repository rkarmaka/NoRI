import os

import time

import numpy as np
import pandas as pd
import my_module
from my_module.preprocessing import preprocess_images
from my_module.segmentation import sam_segmentation, nuclei_and_brushed_border_seg, lumen_seg
from my_module.postprocessing import stitch_masks, watershed_segment, remove_border_tubules, get_labeled_mask
from my_module.utils import create_directory, read_tiff_and_extract_channels, extract_tubule, create_rgb_image, image_scaling, tile_image, label_tubules, extract_cyto_only_mask

import torch

import cv2 as cv


################################ PATH #######################################################################
# Define image path and name
# Record start time
#st_time = time.time()

image_path = "/Users/ranit/IAC/Project/Will Trim/_DATA/Batch_2/Batch 1 (healthy kidneys)"
image_name = "S3_fullpanel_translated.tif"

# Create a directory to save processed images
create_directory(f"{image_path}/{image_name.split('.')[0]}_2")

################################ READ IMAGE #################################################################
# Define tile parameters
tile_shape = (1970, 2000)
stride = 1500


# Read image and extract lipid and protein channel
print('Loading image...')
ch1, ch2 = read_tiff_and_extract_channels(f"{image_path}/{image_name}")

# Pre-process for contrast enhancement
print('Preprocessing...')
preprocessed = preprocess_images(ch1, ch2)

# Record end time
#end_time = time.time()

# Save processed image
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_processed.tif", image_scaling(preprocessed))

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
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_tubule.tif", image_scaling(reconstructed_mask))

# Watershed segmentation and processing
labels = watershed_segment(reconstructed_mask)
labels_bw = (labels > 1).astype(np.uint8)

# Remove border tubules
mask_no_border_component = remove_border_tubules(labels_bw, same_mask=False)
# Record end time
#end_time = time.time()

print("Saving mask without border objects...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_no_border_tubule.tif", image_scaling(mask_no_border_component))

contours, _ = cv.findContours(mask_no_border_component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

# Display total time taken
#print(f'SAM time: {(end_time - st_time) / 60} seconds')
################################ NUCLEI, LUMEN, BRUSHED BORDER #######################################################
# Extract tubules
#mask_no_border_component = cv.imread(f"{image_path}/{image_name.split('.')[0]}/{image_name.split('.')[0]}_no_border_tubule.png", cv.IMREAD_GRAYSCALE)
# Record start time
#st_time = time.time()
# Add closing
# Extract tubules
contours, _ = cv.findContours(mask_no_border_component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

data_list = []
## check equalizeHist saturation factor
## check the difference between 12 and 8 bit image


# Segmentation
nuclei = np.zeros((ch1.shape))
lumen = np.zeros((ch2.shape))
brushed_border = np.zeros((ch2.shape))
labeled_mask = mask_no_border_component.copy()

for idx, contour in enumerate(contours, start=1):
    tubule_ch1, tubule_mask_ch1, x, y, w, h = extract_tubule(original_image=ch1, contour=contour, binary_mask=mask_no_border_component)
    tubule_ch2, tubule_mask_ch2, x, y, w, h = extract_tubule(original_image=ch2, contour=contour, binary_mask=mask_no_border_component)
    nuclei_temp, brushed_border_temp, nuclei_count, bb_count = nuclei_and_brushed_border_seg(tubule=tubule_ch2, mask=tubule_mask_ch2)
    nuclei[y:y+h, x:x+w] = np.logical_or(nuclei[y:y+h, x:x+w], nuclei_temp)
    brushed_border[y:y+h, x:x+w] = np.logical_or(brushed_border[y:y+h, x:x+w], brushed_border_temp)
    lumen_temp, lumen_count = lumen_seg(tubule=tubule_ch1, mask=tubule_mask_ch1)
    lumen[y:y+h, x:x+w] = np.logical_or(lumen[y:y+h, x:x+w], lumen_temp)

#    display_image(tubule_ch1)
#    display_image(tubule_ch2)
#    display_image(nuclei_temp)
#    display_image(brushed_border_temp)
#    display_image(lumen_temp)

    cyto_only = extract_cyto_only_mask(tubule_mask_ch1, nuclei_mask=nuclei_temp,
                                       bb_mask=brushed_border_temp, lumen_mask=lumen_temp)
    total_protein_intensity = tubule_ch1[cyto_only].sum()
    total_lipid_intensity = tubule_ch2[cyto_only].sum()
    mean_protein_intensity = np.round(tubule_ch1[cyto_only].mean(), 4)
    mean_lipid_intensity = np.round(tubule_ch2[cyto_only].mean(), 4)

    entry = {
            'id': idx,
            'total_protein_intensity': total_protein_intensity,
            'total_lipid_intensity': total_lipid_intensity,
            'mean_protein_intensity': mean_protein_intensity,
            'mean_lipid_intensity': mean_lipid_intensity,
            'nuclie_count': nuclei_count,
            'bb_exists': bb_count,
            'lumen_exists': lumen_count
        }

    labeled_mask = get_labeled_mask(image=labeled_mask, contour=contour, idx=idx)

    data_list.append(entry)
# Record end time
#end_time = time.time()

pd.DataFrame(data_list).to_csv(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}.csv")


# Record end time
#end_time = time.time()

# Display total time taken
#print(f'Other seg time: {(end_time - st_time) / 60} seconds')

# Save nuclei mask
print("Saving nuclei mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_nuclei.tif", image_scaling(nuclei))

# Save brush border mask
print("Saving brushed border mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_bb.tif", image_scaling(brushed_border))

# Save lumen mask
print("Saving lumen mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_lumen.tif", image_scaling(lumen))

# Save labeled mask
print("Saving labeled mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}_2/{image_name.split('.')[0]}_labeled.tif", image_scaling(labeled_mask))



################################ FINAL #######################################################################
# Final mask
mask = ((image_scaling(mask_no_border_component) - np.multiply(127, brushed_border)).astype(np.uint8) -
        np.multiply(255, nuclei).astype(np.uint8))

cyto_only_mask = extract_cyto_only_mask(main_mask=mask_no_border_component,
                                        nuclei_mask=nuclei, bb_mask=brushed_border,
                                        lumen_mask=lumen)

# Save final mask
print("Saving final mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}/{image_name.split('.')[0]}_full_mask.tif",
           image_scaling(cyto_only_mask.astype(np.uint8)))

# Save RGB mask
rgb_mask = create_rgb_image(brushed_border, nuclei, cyto_only_mask)
print("Saving RGB mask...")
cv.imwrite(f"{image_path}/{image_name.split('.')[0]}/{image_name.split('.')[0]}_RGB_mask.tif",
           image_scaling(rgb_mask))


################################ ANALYZE ###################################################################
# Extract each tubule
# Extract tubules
#contours, _ = cv.findContours(mask_no_border_component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#labeled_mask = mask_no_border_component.copy()
#cytoplasm_mask = extract_cyto_only_mask(mask_no_border_component, nuclei_mask=nuclei, lumen_mask=lumen, bb_mask=brushed_border)

#for i, contour in enumerate(contours, start=1):
#    labeled_mask = label_tubules(mask=labeled_mask, contour=contour, idx=i)


# Save final mask
#print("Saving labeled mask...")
#cv.imwrite(f"{image_path}/{image_name.split('.')[0]}/{image_name.split('.')[0]}_labeled_mask.tif",
#           image_scaling(labeled_mask))

# Create cyto-only mask
# Measure total protein in ch1
# Measure total lipid in ch2
# Count the number of nuclei inside
# Whether it has a brushed border
# Whether it has lumen
# Export a CSV
