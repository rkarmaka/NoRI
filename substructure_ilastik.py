from typing import List, Tuple
from nori_modules.image_processing import filter_nuclei, filter_bb, filter_lumen
from nori_modules.utils import read_file_names
import cv2 as cv

ROOT_FOLDER = '../../analysis/all_images(processed)/analyzed/ilastik'
OUT_FOLDER = f'{ROOT_FOLDER}/..'
IMAGE_PATHS: List[str] = read_file_names(root_folder=ROOT_FOLDER, file_type=1)

for image_path in IMAGE_PATHS:
    image_name = image_path.split('/')[-1].split('.')[0]
    ilastik_segmentation = cv.imread(f'{image_path}', cv.IMREAD_GRAYSCALE)
    
    nuclei = 255*(ilastik_segmentation==1).astype('uint8')
    filtered_nuclei = filter_nuclei(nuclei)
    
    brushborder = 255*(ilastik_segmentation==2).astype('uint8')
    filtered_bb, _ = filter_bb(brushborder)

    lumen = 255*(ilastik_segmentation==3).astype('uint8')
    filtered_lumen = filter_lumen(lumen)

    cv.imwrite(f'{OUT_FOLDER}/nuclei/{image_name}.png', filtered_nuclei)
    cv.imwrite(f'{OUT_FOLDER}/brushborder/{image_name}.png', filtered_bb)
    cv.imwrite(f'{OUT_FOLDER}/lumen/{image_name}.png', filtered_lumen)

