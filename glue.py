import cv2 as cv
import numpy as np
import pandas as pd
from typing import List

from nori_modules.data_loader import read_tiff_and_extract_channels
from nori_modules.utils import read_file_names, extract_tubule, extract_cyto_only_mask, get_centroid
from nori_modules.image_processing import remove_border_tubules, update_classification_image, classify_tubule, process_mask_segment
from nori_modules.measure import measure_intensity, measure_content, measure_nuclei_intensity




TEST = True

if TEST:
    ROOT_FOLDER = "../../analysis/all_images(processed)/test/raw"


    FOLDERS = {
    'OUT' : '../../analysis/all_images(processed)/test/out',
    'TUBULES' : "../../analysis/all_images(processed)/analyzed/tubules/corrected",
    'NUCLEI' : "../../analysis/all_images(processed)/analyzed/nuclei",
    'BRUSHBORDER' : "../../analysis/all_images(processed)/analyzed/brushborder",
    'LUMEN' : "../../analysis/all_images(processed)/analyzed/lumen"
    }
else:
    ROOT_FOLDER = "../../_DATA"
    
    FOLDERS = {
    'OUT' : '../../analysis/all_images(processed)/analyzed/out',
    'TUBULES' : "../../analysis/all_images(processed)/analyzed/tubules/corrected",
    'NUCLEI' : "../../analysis/all_images(processed)/analyzed/nuclei",
    'BRUSHBORDER' : "../../analysis/all_images(processed)/analyzed/brushborder",
    'LUMEN' : "../../analysis/all_images(processed)/analyzed/lumen"
    }



CONST = {
    'BSA' : 1.3643,
    'DOPC' : 1.0101
}

THRESHOLDS = {
    'CH4' : 883, #90%
    'CH5' : 754, #90%
    'CH6' : 413 #90%
}


def process_image(file_path: str, folders: dict, thresholds: dict, constants: dict) -> dict:
    """
    Process a single image to extract and measure various features.
    
    Parameters:
        file_path (str): Path to the image file.
        folders (dict): Dictionary containing various folder paths.
        thresholds (dict): Dictionary containing threshold values for channel classification.
        constants (dict): Dictionary containing constant values for measurement calculations.

    Returns:
        dict: Processed data for the image.
    """
    image_name = file_path.split('/')[-1].split('.')[0]
    print(image_name)

    try:
        protein_channel, lipid_channel, ch3, ch4, ch5, ch6 = read_tiff_and_extract_channels(file_path)
    except Exception as e:
        print(f'Cannot open {image_name}: {e}')
        return {}

    try:
        print(f'Reading tubule masks...')
        tubule = cv.imread(f'{folders["TUBULES"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        tubule = remove_border_tubules(tubule)
        
        print('Reading substructure masks...')
        nuclei = cv.imread(f'{folders["NUCLEI"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        brushborder = cv.imread(f'{folders["BRUSHBORDER"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        lumen = cv.imread(f'{folders["LUMEN"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)

        contours, _ = cv.findContours(tubule, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        data_list = []
        
        # cyto_only = np.zeros_like(tubule)
        tubule_class_image = np.zeros((tubule.shape[0], tubule.shape[1], 3), dtype='uint8')

        for idx, contour in enumerate(contours, start=1):
            data = process_contour(idx=idx,
                                   contour=contour,
                                   protein_channel=protein_channel,
                                   lipid_channel=lipid_channel,
                                   tubule=tubule,
                                   nuclei=nuclei,
                                   brushborder=brushborder,
                                   lumen=lumen,
                                   ch3=ch3,
                                   ch4=ch4,
                                   ch5=ch5,
                                   ch6=ch6,
                                   thresholds=thresholds,
                                   constants=constants)
            if data:
                data_list.append(data)
                update_classification_image(tubule_class_image, contour, data)

        save_results(data_list, tubule_class_image, folders['OUT'], image_name)
    except Exception as e:
        print(f'Cannot find mask for {image_name}: {e}')

def process_contour(idx: int, contour: np.ndarray, protein_channel: np.ndarray, lipid_channel: np.ndarray, tubule: np.ndarray, nuclei: np.ndarray,
                    brushborder: np.ndarray, lumen: np.ndarray, ch3: np.ndarray, ch4: np.ndarray, ch5: np.ndarray, ch6: np.ndarray, 
                    thresholds: dict, constants: dict) -> dict:
    """
    Process a single tubule contour to measure various features.
    
    Parameters:
        idx (int): Index of the contour.
        contour (np.ndarray): Contour array.
        protein_channel (np.ndarray): Protein channel image.
        lipid_channel (np.ndarray): Lipid channel image.
        tubule (np.ndarray): Tubule mask image.
        nuclei (np.ndarray): Nuclei mask image.
        brushborder (np.ndarray): Brushborder mask image.
        lumen (np.ndarray): Lumen mask image.
        thresholds (dict): Dictionary containing threshold values for channel classification.
        constants (dict): Dictionary containing constant values for measurement calculations.

    Returns:
        dict: Processed data for the contour.
    """
    tubule_ch1, tubule_mask_ch1, x, y, w, h = extract_tubule(original_image=protein_channel, contour=contour, binary_mask=tubule)
    tubule_ch2, _, _, _, _, _ = extract_tubule(original_image=lipid_channel, contour=contour, binary_mask=tubule)
    
    nuclei_temp = process_mask_segment(nuclei, tubule_mask_ch1, x, y, w, h)
    brushborder_temp = process_mask_segment(brushborder, tubule_mask_ch1, x, y, w, h)
    lumen_temp = process_mask_segment(lumen, tubule_mask_ch1, x, y, w, h)

    mean_nuclei_protein_intensity, std_nuclei_protein_intensity, nuclei_count = measure_nuclei_intensity(tubule_ch1, nuclei_temp)
    mean_nuclei_lipid_intensity, std_nuclei_lipid_intensity, _ = measure_nuclei_intensity(tubule_ch2, nuclei_temp)

    _, mean_bb_protein_intensity, _ = measure_intensity(tubule_ch1, brushborder_temp)
    _, mean_bb_lipid_intensity, _ = measure_intensity(tubule_ch2, brushborder_temp)
    _, mean_lumen_protein_intensity, _ = measure_intensity(tubule_ch1, lumen_temp)
    _, mean_lumen_lipid_intensity, _ = measure_intensity(tubule_ch2, lumen_temp)

    cyto_only_temp = extract_cyto_only_mask(tubule_mask_ch1, nuclei_mask=nuclei_temp, bb_mask=brushborder_temp, lumen_mask=lumen_temp)
    total_protein_intensity, mean_protein_intensity, std_protein_intensity = measure_intensity(tubule_ch1, cyto_only_temp)
    total_lipid_intensity, mean_lipid_intensity, std_lipid_intensity = measure_intensity(tubule_ch2, cyto_only_temp)

    total_protein, mean_protein = measure_content(tubule_ch1, cyto_only_temp, constants['BSA'])
    total_lipid, mean_lipid = measure_content(tubule_ch2, cyto_only_temp, constants['DOPC'])

    _, mean_ch3_intensity, _ = measure_intensity(ch3[y:y+h, x:x+w], cyto_only_temp)
    _, mean_ch4_intensity, _ = measure_intensity(ch4[y:y+h, x:x+w], cyto_only_temp)
    _, mean_ch5_intensity, _ = measure_intensity(ch5[y:y+h, x:x+w], cyto_only_temp)
    _, mean_ch6_intensity, _ = measure_intensity(ch6[y:y+h, x:x+w], cyto_only_temp)

    tubule_type = classify_tubule(mean_ch4_intensity, mean_ch5_intensity, mean_ch6_intensity, thresholds)

    if cyto_only_temp.sum() == 0:
        return {}

    total_cyto_area = cyto_only_temp.sum()
    cx, cy = get_centroid(contour)

    return {
        'id': idx,
        'x': cx,
        'y': cy,
        'total_cyto_area': total_cyto_area,
        'total_protein': total_protein,
        'mean_protein': mean_protein,
        'total_lipid': total_lipid,
        'mean_lipid': mean_lipid,
        'total_protein_intensity': total_protein_intensity,
        'total_lipid_intensity': total_lipid_intensity,
        'mean_protein_intensity': mean_protein_intensity,
        'mean_lipid_intensity': mean_lipid_intensity,
        'mean_ch3_intensity': mean_ch3_intensity,
        'mean_ch4_intensity': mean_ch4_intensity,
        'mean_ch5_intensity': mean_ch5_intensity,
        'mean_ch6_intensity': mean_ch6_intensity,
        'LTL': tubule_type == 'LTL',
        'Uro': tubule_type == 'Uro',
        'AQP2': tubule_type == 'AQP2',
        'std_protein_intensity': std_protein_intensity,
        'std_lipid_intensity': std_lipid_intensity,
        'nuclei_count': nuclei_count,
        'mean_nuclei_protein_intensity': mean_nuclei_protein_intensity,
        'mean_nuclei_lipid_intensity': mean_nuclei_lipid_intensity,
        'std_nuclei_protein_intensity': std_nuclei_protein_intensity,
        'std_nuclei_lipid_intensity': std_nuclei_lipid_intensity,
        'bb_exists': brushborder_temp.sum() != 0,
        'bb_protein_intensity': mean_bb_protein_intensity,
        'bb_lipid_intensity': mean_bb_lipid_intensity,
        'lumen_exists': lumen_temp.sum() != 0,
        'lumen_protein_intensity': mean_lumen_protein_intensity,
        'lumen_lipid_intensity': mean_lumen_lipid_intensity,
    }


def save_results(data_list: List[dict], tubule_class_image: np.ndarray, out_folder: str, image_name: str):
    """
    Save the results of the image processing.

    Parameters:
        data_list (List[dict]): List of processed data for the image.
        tubule_class_image (np.ndarray): The classification image.
        out_folder (str): Output folder path.
        image_name (str): Name of the image.
    """
    if not data_list:
        return
    d = pd.DataFrame(data_list)
    cv.imwrite(f'{out_folder}/tubule_class_mask/{image_name}_.png', tubule_class_image)
    d.to_csv(f'{out_folder}/csv/{image_name}_.csv')










def main():
    """
    Main function to process all images in the specified folder.
    """
    file_paths = read_file_names(root_folder=ROOT_FOLDER, file_type=0)
    for file_path in file_paths:
        process_image(file_path, FOLDERS, THRESHOLDS, CONST)

        break

if __name__ == '__main__':
    main()
