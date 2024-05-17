import numpy as np
import cv2 as cv
from typing import List, Dict
from skimage.morphology import skeletonize
from skimage import measure
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import clear_border
import cv2 as cv

# Constants
BACKGROUND_INTENSITY_THRESHOLD = 50
AREA_THRESHOLD_PERCENTAGE = 0.3
CIRCULARITY_THRESHOLD = 0.6
AREA_THRESHOLD = 500

def filter_tubule_masks(image: np.ndarray, masks: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Filters tubule masks based on area and average intensity constraints.

    Parameters:
    - image (np.ndarray): The input image array.
    - masks (List[Dict]): List of dictionaries containing 'area' and 'segmentation'.

    Returns:
    - List[Dict]: Filtered list of mask dictionaries.
    """
    h, w, _ = image.shape
    th_area_top = h * w * AREA_THRESHOLD_PERCENTAGE  # Upper area threshold
    th_area_bot = 0  # Lower area threshold

    filtered_mask = []
    for mask in masks:
        if th_area_bot < mask['area'] < th_area_top and image[mask['segmentation']].mean() > BACKGROUND_INTENSITY_THRESHOLD:
            filtered_mask.append(mask)
    
    return filtered_mask

def combine_masks(masks: List[Dict[str, any]], image: np.ndarray) -> np.ndarray:
    """
    Combines multiple masks into a single mask, applying erosion to each.

    Parameters:
    - masks (List[Dict]): List of mask dictionaries.
    - image (np.ndarray): The input image array for shape reference.

    Returns:
    - np.ndarray: Combined binary mask.
    """
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        segmentation = (255 * mask['segmentation'].astype(np.uint8))
        eroded = erode_tubule(segmentation)
        combined = np.logical_or(combined, eroded)

    return combined.astype(np.uint8)

def erode_tubule(segmentation: np.ndarray) -> np.ndarray:
    """
    Applies erosion to a binary mask segmentation.

    Parameters:
    - segmentation (np.ndarray): Binary mask segmentation.

    Returns:
    - np.ndarray: Eroded binary mask.
    """
    se = create_circular_se(5)
    return cv.erode(segmentation, se)

def create_circular_se(radius: int) -> np.ndarray:
    """
    Creates a circular structuring element.

    Parameters:
    - radius (int): The radius of the circular structuring element.

    Returns:
    - np.ndarray: The structuring element.
    """
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def combine_input_channels(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Preprocess two images by element-wise multiplication, normalization, and histogram equalization.

    Parameters:
    - image1 (np.ndarray): First input image.
    - image2 (np.ndarray): Second input image.

    Returns:
    - np.ndarray: Preprocessed image.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Multiply images element-wise in float32 type for precision
    result = np.multiply(image1, image2, dtype='float32')

    # Normalize result to scale between 0-255
    scaled = normalize_intensity_levels(result)

    # Apply histogram equalization to improve image contrast
    equalized = cv.equalizeHist(scaled)

    return equalized

def normalize_intensity_levels(image: np.ndarray) -> np.ndarray:
    """
    Normalize the intensity levels of an image to fit within the 0-255 range.

    Parameters:
    - image (np.ndarray): Image to normalize.

    Returns:
    - np.ndarray: Normalized image with intensity scaled to 8-bit.
    """
    min_val, max_val = image.min(), image.max()
    if max_val - min_val == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)





def filter_nuclei(mask, model=None):

    if model==None:
        k = create_circular_se(2)
        opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=k, iterations=1)

        # Find contours in the binary mask
        contours, _ = cv.findContours(opened.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create an empty mask to store the filtered objects
        filtered_mask = np.zeros_like(opened)

        for contour in contours:
        # Calculate circularity, area, and centroid for each contour
          perimeter = cv.arcLength(contour, True)
          area = cv.contourArea(contour)

          if perimeter == 0:
            circularity = 0  # Avoid division by zero
          else:
            circularity = 4 * np.pi * area / (perimeter ** 2)


        # Filter objects based on circularity and area
          if (circularity > CIRCULARITY_THRESHOLD):
            # Draw the contour on the filtered mask
            cv.drawContours(filtered_mask, [contour], 0, 255, thickness=cv.FILLED)
        
        
        # nuclei_count = len(c)
    return filtered_mask


def filter_bb(mask):
  k = create_circular_se(3)
  opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=k, iterations=1)

  # Find contours in the binary mask
  contours, _ = cv.findContours(opened.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  # Create an empty mask to store the filtered objects
  filtered_mask = np.zeros_like(opened)
  ar=[]
  for contour in contours:
    # Calculate circularity, area, and centroid for each contour
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)

    if perimeter == 0:
      circularity = 0  # Avoid division by zero
    else:
      circularity = 4 * np.pi * area / (perimeter ** 2)

    ar.append(area)
    # Filter objects based on circularity and area
    if (area > AREA_THRESHOLD):
      # Draw the contour on the filtered mask
      cv.drawContours(filtered_mask, [contour], 0, 255, thickness=cv.FILLED)

  return filtered_mask, ar



def filter_lumen(mask):
  k = create_circular_se(2)
  opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=k, iterations=1)

  return opened



def find_objects_near_skeleton(main_mask, object_mask, proximity_threshold=10, area_threshold=50):
    """
    Identifies objects in `object_mask` that are within `proximity_threshold` of the skeleton of `main_mask`.
    
    Args:
    main_mask (array-like): Binary mask of the main object where the skeleton will be computed.
    object_mask (array-like): Binary mask of other objects to check proximity against the skeleton.
    proximity_threshold (int): Maximum distance in pixels to consider objects as being close to the skeleton.
    
    Returns:
    numpy.ndarray: A binary mask of the same size as input masks, with 1s indicating objects close to the skeleton.
    """
    # Ensure inputs are boolean arrays
    # main_mask = img_as_bool(main_mask)
    # object_mask = img_as_bool(object_mask)
    
    # Compute the skeleton of the main object
    skeleton = skeletonize(main_mask)
    
    # Compute the distance transform
    distance_to_skeleton = distance_transform_edt(~skeleton)
    
    # Detect other objects using connected components
    labeled_objects, _ = measure.label(object_mask, return_num=True, connectivity=2)
    properties = measure.regionprops(labeled_objects)
    
    # Initialize the output mask
    close_objects_mask = np.zeros_like(object_mask, dtype=bool)
    
    # Mark objects that are close to the skeleton
    for prop in properties:
        print(prop.area)
        if prop.area > area_threshold:
            min_dist = np.min(distance_to_skeleton[prop.coords[:, 0], prop.coords[:, 1]])
            if min_dist <= proximity_threshold:
                for coord in prop.coords:
                    close_objects_mask[coord[0], coord[1]] = True
    
    return close_objects_mask



def update_classification_image(tubule_class_image: np.ndarray, contour: np.ndarray, data: dict):
    """
    Update the classification image based on the tubule type.

    Parameters:
        tubule_class_image (np.ndarray): The classification image.
        contour (np.ndarray): The contour array.
        data (dict): Processed data for the contour.
    """
    color_map = {'LTL': (255, 0, 0), 'Uro': (0, 255, 0), 'AQP2': (0, 0, 255), 'None': (255, 255, 255)}
    tubule_type = 'LTL' if data['LTL'] else 'Uro' if data['Uro'] else 'AQP2' if data['AQP2'] else 'None'
    color = color_map[tubule_type]
    cv.drawContours(tubule_class_image, [contour], -1, color, thickness=cv.FILLED)





def classify_tubule(mean_ch4_intensity: float, mean_ch5_intensity: float, mean_ch6_intensity: float, thresholds: dict) -> str:
    """
    Classify the type of a tubule based on channel intensities.

    Parameters:
        mean_ch4_intensity (float): Mean intensity of channel 4.
        mean_ch5_intensity (float): Mean intensity of channel 5.
        mean_ch6_intensity (float): Mean intensity of channel 6.
        thresholds (dict): Dictionary containing threshold values for classification.

    Returns:
        str: Type of the tubule ('LTL', 'Uro', 'AQP2', or 'None').
    """
    if mean_ch4_intensity > thresholds['CH4']:
        return 'LTL'
    elif mean_ch5_intensity > thresholds['CH5']:
        return 'Uro'
    elif mean_ch6_intensity > thresholds['CH6']:
        return 'AQP2'
    return 'None'



def process_mask_segment(mask: np.ndarray, tubule_mask_ch1: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Process a segment of a mask image based on the tubule mask.

    Parameters:
        mask (np.ndarray): The mask image.
        tubule_mask_ch1 (np.ndarray): The tubule mask.
        x (int): The x-coordinate of the region.
        y (int): The y-coordinate of the region.
        w (int): The width of the region.
        h (int): The height of the region.

    Returns:
        np.ndarray: The processed mask segment.
    """
    mask_temp = mask[y:y+h, x:x+w]
    return np.logical_and(tubule_mask_ch1, mask_temp)



def remove_border_tubules(mask, px=10):
    """
    Remove tubules near the image border.

    Parameters:
    - mask (numpy.ndarray): Binary mask of tubules.
    - px (int): Padding around the border to keep.
    - same_mask (bool): Whether to modify the input mask or return a new one.

    Returns:
    - numpy.ndarray: Mask with border tubules removed.
    """
    # kernel = create_circular_se(radius=1)
    # eroded = cv.erode(mask, kernel=kernel)

    cleared = clear_border(mask[px:mask.shape[0]-px, px:mask.shape[1]-px])

    padded = np.pad(cleared, pad_width=px)

    return padded