import numpy as np
from my_module.utils import get_centroid, euclidean_distance, pad_image, image_scaling, gray_to_rgb
import cv2 as cv

from skimage.segmentation import clear_border

def combine_masks(masks, image):
    """
    Combine masks based on area, intensity, and size constraints.

    Parameters:
    - masks (list): List of dictionaries containing 'area' and 'segmentation'.
    - image (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Combined binary mask.
    """
    mask = np.zeros(masks[0]['segmentation'].shape)
    th_area_top = (1970 * 2000 * 0.05)  # if the area of each tubule is greater than 5%
    th_area_bot = (1970 * 2000 * 0.00001)
    th_int = 75  # an arbitrary threshold. Basically trying to remove background and areas with very low intensity

    for i in range(len(masks)):
        if ((masks[i]['area'] > th_area_bot) & (masks[i]['area'] < th_area_top) & (image[masks[i]['segmentation']].mean() > th_int)):
            mask = np.logical_or(mask, masks[i]['segmentation'])

    return mask.astype(np.uint8)

def stitch_masks(tiles, image_shape, tile_shape, stride):
    """
    Stitch together a set of binary masks to reconstruct the original image.

    Parameters:
    - tiles (numpy.ndarray): Binary masks of individual tiles.
    - image_shape (tuple): Original image shape.
    - tile_shape (tuple): Shape of individual tiles.
    - stride (int): Stride used during tiling.

    Returns:
    - numpy.ndarray: Reconstructed binary mask.
    """
    image_reconstructed = np.zeros(image_shape)
    image_reconstructed = pad_image(image_reconstructed, tile_shape, stride)

    # Calculate the number of rows and columns for the tiles
    rows = (image_reconstructed.shape[0] - tile_shape[0]) // stride + 1
    cols = (image_reconstructed.shape[1] - tile_shape[1]) // stride + 1

    # Reconstruct the image from the tiles
    for i in range(rows):
        for j in range(cols):
            start_x = j * stride
            start_y = i * stride
            end_x = start_x + tile_shape[1]
            end_y = start_y + tile_shape[0]

            ch = (i * cols) + j
            image_reconstructed[start_y:end_y, start_x:end_x] = np.logical_or(image_reconstructed[start_y:end_y, start_x:end_x], tiles[:, :, ch])

    # Crop the image to its original dimensions
    image_cropped = image_reconstructed[:image_shape[0], :image_shape[1]]

    return image_cropped

def create_circular_kernel(radius):
    """
    Create a circular kernel for morphological operations.

    Parameters:
    - radius (int): Radius of the circular kernel.

    Returns:
    - numpy.ndarray: Circular kernel.
    """
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def image_opening(image, k):
    """
    Perform the opening operation on the input image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - k (int): Radius of the circular kernel.

    Returns:
    - numpy.ndarray: Result of the opening operation.
    """
    kernel = create_circular_kernel(k)
    opened_image = cv.morphologyEx(image, cv.MORPH_ERODE, kernel, iterations=4)
    return opened_image

def watershed_segment(image):
    """
    Perform image segmentation using the watershed algorithm.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Segmentation markers.
    """
    sure_bg = cv.dilate(image_scaling(image), create_circular_kernel(3))
    sure_fg = image_opening(image_scaling(image), 2)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv.watershed(gray_to_rgb(image), markers)

    return markers

def remove_border_tubules(mask, px=10, same_mask=True):
    """
    Remove tubules near the image border.

    Parameters:
    - mask (numpy.ndarray): Binary mask of tubules.
    - px (int): Padding around the border to keep.
    - same_mask (bool): Whether to modify the input mask or return a new one.

    Returns:
    - numpy.ndarray: Mask with border tubules removed.
    """
    kernel = create_circular_kernel(radius=1)
    eroded = cv.erode(mask, kernel=kernel)

    cleared = clear_border(eroded[px:mask.shape[0]-px, px:mask.shape[1]-px])

    padded = np.pad(cleared, pad_width=px)

    if same_mask:
        dilated = cv.dilate(padded, kernel=kernel, iterations=3)
        return np.multiply(mask, dilated)
    else:
        return padded

def filter_lumen(full_mask, mask, area_threshold=50):
    """
    Filter lumen based on contours and area thresholding.

    Parameters:
    - full_mask (numpy.ndarray): Full binary mask.
    - mask (numpy.ndarray): Binary mask to filter lumen from.
    - area_threshold (int): Minimum area of lumen regions to keep.

    Returns:
    - numpy.ndarray: Filtered lumen mask.
    """
    contours, _ = cv.findContours(full_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calculate the centroid of the current object
        M = cv.moments(contour)
        if M["m00"] != 0:
            full_centroid = np.array([
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            ])
    # Find contours in the object mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize variables to keep track of the closest object and its distance
    closest_object = None
    min_distance = float('inf')

    for contour in contours:
        # Calculate the centroid of the current object
        area = cv.contourArea(contour)
        M = cv.moments(contour)
        if (area > area_threshold):
            object_centroid = get_centroid(contour=contour)

            # Calculate the Euclidean distance between the centroids
            distance = euclidean_distance(centroid1=full_centroid, centroid2=object_centroid)

            # Update closest object if the current object is closer
            if distance < min_distance:
                min_distance = distance
                closest_object = contour

    # Create a new mask with only the closest object
    result_mask = np.zeros_like(mask)

    if closest_object is not None:
        cv.drawContours(result_mask, [closest_object], -1, 255, thickness=cv.FILLED)

    return result_mask

def filter_nuclei(binary_mask, full_mask, circularity_threshold=0.8, area_threshold_bot=50, area_threshold_top=100, centroid_threshold=10):
    """
    Filter nuclei based on circularity, area, and centroid distance.

    Parameters:
    - binary_mask (numpy.ndarray): Binary mask of nuclei.
    - full_mask (numpy.ndarray): Full binary mask.
    - circularity_threshold (float): Minimum circularity of nuclei.
    - area_threshold_bot (int): Minimum area of nuclei.
    - area_threshold_top (int): Maximum area of nuclei.
    - centroid_threshold (int): Minimum centroid distance.

    Returns:
    - numpy.ndarray: Filtered nuclei mask.
    """
    contours, _ = cv.findContours(full_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    full_centroid = get_centroid(contours[0])

    # Find contours in the binary mask
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the filtered objects
    filtered_mask = np.zeros_like(binary_mask)

    for contour in contours:
        # Calculate circularity, area, and centroid for each contour
        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)

        if perimeter == 0:
            circularity = 0  # Avoid division by zero
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        centroid = get_centroid(contour)
        distance = euclidean_distance(full_centroid, centroid)

        # Filter objects based on circularity and area
        if (circularity > circularity_threshold) and (area > area_threshold_bot) and (area < area_threshold_top) and (distance > centroid_threshold):
            # Draw the contour on the filtered mask
            cv.drawContours(filtered_mask, [contour], 0, 255, thickness=cv.FILLED)

    return filtered_mask

def filter_brush_border(binary_mask, full_mask, centroid_threshold=5):
    """
    Filter brushed border based on centroid distance.

    Parameters:
    - binary_mask (numpy.ndarray): Binary mask of brushed border.
    - full_mask (numpy.ndarray): Full binary mask.
    - centroid_threshold (int): Minimum centroid distance.

    Returns:
    - numpy.ndarray: Filtered brushed border mask.
    """
    filtered_mask = np.zeros_like(binary_mask)
    contours, _ = cv.findContours(full_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    full_centroid = get_centroid(contours[0])

    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)
    centroid = get_centroid(largest_contour)

    distance = euclidean_distance(full_centroid, centroid)

    if (distance < centroid_threshold):
        # Draw the contour on the filtered mask
        cv.drawContours(filtered_mask, [largest_contour], 0, 255, thickness=cv.FILLED)

    return filtered_mask



def get_labeled_mask(image, contour, idx):
    image = gray_to_rgb(image=image)

    cx, cy = get_centroid(contour=contour)
    
    cv.drawContours(image, [contour], -1, (255,255,0), 2)
    cv.putText(image, str(idx+1), (cx-10, cy-10),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    
    return image