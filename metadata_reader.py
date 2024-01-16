import os
import tifffile
import numpy as np
import pandas as pd
import math

def extract_channel_information(channel, image):
    """
    Extracts statistical information for a specific channel in the image.

    Args:
        channel (int): Channel index.
        image (numpy.ndarray): Image array.

    Returns:
        dict: Dictionary containing statistical information for the channel.
    """
    # Extract channel-specific information
    channel_info = {
        f"Ch{channel + 1}_Dimension": f"{image[channel, :, :].shape[0]}x{image[channel, :, :].shape[1]}",   # Dimension
        f"Ch{channel + 1}_Mean_Intensity": np.mean(image[channel, :, :]),                                   # Mean Intensity
        f"Ch{channel + 1}_Median_Intensity": np.median(image[channel, :, :]),                               # Median Intensity
        f"Ch{channel + 1}_Intensity_Range": [np.min(image[channel, :, :]), np.max(image[channel, :, :])]    # Intensity Range
    }

    # Calculate percentage of saturated pixels
    total_pixels = image[channel, :, :].size
    saturated_pixels = np.sum(image[channel, :, :] == image[channel, :, :].max())
    channel_info[f"Ch{channel + 1}_Percent_Saturation"] = saturated_pixels * 100 / total_pixels

    return channel_info

def read_tiff_and_extract_metadata(file_path):
    """
    Reads a TIFF file, extracts metadata, and returns a dictionary with image information.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        dict or None: Dictionary with image information or None if an error occurs.
    """
    try:
        # Read TIFF file
        image = tifffile.imread(file_path)

        # Create a dictionary for each image
        image_info = {
            'File_Path': file_path,
            'File_Name': file_path.split("/")[-1].split(".")[0],  # Extracting file name without extension
            'Number_of_Channels': image.shape[0],
            'Image_Height': image.shape[1],
            'Image_Width': image.shape[2],
            'Bit_Depth': math.ceil(math.log(image.max(), 2)),
        }

        # Display information for the current image
        print(f"Processing Image: {image_info['File_Name']}")

        # Collect information for each channel
        for i in range(image.shape[0]):
            channel_info = extract_channel_information(i, image)
            image_info.update(channel_info)

        # Return the image information
        return image_info

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_tiffs_in_directory(directory):
    """
    Reads all TIFF files in a directory, extracts metadata, and returns a DataFrame.

    Args:
        directory (str): Directory path.

    Returns:
        pandas.DataFrame: DataFrame containing metadata for each image.
    """
    metadata_list = []

    # List all files in the directory with .tif or .tiff extensions
    files = [f for f in os.listdir(directory) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

    for file_name in sorted(files):
        file_path = os.path.join(directory, file_name)
        metadata = read_tiff_and_extract_metadata(file_path)

        if metadata:
            metadata_list.append(metadata)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(metadata_list)

    return df


def read_tiff_file_names(root_folder):
    tiff_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                file_path = os.path.join(root, file)
                tiff_files.append(file_path)

    return tiff_files


if __name__ == "__main__":
    # Replace '/path/to/your/directory' with the actual path to your directory containing TIFF files
    root_folder = "/Users/ranit/IAC/Project/Will Trim/_DATA/Batch_2"

    tiff_files = read_tiff_file_names(root_folder=root_folder)

    metadata_list = []
    print(f'{len(tiff_files)} files found')

    for tiff_file in tiff_files:
        metadata = read_tiff_and_extract_metadata(tiff_file)

        if metadata:
            metadata_list.append(metadata)

    # Generate metadata DataFrame
#    metadata_df = read_tiffs_in_directory(tiff_directory)

    # Save the DataFrame to a CSV file in the specified path
    csv_path = os.path.join(root_folder, 'metadata_table.csv')
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(csv_path, index=False)

    # Display the DataFrame
    print(metadata_df)
    print(f"CSV file saved at: {csv_path}")
