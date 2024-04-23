from tifffile import imread


## Read file
def read_tiff_and_extract_channels(file_path):
    """
    Reads a TIFF file, extracts protein and lipid channels, and returns 2 image stacks with image patches.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:

    """

    try:
        # Read TIFF file. Image is read in (c x h x w) format
        image = imread(file_path)

        # Check if the image has 6 or 7 channels
        if image.shape[0]==7:
            print("Image has 7 channels")
            image_ch1 = image[1,:,:]
            image_ch2 = image[2,:,:]

        elif image.shape[0]==6:
            print("Image has 6 channels")
            image_ch1 = image[0,:,:]
            image_ch2 = image[1,:,:]

        else:
            print("Image has {image.shape[0]} channels")


    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return [image_ch1, image_ch2]





