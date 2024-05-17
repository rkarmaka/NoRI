import tifffile


def read_tiff_and_extract_channels(file_path, separate_channels=True):
    """
    Reads a TIFF file, extracts protein and lipid channels, and returns 2 image stacks with image patches.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        list: List containing two image channels.
    """
    try:
        # Read TIFF file. Image is read in (c x h x w) format
        image = tifffile.imread(file_path)


        if separate_channels:
            # Check the number of channels in the image
            if image.shape[0] == 7:
                print("Image has 7 channels")
                protein_channel = image[1, :, :]
                lipid_channel = image[2, :, :]
                endomucine_channel = image[3, :, :]
                ch4 = image[4, :, :]
                ch5 = image[5, :, :]
                ch6 = image[6, :, :]

                return [protein_channel, lipid_channel, endomucine_channel, ch4, ch5, ch6]
            elif image.shape[0] == 6:
                print("Image has 6 channels")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]
                endomucine_channel = image[2, :, :]                
                ch4 = image[3, :, :]
                ch5 = image[4, :, :]
                ch6 = image[5, :, :]

                return [protein_channel, lipid_channel, endomucine_channel, ch4, ch5, ch6]
            elif image.shape[0] == 3:
                print("Image has 3 channels. Assigning Ch1 as Protein and Ch2 as Lipid")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]
                endomucine_channel = image[2, :, :]

                return [protein_channel, lipid_channel, endomucine_channel]
            else:
                print(f"Image has {image.shape[0]} channels")
                return None
        else:
            return image

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None





