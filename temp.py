import tifffile

def read_tiff_and_extract_metadata(file_path):
    try:
        # Read TIFF file
        image = tifffile.imread(file_path)
        print(file_path.split("/")[-1].split(".")[0])

        # Extract metadata
        metadata = tifffile.TiffFile(file_path).imagej_metadata

        # Display metadata
        print(f"File: {file_path}")
        print(f"Number of Layers: {image.shape[0]}")
        print(metadata["max"])
        print("Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Replace 'your_tiff_file.tif' with the actual path to your TIFF file
    tiff_file_path = '/home/bia/thinclient_drives/IAC/Project/Will Trim/DATA/from Seungeun/C3-Fused_Cy3.tif'

    read_tiff_and_extract_metadata(tiff_file_path)
