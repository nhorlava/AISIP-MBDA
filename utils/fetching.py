# Import libraries
import os
import pickle

from nilearn.datasets import fetch_neurovault, fetch_atlas_difumo

# ================================
# === FETCHING FROM NEUROVAULT ===
# ================================
def fetch_nv(out_folder='../../Data', nv_filepath='../../cache',
             download=False,
             max_images=None):
    """
    Loads neurovault into memory, either downloading it from the web-API or
    loading it from the disk.
    :param out_folder: str
        Path where the data is downloaded.
    :param nv_filepath: str
        Pickle file where the full data is saved
        (for faster loading than the fetch_neurovault).
    :param download: bool, default=False
        If True: the data is downloaded from the web-API.
    :param max_images: int, default=None
        Number of images to load from neurovalt.
    :return: Bunch
        A dict-like object containing the data from fMRIs fetched from
        Neurovault.
    """

    # Download and save to disk or load from disk
    if download:

        print("Download from Neurovault API...")

        # Create folders (if not already exists)
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(nv_filepath, exist_ok=True)

        # Get output file name
        nv_file = os.path.join(nv_filepath, "nv_meta.p")
        
        # Fetch Neurovault (collection_id = 4337)
        neurovault = fetch_neurovault(max_images=max_images,
                                      collection_terms = {},
                                      image_terms = {},
                                      data_dir = out_folder,
                                      mode = "download_new",
                                      verbose = 2,
                                      collection_id = 4337)
        
        # Save the output
        with open(nv_file, 'wb') as f:
            pickle.dump(neurovault, f)

    else:
        print("Load pre-fetched data from Neurovault...")

        # Load the file
        with open(nv_file, 'rb') as f:
            neurovault = pickle.load(f)

    n_fmri_dl = len(neurovault.images)
    print(f"Number of (down)loaded fMRI files: {n_fmri_dl}")

    return neurovault