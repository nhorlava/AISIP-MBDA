# Import libraries
import os
import pickle
import json

import numpy as np
import pandas as pd

from nilearn.datasets import fetch_neurovault


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
    nv_file = os.path.join(nv_filepath, "nv_meta.p")

    if download:

        print("Download from Neurovault API...")

        # Create folders (if not already exists)
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(nv_filepath, exist_ok=True)

        # Get output file name

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

def get_dataset_labels(base_path, study="hcp"):
    """Fetches the relavant metadata from json files contained in base_path and
    creates labels in format compatible with the training pipeline

    Parameters
    ----------
    base_path : str
        Path to folder where neurovault data was downloaded.
    study : str, optional
        by default "hcp"

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the subjects and contrast as used in the training
        scripts.
    """

    Y = list()
    for fname in os.listdir(base_path):
        if fname.endswith(".json") and fname.startswith("image_"):
            with open(os.path.join(base_path, fname), "r") as f:
                metadata = json.load(f)
            img_filename= f"image_{metadata['id']}.nii.gz"
            if os.path.exists(os.path.join(base_path, img_filename)):
                Y.append({
                    "study": study,
                    "subject_id": metadata["name"].split("_")[0],
                    "contrast": metadata["contrast_definition"],
                    "meta_path": os.path.join(base_path, fname),
                    "path": os.path.join(base_path, img_filename),
                })
    return pd.DataFrame(Y)


def filter_subjects_with_all_tasks(Y, n_tasks=23):
    """Screens the labels dataframe Y and filters out subjects which do not have
    data for the minimum desired number of tasks

    Parameters
    ----------
    Y : pandas.DataFrame
        Dataframe of contrast and subjects, as obtained using
        get_dataset_labels.
    n_tasks : int, optional
        Minimum number of tasks per subject, by default 23

    Returns
    -------
    pandas.DataFrame
        Masked labels dataframe Y.
    """
    mask = np.zeros(Y.shape[0]).astype(bool)
    for subject in Y["subject"].unique():
        n_tasks_subj = Y[Y["subject"] == subject].shape[0]
        if n_tasks_subj >= n_tasks:
            mask = np.logical_or(mask, (Y["subject"] == subject).values)

    mask = pd.Series(mask)
    return Y[mask].reset_index(drop=True)
