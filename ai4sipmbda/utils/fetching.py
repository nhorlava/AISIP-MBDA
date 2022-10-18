# Import libraries
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from nilearn.datasets import fetch_neurovault


# ================================
# === FETCHING FROM NEUROVAULT ===
# ================================
def fetch_nv(
    data_dir: str,
    max_images: int = None,
    verbose=2,
) -> Bunch:
    """
    Loads neurovault into memory, either downloading it from the web-API or
    loading it from the disk.

    Parameters
    ----------
    data_dir: str
        Path where the data is downloaded.
    max_images: int, default=None
        Number of images to load from neurovalt.

    Returns
    -------
    sklearn.utils.Bunch
        A dict-like object containing the data from fMRIs fetched from
        Neurovault.
    """
    # Fetch Neurovault (collection_id = 4337)
    neurovault = fetch_neurovault(
        max_images=max_images,
        collection_terms={},
        image_terms={},
        data_dir=data_dir,
        mode="download_new",
        verbose=verbose,
        collection_id=4337
    )

    return neurovault


def get_dataset_labels(
    data: Bunch,
    study: str = "hcp"
) -> pd.DataFrame:
    """Fetches the relavant metadata from json files contained in base_path and
    creates labels in format compatible with the training pipeline

    Parameters
    ----------
    data : sklearn.utils.Bunch
        
    study : str, optional
        by default "hcp"

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the subjects and contrast as used in the training
        scripts.
    """

    Y = list()
    for metadata in data["images_meta"]:
        Y.append({
            "study": study,
            "subject": metadata["name"].split("_")[0],
            "contrast": metadata["contrast_definition"],
        })
    return pd.DataFrame(Y)


def filter_subjects_with_all_tasks(X, Y, n_tasks=23):
    """Screens the labels dataframe Y and filters out subjects which do not
    have data for the minimum desired number of tasks

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
