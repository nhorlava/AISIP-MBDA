import os

# Import Libraries
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from nilearn.image import resample_to_img
import nibabel

import numpy as np
from joblib import Parallel, delayed


def get_DiFuMo_map(dimension=1024):
    """
    Get the map for DiFuMo projection (unsampled).
    :param dimension: int
        The dimension of the projector DiFuMo dictionary.
    """

    # Load DiFuMo map
    difumo = fetch_atlas_difumo(dimension=dimension)
    difumo_maps = nibabel.load(difumo["maps"])

    return difumo_maps


def get_mask(neurovault, difumo, save=False):
    """
    Creates the mask from neurovault data and difumo operator.
    :param neurovault: Neurovault data. Must be the output of the fetching function in fecthing.py.
    :param difumo: DiFuMo operator. Must be the output of get_DiFuMo_map function.
    :param save: bool. Whether or not saving the resulting mask. Since the output is big,
                       it is suggested to not save it.
    Output:
    :param maps_data: the projection of an image. Required to compute the projector Z.
    :param mask: The actual mask.
    """

    # Get an example image
    image = nibabel.load(neurovault["images"][0])

    # Resample the image
    print("Resampling. Please wait...")
    resampled_difumo_maps = resample_to_img(difumo, image, interpolation="nearest")

    # Initialize the mask
    masker = NiftiMapsMasker(resampled_difumo_maps)

    # Fit the mask (TAKES TIME)
    print("Fitting the maks. Please wait...")
    _ = masker.fit_transform(image)

    # We threshold the first image to get a mask of the brain voxels
    maps_img = masker.maps_img_
    maps_data = maps_img.get_data()

    # Treshold the pixels after summing along the 1024-length dimension
    mask = np.sum(maps_data, axis=-1)
    mask = mask > 0

    # Save if required
    if save:
        difumo_matrices_dir = "./hcp900_difumo_matrices/"
        os.makedirs(difumo_matrices_dir, exist_ok=True)
        np.save(os.path.join(difumo_matrices_dir, "mask"), mask)

    return maps_data, mask


def get_projector_from_mask(maps_data, mask, save=False):
    """
    Creates the projector from the output of the function get_mask.
    :param maps_data: the projection of an image. Required to compute the projector Z.
    :param mask: The actual mask.
    :param save: bool. Whether or not saving the resulting mask. Since the output is big,
                       it is suggested to not save it.
    Output:
    :param Z: The projector operator.
    """

    Z = maps_data[mask, :]

    # Save if required
    if save:
        difumo_matrices_dir = "./hcp900_difumo_matrices/"
        os.makedirs(difumo_matrices_dir, exist_ok=True)
        np.save(os.path.join(difumo_matrices_dir, "Z"), Z)

    return Z


def _project_one(img, Z_inv, mask):
    return Z_inv.dot(img.data.squeeze()[mask])


def project_to_difumo(img_tio, Z_inv, mask, n_jobs=1):
    parallel = Parallel(n_jobs=n_jobs)
    projected_imgs = parallel(
        delayed(_project_one)(img, Z_inv, mask) for img in img_tio
    )
    return np.vstack(projected_imgs)
