from typing import List

from tqdm import tqdm
import numpy as np
from torchio import transforms
import torchio as tio


AUGMENTATIONS = {
    "RandomElasticDeformation": tio.RandomElasticDeformation(
        num_control_points=16,
        max_displacement=2
    ),
    "RandomMotion": tio.RandomMotion(
        degrees=0.2,
        translation=0.2,
        num_transforms=2,
    ),
    "RandomGhosting": tio.RandomGhosting(
        num_ghosts=1,
        intensity=0.02,
        restore = 1.0,
    ),
    "RandomSpike": tio.RandomSpike(
        num_spikes=2,
        intensity=1.15,
    ),
    "RandomBiasField": tio.RandomBiasField(
        order=1,
        coefficients=0.05,
    ),
    "RandomBlur": tio.RandomBlur(
        std=1.05
    ),
    "RandomNoise":tio.RandomNoise(
        mean=0.3,
        std=0.5,
    ),
    "RandomGamma": tio.RandomGamma(
        log_gamma=0.075
    ),
    "RandomFlip": tio.RandomFlip(
        flip_probability=1.0
    ),
    "None": None,
}


def create_augmentation(
    aug_names: List[str],
) -> transforms.Transform:
    augmentation_list = [AUGMENTATIONS[aug] for aug in aug_names]
    return tio.transforms.OneOf(augmentation_list)


def transform_based_augmentation(
    images_paths,
    augmentation,
    Z_inv,
    mask,
    nb_fakes=10
):
    X = list()
    for image_path in tqdm(images_paths):
        image_tio = tio.ScalarImage(image_path)

        for _ in range(nb_fakes):
            # transform
            trf_img_tio = augmentation(image_tio)

            # project
            trf_difumo_vec = Z_inv.dot(trf_img_tio.data.squeeze()[mask])

            X.append(trf_difumo_vec)

    return np.vstack(X)
