from typing import List

import numpy as np
from joblib import Parallel, delayed
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
}


def create_augmentation(
    aug_names: List[str],
) -> transforms.Transform:
    augmentation_list = [AUGMENTATIONS[aug] for aug in aug_names]
    return tio.transforms.OneOf(augmentation_list)


def transform_based_augmentation(augmentation, mask, Z_inv, images_paths, labels, nb_fakes=10, n_jobs=1, verbose=0):

    def _create_fakes(image_path, task):
        print(f"Starting to augment {image_path}")

        image_tio = tio.ScalarImage(image_path)

        sub_X = [Z_inv.dot(image_tio.data.squeeze()[mask])]

        for _ in range(nb_fakes):
            # transform
            trf_img_tio = augmentation(image_tio)

            # project
            trf_difumo_vec = Z_inv.dot(trf_img_tio.data.squeeze()[mask])

            sub_X.append(trf_difumo_vec)
        print(f"Finished to augment {image_path}")
        return np.vstack(sub_X), np.vstack([task] * (1 + nb_fakes))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    ret = parallel(delayed(_create_fakes)(image_path, task) for image_path, task in zip(images_paths, labels))

    X, Y = zip(*ret)

    return np.vstack(X), np.vstack(Y)
