import argparse

import numpy as np

from utils import fetching
from training import do_classif
from data_augmentation import (
    AUGMENTATIONS,
    create_augmentation,
    transform_based_augmentation,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_images",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-j", "--njobs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--data-dir",
        default="/storage/store2/data/",
        type=str,
    )
    parser.add_argument(
        "-v", "--verbose",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--nb_fakes",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--train_size",
        default=100,
        type=int
    )
    parser.add_argument(
        "--nsplits",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--output_path",
        default="results/hcp_all_augs.csv",
        type=str,
    )
    args = parser.parse_args()

    print(" --- LOADING DATA --- ")
    data = fetching.fetch_nv(
        args.data_dir,
        max_images=args.max_images,
        verbose=args.verbose,
    )

    print(" --- CREATING LABELS --- ")
    labels = fetching.get_dataset_labels(data)

    # TODO: Should be replaced by function loading or creating matrices
    print(" --- LOADING/CREATING DIFUMO PROJECTORS --- ")
    Z_inv = np.load("hcp900_difumo_matrices/Zinv.npy")
    mask = np.load("hcp900_difumo_matrices/mask.npy")

    all_augs = create_augmentation(list(AUGMENTATIONS.keys()))

    def augmenter(X, Y):
        return transform_based_augmentation(
            all_augs, mask, Z_inv, X, labels=Y, nb_fakes=args.nb_fakes,
            n_jobs=args.njobs,
            verbose=args.verbose
        )

    print(" --- STARTING CLASSIFICATION --- ")
    do_classif(
        data["images"],
        Z_inv,
        mask,
        labels,
        augmenter,
        method_name="all_augs",
        filename=args.output_path,
        train_size=args.train_size,
        n_splits=args.nsplits,
        n_jobs=1,
    )
