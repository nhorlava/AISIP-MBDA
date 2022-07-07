
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import nibabel
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ai4sipmbda.utils import fetching, difumo_utils
import torchio as tio

class NeuroData(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        all_transformations: Optional[Callable],
        label: str = None,
        augmentation_transformations: Optional[Callable] = None,
        eval_mode: bool = False,
        difumo_matrices_path: str = None
    ):

        self.all_transformations = all_transformations
        self.augmentation_transformations = augmentation_transformations
        self.eval_mode = eval_mode
        self.label = label
        self.df = data_df

        self.difumo_matrices_path =  difumo_matrices_path

        self.mask = np.load(os.path.join(self.difumo_matrices_dir, "mask.npy"))
        self.pseudo_inv_Z = np.load(os.path.join(self.difumo_matrices_dir, "Zinv.npy"))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        # import nibabel as nib
        participant, image_path, label = self._get_meta_data(idx)
        image_tio = tio.ScalarImage(image_path)
        if self.eval_mode:
            image_tio = self.augmentation_transformations(image_tio)

        difumo_proj = self.pseudo_inv_Z.dot(image_tio.data.squeeze().numpy().get_data()[self.mask])
        difumo_proj = self.all_transformations(difumo_proj)


        sample = {
            "image":  difumo_proj,
            "label": label,
            "participant_id": participant,
            "image_path": image_path,
        }

        return sample

    def _get_meta_data(self, idx: int) -> Tuple[str, str, int]:
        """
        Gets all meta data necessary to compute the path with _get_image_path

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            participant (str): ID of the participant.
            path (str): path to the file.
            label (str or float or int): value of the label to be used in criterion.
        """

        participant = self.df.loc[idx, "id"]
        path = self.df.loc[idx, "path"]

        # print(idx, participant,session )
        if self.label is not None:
            target = self.df.loc[idx, self.label]
            label = self.label_fn(target)
        else:
            label = -1

        return participant, path, label



from ai4sipmbda.transforms import get_transforms
if __name__=="__main__":
    import pickle
    import os

    nv_filepath = "../../cache/"
    nv_file = os.path.join(nv_filepath, "nv_meta.p")

    # Number of images to fetch
    max_images = 10

    # Fetching...
    neurovault = fetching.fetch_nv(max_images=max_images, download=False)

    img_pd = pd.DataFrame.from_dict(neurovault["images"])
    img_pd = img_pd.rename(columns={0: "path"})

    meta_pd = pd.DataFrame.from_dict(neurovault["images_meta"])

    concated_pd= pd.concat([img_pd, meta_pd])

    augmentation_transforms, all_transforms = get_transforms(data_augmentation= "all")
    NeuroData_obj = NeuroData(data_df=concated_pd, all_transformations=all_transforms, label = "contrast")


