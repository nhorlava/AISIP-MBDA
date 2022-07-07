from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torchvision.transforms as transforms
import torchio as tio
def get_transforms(
     **kwargs
) -> Tuple[transforms.Compose, Any]:
    """
    Outputs the transformations that will be applied to the dataset

    Args:
        normalize: if True will perform MinMaxNormalization.
        data_augmentation: list of data augmentation performed on the training set.

    Returns:
        transforms to apply in train and evaluation mode / transforms to apply in evaluation mode only.
    """
    augmentation_dict = {
        "RandomElasticDeformation": tio.RandomElasticDeformation(num_control_points = 16, max_displacement = 2),
        "RandomMotion":tio.RandomMotion(degrees = 0.2, translation = 0.2, num_transforms = 2),
        "RandomGhosting": tio.RandomGhosting(num_ghosts = 1, intensity = 0.02, restore = 1.0),
        "RandomSpike": tio.RandomSpike(num_spikes = 2, intensity = 1.15),
        "RandomBiasField": tio.RandomBiasField(order = 1,coefficients=0.05 ),
        "RandomBlur": tio.RandomBlur(std = 1.05),
        "RandomNoise":tio.RandomNoise(mean = 0.3, std = 0.5),
        "RandomGamma": tio.RandomGamma(log_gamma=0.075),
        "RandomFlip": tio.RandomFlip(flip_probability=1.0),
        # "None": None,
    }
    if kwargs["data_augmentation"]:
        if kwargs["data_augmentation"] == "all":
            augmentation_list = augmentation_dict.values()
        else:
            augmentation_list = [
                augmentation_dict[augmentation] for augmentation in kwargs["data_augmentation"]
            ]
    else:
        augmentation_list = []


    transformations_list = [transforms.ToTensor()]

    all_transforms = transforms.Compose(transformations_list)
    augmentation_transforms = tio.transforms.OneOf(augmentation_list)

    return augmentation_transforms, all_transforms