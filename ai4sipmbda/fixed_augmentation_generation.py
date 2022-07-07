import pandas as pd

from ai4sipmbda.transforms import get_transforms
from ai4sipmbda.utils.fetching import get_dataset_labels, filter_subjects_with_all_tasks
from ai4sipmbda.data import NeuroData
import pandas as pd
from typing import List
from tqdm import tqdm

def get_row(NeuroDataItem):
    difumo_cols = [f"difumo_{i}" for i in range(len(NeuroDataItem["difumo_vector"]))]
    meta_cols = ["subject_id", "label","image_path"]
    all_cols = meta_cols +difumo_cols
    concatenated_vector  = list([*[NeuroDataItem[key] for key in meta_cols], *NeuroDataItem["difumo_vector"]])
    difumo_vector_pd = pd.DataFrame.from_records([concatenated_vector], columns = all_cols)

    return difumo_vector_pd

def project_difumo(
        df,
        difumo_matrices_path:str,
        prior_augmentation: bool = False,
        augmentation_name: List [str] = None,
        num_generated_samples: int = None,
        save_path: str  = None
            ):
    if num_generated_samples is None:
        num_generated_samples = df.shape[0]
    if prior_augmentation:
        augmentation_transforms, all_transforms = get_transforms(data_augmentation=augmentation_name)
    else:
        augmentation_transforms, all_transforms = None, None

    NeuroData_obj = NeuroData(data_df=df, all_transformations=all_transforms,
                              augmentation_transformations=augmentation_transforms,
                              label="contrast",
                              difumo_matrices_path=difumo_matrices_path, eval_mode=False)

    projected_df = pd.DataFrame()

    generated_ind= 0
    while generated_ind<num_generated_samples:
        print(f"Iterating over the dataset until we reach the desired number of samples, {num_generated_samples - generated_ind} samples left")
        num_samples_from_df = max((num_generated_samples - generated_ind)%df.shape[0], num_generated_samples)
        for ind, _ in tqdm(df.loc[:num_samples_from_df].iterrows()):
            if generated_ind>=num_generated_samples:
                break

            row = get_row(NeuroData_obj.__getitem__(ind))
            projected_df = pd.concat([projected_df, row], ignore_index=True)
            generated_ind+=1

    augmentation_name_str = '-'.join(augmentation_name) if isinstance(augmentation_name, list) else augmentation_name
    projected_df["augmentation"] = augmentation_name_str
    os.makedirs(save_path, exist_ok=True)

    projected_df.to_csv(os.path.join(save_path,  f"difumo-augm_{augmentation_name_str}.csv"))


def execute_projections(base_dataset_path:str , difumo_maps_path:str, save_path:str, num_samples: int):

    labels_pd = get_dataset_labels(
        base_path=base_dataset_path)
    # filtered_pd = labels_pd
    # filtered_pd = filter_subjects_with_all_tasks(labels_pd)

    project_difumo(labels_pd, difumo_matrices_path=difumo_maps_path,  save_path=save_path, num_generated_samples=num_samples)

    augmentation_list = ["RandomElasticDeformation", "RandomMotion",  "RandomGhosting",
                         "RandomSpike",  "RandomBiasField",   "RandomBlur",  "RandomNoise",  "RandomGamma",   "RandomFlip"]

    for augm in augmentation_list:
        project_difumo(labels_pd, prior_augmentation=True, num_generated_samples=num_samples, augmentation_name=[augm], difumo_matrices_path=difumo_maps_path,  save_path=save_path)



if __name__=="__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Project images into difumo space')
    parser.add_argument('--base_dataset_path', type=str,
                        help='path to your dataset')

    parser.add_argument('--difumo_maps_path', type=str,
                        help='path to your the stored difumo maps')

    parser.add_argument('--save_path', type=str,
                        help='path where to store projections')

    parser.add_argument('--num_samples', type=int, default = None,
                        help='How many samples to generated for augmented samples')

    args = parser.parse_args()

    execute_projections(base_dataset_path = args.base_dataset_path,
                        difumo_maps_path = args.difumo_maps_path,
                        save_path = args.save_path,
                        num_samples = args.num_samples)

