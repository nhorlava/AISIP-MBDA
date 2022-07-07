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
        difumo_matrices_path:str ="../../Data/hcp900_difumo_matrices/",
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
        print("Iterating over the dataset until we reach the desired number of samples")
        for ind, _ in tqdm(filtered_pd.iterrows()):
            if generated_ind>=num_generated_samples:
                break

            row = get_row(NeuroData_obj.__getitem__(ind))
            projected_df = pd.concat([projected_df, row], ignore_index=True)
            generated_ind+=1

    augmentation_name_str = '-'.join(augmentation_name) if isinstance(augmentation_name, list) else augmentation_name
    projected_df["augmentation"] = augmentation_name_str
    os.makedirs(save_path, exist_ok=True)

    projected_df.to_csv(os.path.join(save_path,  f"difumo-augm_{augmentation_name_str}.csv"))



if __name__=="__main__":
    import os

    labels_pd = get_dataset_labels(
        base_path="/Users/nastassya.horlava/Documents/Projects/ParisSummerSchool/Code/Data/neurovault/neurovault/collection_4337")
    filtered_pd = labels_pd
    # filtered_pd = filter_subjects_with_all_tasks(labels_pd)

    project_difumo(labels_pd, save_path="../../Data/HCP_difumo", num_generated_samples=15)

    augmentation_list = ["RandomElasticDeformation", "RandomMotion",  "RandomGhosting",
                         "RandomSpike",  "RandomBiasField",   "RandomBlur",  "RandomNoise",  "RandomGamma",   "RandomFlip"]

    for augm in augmentation_list:
        project_difumo(labels_pd, prior_augmentation=True, num_generated_samples=15, augmentation_name=[augm], save_path="../../Data/HCP_difumo")





