import os
from joblib import load
import pandas as pd


def load_reduced_hcp(
    file_task="reduced_data/modl_%i/T/studies/data_hcp.pt",
    file_rest="reduced_data/modl_%i/T/hcp_rest/rest.pt",
    restrict=True,
    toy=False,
    task_only=False,
    rest_only=False,
):
    """
    Parameters
    ----------

     pattern_task: str
        path to .pt file for task
    pattern_rest: str
        path to .pt file for rest
    restrict: bool
        True if we restrict to subjects with task and rest data
        and subjects with number of task = 23
    task_only: bool
        only task data are loaded
    rest_only: bool
        only rest data are loaded

    Returns
    -------
    X_t : np array

    Y_t : pandas

    X_r : np array

    Y_r : pandas
    """
    if toy:
        file_task = file_task + "toy"
        file_rest = file_rest + "toy"

    if rest_only is False:
        X_t, Y_t = load(file_task)

    if task_only is False:
        X_r, paths_rest = load(file_rest)
        Y_r = []
        for path in paths_rest:
            subject = path.split("/MNINonLinear")[0].split("/")[-1]
            session, modality = (
                path.split("rfMRI_REST")[1].split("/")[0].split("_")
            )
            Y_r.append((subject, session, modality))
        Y_r = pd.DataFrame(Y_r, columns=["subject", "session", "modality"])

    if restrict:
        # Restrict to subject with 23 tasks
        i_task = Y_t.groupby("subject")["contrast"].transform("size") == 23
        Y_t = Y_t[i_task]
        X_t = X_t[i_task]

        if task_only is False and rest_only is False:
            # Restrict to subject with rest and task
            u_t = set(Y_t["subject"].unique())
            u_r = set(Y_r["subject"].unique())
            u_inter = u_t.intersection(u_r)
            i_inter = Y_t["subject"].isin(u_inter)
            Y_t = Y_t[i_inter]
            X_t = X_t[i_inter]
            i_inter = Y_r["subject"].isin(u_inter)
            X_r = X_r[i_inter]
            Y_r = Y_r[i_inter]

    if task_only:
        X_r, Y_r = None, None

    if rest_only:
        X_t, Y_t = None, None

    return X_t, Y_t, X_r, Y_r


def preprocess_label(Y_t, use_dict=None, return_dict=False):
    """
    Preprocess label so that they match classifier like format

    Parameters
    ----------
    Y_t: pandas DataFrame
        labels to preprocess
    use_dict: dictionary
        If a dictionary labels -> class number is already known
    return_dict: bool
        If true, returns the dictionary labels -> class number
    Returns
    --------
    Y: np array
        np array where each label is replaced by its class number
    dict (optional): dict
        dictionary label -> class number (only returned if return_dict is True)
    """
    if use_dict is None:
        Y_dict = {v: k for k, v in enumerate(Y_t["contrast"].unique())}
    else:
        Y_dict = use_dict

    if return_dict:
        return Y_t["contrast"].apply(lambda x: Y_dict[x]).values, Y_dict
    else:
        return Y_t["contrast"].apply(lambda x: Y_dict[x]).values
