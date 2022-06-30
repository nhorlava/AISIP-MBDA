from miccai2020expe.task import do_classif
from miccai2020expe.task_loading import load_reduced_hcp
import numpy as np
from condica.main import condica
from joblib import load

studies = [
    ("camcan", 100),
    ("archi", 40),
    ("brainomics", 50),
    ("la5c", 100),
    ("pinel2012archi", 40),
    ("pinel2009twins", 35),
    ("pinel2007fast", 70),
    ("hcp", 100),
]

for study, train_size in studies:
    path = (
        "/data/parietal/store2/work/btajini/ts_1024_10sub_task/studies_difumo_1024_testonly/data_%s_smooth_8.pt"
        % study
    )
    if study == "hcp":
        X_t, Y_t, _, _ = load_reduced_hcp(
            file_task=path, file_rest=None, task_only=True
        )
    else:
        X_t, Y_t = load(path)
    m = len(X_t)
    A = np.load(
        "/data/parietal/store2/work/btajini/ts_1024_10sub_task/data_generation_task/data_generation/estimated_mix_mat_900_new.npy"
    )
    f = lambda X, Y: condica(A, X, Y, 200)
    do_classif(
        X_t,
        Y_t,
        f,
        method_name="CondICA",
        filename="../results/%s_%s.csv" % (study, "CondICA"),
        train_size=train_size,
        n_splits=10,
        n_jobs=10,
    )
    f = None
    do_classif(
        X_t,
        Y_t,
        f,
        method_name="Original",
        filename="../results/%s_%s.csv" % (study, "Original"),
        train_size=train_size,
        n_splits=10,
        n_jobs=10,
    )
