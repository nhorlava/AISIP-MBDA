import os
from condica.main import condica
from miccai2020expe.resting_state import distinguish_fake_real
import numpy as np

os.makedirs("../results/", exist_ok=True)
X_rest = np.load(
    "/data/parietal/store2/work/btajini/ts_1024_200sub/time_series1024_200sub.npy"
)
A = np.load(
    "/data/parietal/store2/work/btajini/ts_1024_10sub_task/data_generation_task/data_generation/estimated_mix_mat_900_new.npy"
)
X_rest = np.row_stack(X_rest)
# Such that SA^T = X
X_fake = condica(A, X_rest, nb_fakes=len(X_rest))
rng = np.random.RandomState(0)
I = rng.choice(np.arange(len(X_rest)), 10000)
distinguish_fake_real(X_rest[I], X_fake[I], "condica", "../results/accuracy_%s")

