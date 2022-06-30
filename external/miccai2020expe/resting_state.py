import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

# studies_dir = "/storage/store2/work/btajini/ts_1024_10sub_task/studies_difumo_1024_testonly"


def distinguish_fake_real(X, X_f, method_name, filename_pattern):
    """
    Records the 5-Fold cross validated accuracy of 4 different classifiers.
    Results are recorded in a pandas dataframe where each line is of the form:
    method_name, name, score_split, split

    Parameters
    ----------
    X: np array of shape (n_samples, n_features)
        True rest data
    X_f: np array of shape (n_samples, n_features)
        Fake rest data
    method_name: str
        the name of the method used to produce fake data
    filename_pattern: str
        filename_pattern % method_name is the path result file
    """

    models = []
    # parameters for MLP only
    params_2L = {
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "constant",
        "momentum": 0.9,
        "learning_rate_init": 0.0001,
        "alpha": 0.00001,
        "random_state": 0,
        "batch_size": 32,
        "hidden_layer_sizes": (1024, 1024),
        "max_iter": 20000,
    }

    models.append((LinearDiscriminantAnalysis(), "LDA"))
    models.append((RandomForestClassifier(verbose=True), "RF"))
    models.append((MLPClassifier(verbose=True, **params_2L), "MLP"))
    models.append(
        (
            GridSearchCV(
                LogisticRegression(
                    solver="lbfgs",
                    tol=1e-4,
                    random_state=11,
                    penalty="l2",
                    max_iter=20000,
                    n_jobs=1,
                    verbose=True,
                ),
                {"C": [0.1, 0.01, 0.001, 1]},
                cv=5,
            ),
            "LogReg",
        )
    )

    Y_ = np.array([1] * len(X) + [0] * len(X_f))
    X_ = np.concatenate([X, X_f], axis=0)

    scores = []
    for model, name in models:
        print("Fitting %s " % name)
        score = cross_val_score(
            model,
            X_,
            Y_,
            cv=ShuffleSplit(n_splits=5, test_size=0.20, random_state=0),
            n_jobs=5,
        )
        for split, score_split in enumerate(score):
            scores.append((method_name, name, score_split, split))

    scores = pd.DataFrame(
        scores, columns=["method_name", "algo", "score", "split"]
    )
    scores.to_csv(filename_pattern % method_name)
