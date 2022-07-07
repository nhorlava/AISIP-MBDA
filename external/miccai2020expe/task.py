from joblib.parallel import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

from task_loading import preprocess_label


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, f):
        """
        Trains a classifier using an augmentation method.
        Parameters
        ----------
        model: BaseEstimator
            The classifier used
        f: function (X, Y) -> X_fake, Y_fake
            The data augmentation function that
            generates fake (labeled) data from
            input data
        """

        self.model = model
        self.f = f

    def fit(self, X, Y):
        if self.f is None:
            return self.model.fit(X, Y)
        else:
            X_fake, Y_fake = self.f(X, Y)
            return self.model.fit(
                np.row_stack([X_fake, X]), np.concatenate([Y_fake, Y])
            )

    def predict(self, X, y):
        return self.model.predict(X, y)

    def score(self, X, y):
        return self.model.score(X, y)


def do_classif(
    X, Y, f, method_name, filename, train_size, n_splits=5, n_jobs=5
):
    """
    Tries 4 different classifier with the given augmentation method
    Parameters
    ----------
    X: np array of shape (n_samples, n_features)
        Input data
    Y: np array of shape (n_samples,)
        Labels
    f: function of X, Y
        f returns fake data and fake labels
    method_name: str
        the name of the method used to produce fake data
    filename: str
        filename is the path result file
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    n_jobs: int, default: None
        The maximum number of concurrently running jobs,
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

    models.append(
        (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "LDA")
    )
    models.append((RandomForestClassifier(verbose=True), "RF"))
    # models.append((MLPClassifier(verbose=True, **params_2L), "MLP"))
    # models.append(
    #     (
    #         GridSearchCV(
    #             LogisticRegression(
    #                 solver="lbfgs",
    #                 tol=1e-4,
    #                 random_state=11,
    #                 penalty="l2",
    #                 max_iter=20000,
    #                 n_jobs=1,
    #                 verbose=True,
    #             ),
    #             {"C": [0.1, 0.01, 0.001, 1]},
    #             cv=5,
    #         ),
    #         "LogReg",
    #     )
    # )

    def do_split(split, X, Y, f, model, subjects):
        train, test = split
        train, test = subjects[train], subjects[test]
        train = Y["subject"].isin(train)
        test = Y["subject"].isin(test)
        Y_train, dict = preprocess_label(Y[train], return_dict=True)
        Y_test = preprocess_label(Y[test], use_dict=dict)
        X_train = X[train]
        X_test = X[test]
        clf = AugmentedClassifier(model, f)
        clf.fit(X_train, Y_train)
        score_split = clf.score(X_test, Y_test)
        return score_split

    scores = []
    for model, name in models:
        subjects = Y["subject"].unique()
        sf = ShuffleSplit(
            n_splits=n_splits, train_size=train_size, random_state=0
        )
        scores_split = Parallel(verbose=True, n_jobs=n_jobs)(
            delayed(do_split)(split, X, Y, f, model, subjects)
            for split in sf.split(range(len(subjects)))
        )
        for i_split, score_split in enumerate(scores_split):
            scores.append((method_name, name, score_split, i_split))

    scores = pd.DataFrame(
        scores, columns=["method_name", "algo", "score", "split"]
    )
    scores.to_csv(filename)
