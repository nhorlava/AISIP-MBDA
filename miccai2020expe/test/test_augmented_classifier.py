from miccai2020expe.task import AugmentedClassifier
from condica.main import condica
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.random.rand(100, 10)
Y = np.random.randint(0, 4, size=100)
A = np.random.rand(10, 10)


def test_shapes():
    lda = LinearDiscriminantAnalysis()
    f = lambda X, Y: condica(A, X, Y, 10, n_quantiles=10)
    clf = AugmentedClassifier(lda, f)
    clf.fit(X, Y)
