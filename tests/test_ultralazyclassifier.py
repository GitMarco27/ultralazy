import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ultralazy.hp_reader import HyperparametersReader
from ultralazy.ultralazyclassifier import UltraLazyClassifier
from ultralazy.utils.utils import filter_estimators_by_keys


def test_simple_pipeline_no_cv():
    """Testing a basic UltraLazyClassifier Pipeline"""

    classifiers = HyperparametersReader.get_estimators("classifier")

    classifiers = filter_estimators_by_keys(
        keys=["XGBClassifier", "LGBMClassifier"], estimators=classifiers
    )

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    ulc = UltraLazyClassifier(
        classifiers,
        source_hp_path="./resources/ultralazy_hp_tests.json",
        cross_validation=True,
        logs_path="./logs",
    )

    # Generate a synthetic binary classification dataset
    x, y = make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    _ = ulc.fit(x_train, x_test, y_train, y_test)
