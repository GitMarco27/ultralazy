# Ultralazy

[![Unit Testing](https://github.com/GitMarco27/ultralazy/actions/workflows/python-app.yml/badge.svg)](https://github.com/GitMarco27/ultralazy/actions/workflows/python-app.yml)

![ultralazy_logo](./resources/ultralazy.png)

## Get started

```python
import json
import os

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ultralazy.hp_reader import HyperparametersReader
from ultralazy.ultralazyclassifier import UltraLazyClassifier
from ultralazy.utils.utils import filter_estimators_by_keys

# Getting all the available classifiers
classifiers = HyperparametersReader.get_estimators("classifier")

# Filtering only the target classifiers
classifiers = filter_estimators_by_keys(
    keys=["XGBClassifier", "LGBMClassifier"], estimators=classifiers
)

if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Loading GPT-4 Generated Hyperparameters
with open("./resources/ultralazy_hp_tests.json", "r", encoding="utf-8") as f:
    hp_dict = json.load(f)

ulc = UltraLazyClassifier(
    classifiers,
    source_hp_dict=hp_dict,
    cross_validation=True,
    logs_path="./logs",
)

# Generate a synthetic binary classification dataset
x, y = make_classification(
    n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42
)
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Fitting with cross-validation and grid-search algorithm
scores = ulc.fit(x_train, x_test, y_train, y_test)

# Checking the results
best_model = ulc.models[scores.index[0]]

y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy (percentage) :", round(accuracy,5)*100,"%")

```


## Examples

- [Heart Disease Dataset UltraLazy Classifier](https://www.kaggle.com/code/marcosanguineti/heart-disease-dataset-ultralazy-classifier)
