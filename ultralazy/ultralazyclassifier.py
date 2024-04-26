import json
import os
import time
import traceback
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from ultralazy.utils.decorators import suppress_warnings


class UltraLazyClassifier:
    """
    This module helps in fitting to all the classification algorithms
    that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).
    """

    def __init__(
        self,
        classifiers: List[tuple],
        verbose=0,
        cv_verbose: int = 0,
        custom_metric=None,
        random_state=42,
        cross_validation: bool = True,
        cv: int = 5,
        source_hp_path: Optional[str] = None,
        logs_path: Optional[str] = None,
        source_hp_dict: Optional[dict] = None,
        n_jobs: int = -1,
    ):

        # Assertions
        if source_hp_path is not None and source_hp_dict is not None:
            raise ValueError(
                "Both source_hp_path and source_hp_dict are provided."
                "You must provide only"
                "one of them."
            )

        if cross_validation and (source_hp_path is None and source_hp_dict is None):
            raise ValueError(
                "When cross_validation is set to True, source_hp_path or source_hp_dict must be provided."
            )

        if logs_path is None:
            print("Warning: logs path is not provided. No logs will be written to file")

        else:
            if not os.path.exists(logs_path):
                raise FileNotFoundError("Logs path does not exist")

        self.verbose = verbose
        self.custom_metric = custom_metric
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

        self.cross_validation = cross_validation
        self.cv = cv
        self.cv_verbose = cv_verbose

        self.logs_path = logs_path

        self.n_jobs = n_jobs

        if source_hp_path is not None:
            with open(source_hp_path, "r", encoding="utf-8") as f:
                self.hyperparameters = json.load(f)

        if source_hp_dict is not None:
            self.hyperparameters = source_hp_dict

    def get_run_path(self) -> str | None:
        """Get actual run logging path

        Returns:
            str | None: run logging path, if self.logs_path is not None
        """
        if self.logs_path is None:
            return

        # Get today's date in the format 'year_month_day'
        today_str = datetime.now().strftime("%Y_%m_%d")

        # Prepare to find the next available run number
        next_run_number = 0
        pattern = f"{today_str}_run_"

        # Check the existing directories to find the next available run number
        for item in os.listdir(self.logs_path):
            if item.startswith(pattern) and os.path.isdir(
                os.path.join(self.logs_path, item)
            ):
                try:
                    run_number = int(item.split("_")[-1])
                    if run_number >= next_run_number:
                        next_run_number = run_number + 1
                except ValueError:
                    continue  # In case the folder name is not formatted correctly

        # Create the new folder name with the next run number
        new_folder_name = f"{today_str}_run_{next_run_number}"
        new_folder_path = os.path.join(self.logs_path, new_folder_name)

        # Create the new folder
        os.makedirs(new_folder_path, exist_ok=True)

        return new_folder_path

    @suppress_warnings
    def fit(
        self,
        x_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """Fit Classification algorithms to X_train and y_train, predict and score on x_test,y_test.

        Parameters
        ----------
        x_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        x_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        """

        accuracy_list = []
        b_accuracy_list = []
        roc_auc_list = []
        f1_list = []
        names = []
        times = []

        if self.cross_validation is not None:
            cv_hp = []

        if self.custom_metric is not None:
            custom_metric_list = []

        if isinstance(x_train, np.ndarray):
            x_train = pd.DataFrame(x_train)
            x_test = pd.DataFrame(x_test)

        run_path = self.get_run_path()

        if run_path is not None:
            # save x_train, x_test, y_train, y_test inside run_path
            x_train.to_csv(os.path.join(run_path, "x_train.csv"))
            x_test.to_csv(os.path.join(run_path, "x_test.csv"))
            pd.DataFrame(y_train, columns=["label"]).to_csv(
                os.path.join(run_path, "y_train.csv")
            )
            pd.DataFrame(y_test, columns=["label"]).to_csv(
                os.path.join(run_path, "y_test.csv")
            )

        scores = pd.DataFrame()

        for name, model in (pbar := tqdm(self.classifiers)):

            pbar.set_description(f"Now processing: {name}")

            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            (
                                "classifier",
                                model(
                                    random_state=self.random_state, n_jobs=self.n_jobs
                                ),
                            ),
                        ]
                    )
                else:
                    pipe = Pipeline(steps=[("classifier", model(n_jobs=self.n_jobs))])

                if not self.cross_validation:
                    pipe.fit(x_train, y_train)
                else:
                    param_grid: dict = self.hyperparameters[name]
                    f1_scorer = make_scorer(f1_score, average="macro")

                    # Create a StratifiedKFold cross-validator
                    stratified_kfold = StratifiedKFold(
                        n_splits=self.cv,
                        shuffle=True,
                        random_state=(
                            self.random_state if self.random_state is not None else None
                        ),
                    )

                    # Setup the grid search with StratifiedKFold cross-validation
                    grid_search = GridSearchCV(
                        pipe["classifier"],
                        param_grid,
                        cv=stratified_kfold,
                        verbose=self.cv_verbose,
                        scoring=f1_scorer,
                        n_jobs=self.n_jobs,
                    )

                    # Perform grid search
                    grid_search.fit(x_train, y_train)

                    pipe = Pipeline(
                        steps=[
                            (
                                "classifier",
                                model(
                                    **grid_search.best_params_,
                                    random_state=(
                                        self.random_state
                                        if self.random_state is not None
                                        else None
                                    ),
                                    n_jobs=self.n_jobs,
                                ),
                            ),
                        ]
                    )

                    pipe.fit(x_train, y_train)

                    cv_hp.append(grid_search.best_params_)

                # Saving the trained model
                self.models[name] = pipe

                # Evaluating Metrics
                y_pred = pipe.predict(x_test)

                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    print("ROC AUC couldn't be calculated for " + name)
                    print(exception)

                names.append(name)
                accuracy_list.append(accuracy)
                b_accuracy_list.append(b_accuracy)
                roc_auc_list.append(roc_auc)
                f1_list.append(f1)
                times.append(time.time() - start)

                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    custom_metric_list.append(custom_metric)

                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Time taken": time.time() - start,
                            }
                        )

                if self.custom_metric is None:
                    scores = pd.DataFrame(
                        {
                            "Model": names,
                            "Accuracy": accuracy_list,
                            "Balanced Accuracy": b_accuracy_list,
                            "ROC AUC": roc_auc,
                            "F1 Score": f1_list,
                            "Time Taken": times,
                        }
                    )
                else:
                    scores = pd.DataFrame(
                        {
                            "Model": names,
                            "Accuracy": accuracy_list,
                            "Balanced Accuracy": b_accuracy_list,
                            "ROC AUC": roc_auc,
                            "F1 Score": f1_list,
                            self.custom_metric.__name__: custom_metric,
                            "Time Taken": times,
                        }
                    )

                scores = scores.sort_values(
                    by="Balanced Accuracy", ascending=False
                ).set_index("Model")

                # Logging to file
                if run_path is not None:
                    # Dump scores as xlsx
                    scores.to_excel(os.path.join(run_path, "scores.xlsx"))

                    if self.cross_validation is not None:
                        # Dump cv_hp as json
                        with open(
                            os.path.join(run_path, f"{name}_cv_hp.json"),
                            "w",
                            encoding="utf-8",
                        ) as file:
                            json.dump(cv_hp, file, indent=4)

            except Exception:
                print(name + " model failed to execute")
                traceback.print_exc()

        return scores
