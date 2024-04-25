"""Hyperparameters Reader from Scikit-learn API
"""

import argparse
import inspect
import json
import os
from typing import Literal, Optional, Sequence

import lightgbm
import tqdm
import xgboost
from openai import OpenAI
from sklearn.utils import all_estimators

SYSTEM_PROMPT = """
Your task is to return a python dictionary as a string, for hyperparameter optimisation, given the class provided.
You only need to return the string, nothing else.
Avoid any otherr content in your response.
It will be read by json.loads.
You do not have to vary all hyper-parameters whose importance is very marginal or nil during a model selection phase.
Avoid optimizing parameters like "random state".
Avoid any unnecessary special character.
Your response will be used for a model GridSearch Cross validation, so format your response accordingly.
Avoid the "json" formatting key in your answer.
"""


class HyperparametersReader:
    """Hyperparameters Reader from Scikit-learn API"""

    __openai_gpt_version = "gpt-4-turbo"

    def __init__(self, source_path: Optional[str] = None):
        if source_path is None or not os.path.exists(source_path):
            self.hyperparameters = {}

        else:
            if os.path.exists(source_path):
                with open(source_path, "r", encoding="utf-8") as f:
                    self.hyperparameters = json.load(f)
            else:
                print(f"{source_path} does not exist")
                print("HP not loaded...")
                self.hyperparameters = {}

        self.client = OpenAI()

    @staticmethod
    def get_estimators(type_filter: Optional[str] = None):
        """Get all estimators from Scikit-learn API + some bonuses"""
        all_est = all_estimators(type_filter=type_filter)
        all_est.append(("XGBClassifier", xgboost.XGBClassifier))
        all_est.append(("LGBMClassifier", lightgbm.LGBMClassifier))
        return all_est

    def __call__(
        self,
        destination_path: str = "ultralazy_hp.json",
        type_filter: (
            Sequence[Literal["classifier", "regressor", "cluster", "transformer"]]
            | Literal["classifier", "regressor", "cluster", "transformer"]
            | None
        ) = None,
    ):

        all_est = self.get_estimators(type_filter=type_filter)

        for i in tqdm.tqdm(range(len(all_est))):
            if all_est[i][0] not in self.hyperparameters:
                try:
                    target_source_code = inspect.getsource(all_est[i][1])

                    completion = self.client.chat.completions.create(
                        model=self.__openai_gpt_version,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": target_source_code},
                        ],
                    )

                    message = completion.choices[0].message.content

                    message = message.replace("True", "true")
                    message = message.replace("False", "false")
                    message = message.replace("None", "null")

                    self.hyperparameters[all_est[i][0]] = json.loads(message)

                except Exception as e:
                    print(f"HP extract failed for {all_est[i][0]}: {e}")
                    continue

            else:
                print(f"{all_est[i][0]} already in hp. Skipping...")

        with open(destination_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.hyperparameters, indent=4))


if __name__ == "__main__":
    # Defining an arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination_path", type=str, default="ultralazy_hp.json")
    # Adding a filtering option between the following [None, 'classifier',
    # 'regressor', 'cluster', 'transformer']

    parser.add_argument("--estimator_type", type=str, default=None)

    args = parser.parse_args()

    # Running the script
    HyperparametersReader(source_path=args.destination_path)(
        destination_path=args.destination_path, type_filter=args.estimator_type
    )
