#!/usr/bin/env python3

import argparse
import io
import logging
import os
import shutil
import zipfile
from typing import Tuple

import pandas as pd
import requests

# suppress SettingWithCopyWarning: warning
pd.options.mode.chained_assignment = None


CREDIT_HISTORY_SOURCE_URL = "data/credit_history_all.csv"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )


def get_data_credit_history_decision_tree() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get credit history dataset with decision tree model prediction"""
    return get_data_credit_history(True)


def get_data_credit_history_gradient_boosting() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get credit history with gradient boosting model"""
    return get_data_credit_history(False)


def get_data_credit_history(use_model: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:

    raw_data = pd.read_csv(CREDIT_HISTORY_SOURCE_URL)



    target = "loan_status"
    numerical_features = [
        "person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
        "tax_returns_filed", "population", "total_wages", "credit_card_due", "mortgage_due",
        "student_loan_due", "vehicle_loan_due", "hard_pulls", "missed_payments_2y",
        "missed_payments_1y", "missed_payments_6m", "bankruptcies"
    ]
    categorical_features = [
        "person_home_ownership", "loan_intent", "city", "state", "location_type"
    ]

    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder()
    encoder.fit(raw_data[categorical_features])

    transform_training_df = raw_data.copy()
    transform_training_df[categorical_features] = encoder.transform(
        raw_data[categorical_features]
    )

    train_X = transform_training_df[
        transform_training_df.columns.drop(target)
    ]
    train_X = train_X.reindex(sorted(train_X.columns), axis=1)
    train_Y = transform_training_df.loc[:, target]

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.10)

    features = numerical_features + categorical_features

    if use_model:
        from sklearn.tree import DecisionTreeClassifier

        max_depth = 22
        model = DecisionTreeClassifier(max_depth=max_depth)

    else:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=0)


    model.fit(x_train[features], y_train[target])

    reference_bike_data = x_train.copy()
    reference_bike_data[target] = y_train[target]
    reference_bike_data["prediction"] = model.predict(x_train[features])

    production_bike_data = x_test.copy()
    production_bike_data[target] = y_test[target]
    production_bike_data["prediction"] = model.predict(x_test[features])

    return reference_bike_data, production_bike_data


def main(dataset_name: str, dataset_path: str) -> None:
    logging.info("Generate test data for dataset %s", dataset_name)
    dataset_path = os.path.abspath(dataset_path)

    if os.path.exists(dataset_path):
        logging.info("Path %s already exists, remove it", dataset_path)
        shutil.rmtree(dataset_path)

    os.makedirs(dataset_path)

    reference_data, production_data = DATA_SOURCES[dataset_name]()
    logging.info("Save datasets to %s", dataset_path)
    reference_data.to_csv(os.path.join(dataset_path, "reference.csv"), index=False)
    production_data.to_csv(os.path.join(dataset_path, "production.csv"), index=False)

    logging.info("Reference dataset was created with %s rows", reference_data.shape[0])
    logging.info("Production dataset was created with %s rows", production_data.shape[0])


# DATA_SOURCES = {
#     "bike_random_forest": get_data_bike_random_forest,
#     "bike_gradient_boosting": get_data_bike_gradient_boosting,
#     "kdd_k_neighbors_classifier": get_data_kdd_classification,
# }

DATA_SOURCES = {
    "credit_history_decision_tree": get_data_credit_history_decision_tree,
    "credit_history_gradient_boosting": get_data_credit_history_gradient_boosting,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for data and config generation for demo Evidently metrics integration with Grafana"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATA_SOURCES.keys(),
        type=str,
        help="Dataset name for reference.csv= and production.csv files generation.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path for saving dataset files.",
    )

    args = parser.parse_args()
    setup_logger()
    if args.dataset not in DATA_SOURCES:
        exit(f"Incorrect dataset name {args.dataset}, try to see correct names with --help")
    main(args.dataset, args.path)
