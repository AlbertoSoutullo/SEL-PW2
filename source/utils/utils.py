import itertools
import json
import math
from collections import Counter
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd


def divide_set(dataframe: pd.DataFrame, column: int, combinations: [Tuple, float]) -> [pd.DataFrame, pd.DataFrame]:
    """
    Divides the dataframe in two sets by a given combination and a given column.
    :param dataframe: Dataframe to split
    :param column: Column index to split
    :param combinations: Values to split
    :return: 2 resulting dataframes
    """
    if isinstance(combinations, tuple):
        set1, set2 = _divide_set_for_categorical(dataframe, column, combinations)
    else:
        set1, set2 = _divide_set_for_numerical(dataframe, column, combinations)
    return set1, set2


def _divide_set_for_categorical(dataframe: pd.DataFrame, column: int, values: Tuple):
    # Performs the division for categorical values
    if isinstance(values[0], tuple):
        value0 = list(values[0])
    else:
        value0 = [values[0]]

    if isinstance(values[1], tuple):
        value1 = list(values[1])
    else:
        value1 = [values[1]]

    set1 = dataframe.loc[dataframe[column].isin(value0)]
    set2 = dataframe.loc[dataframe[column].isin(value1)]

    return set1, set2


def _divide_set_for_numerical(dataframe: pd.DataFrame, column: int, value: float):
    # Performs the division for numerical values
    set1 = dataframe.loc[dataframe[column] >= value]
    set2 = dataframe.loc[dataframe[column] < value]

    return set1, set2


def _binary_splits(values):
    for result_indices in itertools.product((0, 1), repeat=len(values)):
        result = ([], [])
        for seq_index, result_index in enumerate(result_indices):
            result[result_index].append(values[seq_index])
        # skip results where one of the sides is empty
        if not result[0] or not result[1]:
            continue
        # convert from list to tuple so we can hash it later
        yield map(tuple, result)


def _binary_splits_no_dupes(values):
    """
    It performs the binary combinations. This function was taken in
    https://stackoverflow.com/questions/40158243/find-all-binary-splits-of-a-nominal-attribute
    """
    seen = set()
    for item in _binary_splits(values):
        key = tuple(sorted(item))
        if key in seen:
            continue
        yield key
        seen.add(key)


def get_categorical_splits(values: Tuple):
    splits = []
    for left, right in _binary_splits_no_dupes(values):
        splits.append((left, right))

    return splits


def gini_impurity(dataset: pd.DataFrame) -> float:
    """
    Calculates the gini impurity of a given dataset.
    :param dataset: Dataset to use
    :return: gini impurity
    """
    total_instances = len(dataset)
    count_classes = dataset.iloc[:, -1:].value_counts()
    impurity_sum = 0

    for count in count_classes:
        impurity_sum += (count / total_instances) ** 2

    return 1 - impurity_sum


def count_classes_in_dataset(dataset: pd.DataFrame) -> Dict:
    """
    It count the classes in a dataset. It assumes that the classes are in the last column.
    :param dataset: Dataset
    :return: Dict with the count of the classes
    """
    return dict(Counter(itertools.chain.from_iterable(dataset.iloc[:, -1:].values.tolist())))


def _select_most_relevant_class(dataset: pd.DataFrame) -> [str, int]:
    """
    It selects the most relevant class of a dataframe
    :param dataset: Dataframe
    :return: Class most relevant as string or int
    """
    classes = count_classes_in_dataset(dataset)
    relevant_class = max(classes, key=classes.get)

    return relevant_class


def _calculate_fs_random_forest(dataset: pd.DataFrame) -> List:
    """
    It calculates the F values for a random forest.
    :param dataset: Dataset to check the features.
    :return: List with the F values
    """
    number_of_attributes = len(dataset.columns[:-1])
    f_3 = int(np.log2(number_of_attributes + 1))
    f_4 = int(math.sqrt(number_of_attributes))

    return [1, 3, f_3, f_4]


def _calculate_fs_decision_forest(dataset: pd.DataFrame) -> List:
    """
    Calculates the F values for Decision forest
    :param dataset: Dataset to check the features
    :return: List with the F values
    """
    number_of_attributes = len(dataset.columns[:-1])

    f1 = int(number_of_attributes / 4)
    f2 = int(number_of_attributes / 2)
    f3 = int(3 * number_of_attributes / 4)
    f4 = "runif"

    return [f1, f2, f3, f4]


def check_f_field(config, dataset) -> List:
    """
    It checks the F field in the json file. If it is given, use the given values,
    if not, it calculates the F taking into account the classifier.
    :param config: Json file
    :param dataset: Dataset
    :return: List of F values
    """
    classifier = config["classifier"]
    if "F" in config.keys():
        fs = config["F"]
    else:
        fs = _calculate_fs(classifier, dataset)

    if fs is None:
        print(f"Classifier {classifier} not supported.")
        exit()

    return fs


def _calculate_fs(classifier: str, dataset: pd.DataFrame) -> [List, None]:
    """
    Calculates F depending on the classifier.
    :param classifier: Random Forest or Decision Forest
    :param dataset: Dataset to take into account the features.
    :return: List of F values or None if a not valid classifier was given
    """

    if classifier == "RandomForestClassifier":
        fs = _calculate_fs_random_forest(dataset)
    elif classifier == "DecisionForestClassifier":
        fs = _calculate_fs_decision_forest(dataset)
    else:
        return None

    return fs


def load_json(interpreter_configuration_path: str) -> Dict:
    """
    Loads a json configuration.
    :param interpreter_configuration_path: path to the file
    :return: A json dict
    """
    with open(interpreter_configuration_path) as json_file:
        config = json.load(json_file)

    return config


def train_test_truth_split(dataset: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, test and ground truth of test by a configuration file
    :param dataset: Dataset to split
    :param config: JSON Configuration
    :return:
    """
    train = dataset.sample(frac=config["train_test_split"], random_state=config["seed"])
    test = dataset.drop(train.index)

    ground_truth = test.iloc[:, -1:].iloc[:, 0].values.tolist()

    test = test.iloc[:, :-1]

    return train, test, ground_truth
