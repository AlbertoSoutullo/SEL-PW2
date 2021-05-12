import itertools
from collections import Counter
import pandas as pd


def divide_set(dataframe, column, combinations):
    if isinstance(combinations, tuple):
        set1, set2 = _divide_set_for_categorical(dataframe, column, combinations)
    else:
        set1, set2 = _divide_set_for_numerical(dataframe, column, combinations)
    return set1, set2


def _divide_set_for_categorical(dataframe, column, value):
    if isinstance(value[0], tuple): value0 = list(value[0])
    else: value0 = [value[0]]

    if isinstance(value[1], tuple): value1 = list(value[1])
    else: value1 = [value[1]]

    set1 = dataframe.loc[dataframe[column].isin(value0)]
    set2 = dataframe.loc[dataframe[column].isin(value1)]

    return set1, set2


def _divide_set_for_numerical(dataframe, column, value):
    set1 = dataframe.loc[dataframe[column] >= value]
    set2 = dataframe.loc[dataframe[column] < value]

    return set1, set2


def _binary_splits(values):
    for result_indices in itertools.product((0, 1), repeat=len(values)):
        result = ([], [])
        for seq_index, result_index in enumerate(result_indices):
            result[result_index].append(values[seq_index])
        # skip results where one of the sides is empty
        if not result[0] or not result[1]: continue
        # convert from list to tuple so we can hash it later
        yield map(tuple, result)


def _binary_splits_no_dupes(values):
    seen = set()
    for item in _binary_splits(values):
        key = tuple(sorted(item))
        if key in seen: continue
        yield key
        seen.add(key)


def get_categorical_splits(values):
    splits = []
    for left, right in _binary_splits_no_dupes(values):
        splits.append((left, right))

    return splits


def gini_impurity(dataset: pd.DataFrame):
    total_instances = len(dataset)
    count_classes = dataset.iloc[:, -1:].value_counts()
    impurity_sum = 0

    for count in count_classes:
        impurity_sum += (count / total_instances) ** 2

    return 1 - impurity_sum


def _count_classes_in_dataset(dataset: pd.DataFrame):

    return dict(Counter(itertools.chain.from_iterable(dataset.iloc[:, -1:].values.tolist())))


def _select_most_relevant_class(dataset: pd.DataFrame):
    classes = _count_classes_in_dataset(dataset)
    relevant_class = max(classes, key=classes.get)

    return relevant_class
