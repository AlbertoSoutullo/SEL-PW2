from typing import List
import pandas as pd

from utils import _perform_continuous_partitions, _perform_discrete_partitions


class Cart:

    def __init__(self):
        self._nodes = []
        self._leafs = []

    def fit(self, dataset: pd.DataFrame):
        # Repeat until no more splits can be done
        for column in dataset[:-1]:
            # Select binary attribute split
            if pd.api.types.is_string_dtype(dataset[column]):
                column_partition = _perform_discrete_partitions(list(dataset[column]))
                print(column_partition)
            else:
                column_partition = _perform_continuous_partitions(list(dataset[column]))
                print(column_partition)


test = pd.DataFrame({"height": ["alto", "medio", "bajo", "bajo"],
                     "weight": [70, 80, 90, 60],
                     "age": [20, 30, 40, 10],
                     "class": ["A", "A", "B", "A"]})
print(test)
cart = Cart()
cart.fit(test)


