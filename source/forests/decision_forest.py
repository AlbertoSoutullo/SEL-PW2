import random
import pandas as pd

from source.forests.forest_classifier import ForestClassifier
from source.trees.cart import Cart


class DecisionForestClassifier(ForestClassifier):

    def __init__(self, number_of_trees: int, num_random_features: int, runif: bool = False, seed: int = 0):
        super().__init__(number_of_trees, num_random_features, seed)
        self._runif = runif
        random.seed(seed)

    def fit(self, dataset):

        for i in range(self._NT):
            classes = dataset.iloc[:, -1:]

            if self._runif:
                number_or_random_features = random.randint(1, len(dataset.columns)-1)
            else:
                number_or_random_features = self._F

            random_sample = dataset.iloc[:, :-1].sample(n=number_or_random_features,
                                                        random_state=random.randint(0, 100000), axis=1)
            random_sample_complete = pd.concat([random_sample, classes], axis=1)
            cart_tree = Cart(seed=random.randint(0, 100000))
            cart_tree.fit(random_sample_complete)
            self._update_features(cart_tree.features_chose)
            self._trees.append(cart_tree)
