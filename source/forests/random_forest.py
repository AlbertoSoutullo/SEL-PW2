import random
import pandas as pd
from source.forests.forest_classifier import ForestClassifier
from source.trees.cart import Cart


class RandomForestClassifier(ForestClassifier):
    """
    Random Forest Classifier class, inherits from Forest Classifier.
    """
    def __init__(self, number_of_trees: int, num_random_features: int, random_partition: float = 0.75, seed: int = 0):
        """
        :param number_of_trees: Number of CART tress that will be created.
        :param num_random_features: Number of random features that will be selected per node.
        :param random_partition: Ratio of data that will be bootstrapped for each tree.
        :param seed: Seed for reproduce results.
        """
        super().__init__(number_of_trees, num_random_features, seed)
        self._random_partition = random_partition

    def fit(self, dataset: pd.DataFrame):
        """
        This method trains the Random Forest Classifier. Each tree created will be saved in trees list.
        :param dataset: Pandas dataset used to train.
        :return: None
        """
        for i in range(self._NT):
            random_sample = dataset.sample(frac=self._random_partition, random_state=random.randint(0, 100000))

            cart_tree = Cart(self._F, random.randint(0, 100000))
            cart_tree.fit(random_sample)

            self._update_features(cart_tree.features_chose)
            self._trees.append(cart_tree)
