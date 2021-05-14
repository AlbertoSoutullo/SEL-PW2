import random
import pandas as pd
from source.forests.forest_classifier import ForestClassifier
from source.trees.cart import Cart


class DecisionForestClassifier(ForestClassifier):
    """
    Decision Forest Classifier class, inherits from Forest Classifier.
    """
    def __init__(self, number_of_trees: int, num_random_features: int, runif: bool = False, seed: int = 0):
        """
        :param number_of_trees: Number of CART tress that will be created.
        :param num_random_features: Number of random features that will be selected per tree.
        :param runif: Flag to use num_random_features as a random number between [1, max_features]
        :param seed: Seed for reproduce results.
        """
        super().__init__(number_of_trees, num_random_features, seed)
        self._runif = runif
        random.seed(seed)

    def fit(self, dataset: pd.DataFrame):
        """
        This method trains the Decision Forest Classifier. Each tree created will be saved in trees list.
        :param dataset: Pandas dataset used to train.
        :return: None
        """
        for i in range(self._NT):

            random_sample = self._get_random_sample(dataset)

            cart_tree = Cart(seed=random.randint(0, 100000))
            cart_tree.fit(random_sample)

            self._update_features(cart_tree.features_chose)
            self._trees.append(cart_tree)

    def _get_F_used(self, dataset: pd.DataFrame) -> int:
        """
        Returns the number of features that will be used, depending if runif was enabled or not.
        :param dataset: Dataframe to see how many features there are.
        :return: The number of random features that will be used.
        """
        if self._runif:
            number_or_random_features = random.randint(1, len(dataset.columns) - 1)
        else:
            number_or_random_features = self._F

        return number_or_random_features

    def _get_random_sample(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Gets a random sample for a tree. In this case, it will return all instances, but with random or given columns.
        :param dataset: The entire dataframe
        :return: A subsample pandas dataframe to use.
        """
        classes = dataset.iloc[:, -1:]
        number_or_random_features = self._get_F_used(dataset)

        random_sample = dataset.iloc[:, :-1].sample(n=number_or_random_features,
                                                    random_state=random.randint(0, 100000), axis=1)
        random_sample_complete = pd.concat([random_sample, classes], axis=1)

        return random_sample_complete
