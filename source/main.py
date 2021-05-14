import time
import pandas as pd
from sys import argv
from source.forests.forest_interpreter import ForestInterpreter
from source.utils.utils import load_json, check_f_field, train_test_truth_split


if __name__ == '__main__':

    interpreter_configuration_path = argv[1]
    config = load_json(interpreter_configuration_path)

    dataset = pd.read_csv(config["dataset"], header=None)
    classifier = config["classifier"]
    fs = check_f_field(config, dataset)

    train, test, ground_truth = train_test_truth_split(dataset, config)

    forest_interpreter = ForestInterpreter(classifier,
                                           train,
                                           config["NT"],
                                           fs,
                                           config["csv_out_name"],
                                           config["seed"])

    start_time = time.time()

    forest_interpreter.interpret(test, ground_truth)

    print(f"Execution time: {str(time.time() - start_time)}")
