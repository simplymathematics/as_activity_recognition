from data import Data
from model import Model
import yaml
from numpy import load

def load_experiment(filename="params.yaml", input_data_key="sample", model_key="model", test_data = None):
    with open(filename, "r") as f:
        full = yaml.load(f, Loader=yaml.Loader)
    document = str(full[input_data_key])
    document = str(document)
    input_data = yaml.load("!Data\n" + document, Loader=yaml.Loader)
    assert isinstance(input_data, Data)
    input_data = input_data()
    if test_data is not None or "X_test" not in input_data:
        assert test_data is not None, "test_data must be specified if X_test is not in input_data"
        test_data = load(test_data)
    with open(filename, "r") as f:
        full = yaml.load(f, Loader=yaml.Loader)
    pipe = full["pipeline"]
    document = {}
    for entry in pipe:
        if entry is not model_key:
            document[entry] = full[entry]
    document = str(document)
    config = yaml.load("!Model\n" + document, Loader=yaml.Loader)
    assert isinstance(config, Model)
    loaded_model = config.load()
    return input_data, loaded_model


if __name__ == "__main__":
    data, model = load_experiment()
    assert "X_train" in data, "X_train not found"
    assert "X_test" in data, "X_test not found"
    assert "y_train" in data, "y_train not found"
    assert "y_test" in data, "y_test not found"
    assert hasattr(model, "fit"), "Model must have a fit method"
    assert hasattr(model, "predict"), "Model must have a predict method"
