import yaml
import collections
import importlib
from sklearn.pipeline import Pipeline


class Model(
    collections.namedtuple(
        "Model",
        ("model", "pipeline", "preprocessor", "feature_selector", "search"),
        defaults=(None, None, None, None),
    ),
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    # defaults=(None,)
    # @dataclass
    # class Model:
    #     model: dict
    #     pipeline: dict = None
    #     classifier : bool = True
    #     library : str =  "sklearn"
    #     time_series : bool = False

    def gen_from_tup(self, obj_tuple: tuple, *args) -> list:
        """
        Imports and initializes objects from yml file. Returns a list of instantiated objects.
        :param obj_tuple: (full_object_name, params)
        """
        library_name = ".".join(obj_tuple[0].split(".")[:-1])
        class_name = obj_tuple[0].split(".")[-1]
        global tmp_library
        tmp_library = None
        tmp_library = importlib.import_module(library_name)
        global temp_object
        temp_object = None
        global params
        params = obj_tuple[1]
        if len(args) > 0:
            global positional_arg
            positional_arg = args[:]
            exec(
                f"temp_object = tmp_library.{class_name}(positional_arg, **{params})",
                globals(),
            )
            del positional_arg
        elif len(args) == 0:
            exec(f"temp_object = tmp_library.{class_name}(**params)", globals())
        else:
            raise ValueError("Too many positional arguments")
        del params
        del tmp_library
        return temp_object

    def load(self):
        if self.search is not None and self.pipeline is not None:
            pipe_list = []
            search_name = self.search.pop("name")
            grid = {}
            for name in self.pipeline:
                component = getattr(self, name)
                type_ = component.pop("name")
                for key, value in component.items():
                    new_key = name + "__" + key
                    if not isinstance(value, list):
                        value = [value]
                    grid[new_key] = value
                obj_ = self.gen_from_tup((type_, {}))
                pipe_list.append((name, obj_))
                estimator = Pipeline(pipe_list)
                params = {"param_grid": grid}
                params.update(self.search)
                params.update({"estimator": estimator})
            model = self.gen_from_tup((search_name, params))
        elif self.pipeline is not None and self.search is None:
            pipe_list = []
            i = 0
            for name in self.pipeline:
                component = getattr(self, name)
                type_ = component.pop("name")
                obj_ = self.gen_from_tup((type_, component))
                pipe_list.append((name, obj_))
                i += 1
            model = Pipeline(pipe_list)
        else:
            model = self.gen_from_tup((self.model.pop("name"), self.model))
        return model


yaml.add_constructor("!Model", Model)
if __name__ == "__main__":
    document = """\
    pipeline:
    - preprocessor
    - feature_selector
    - model
    model: # Model
        name : sklearn.linear_model.SGDClassifier
        max_iter : [1000]
        penalty : ["l2", "l1","elasticnet"]
        alpha: [.00001, .00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
        l1_ratio : [.10, .20, .30, .50, .7, .9]
    preprocessor: # Centers and Scales
        name : sklearn.preprocessing.StandardScaler
        with_std : True
        with_mean : True
    feature_selector: # Selects Features
        name : sklearn.feature_selection.SelectKBest
        k : [10, 20, 30, 50, 100]
    search:
        name : sklearn.model_selection.GridSearchCV
        refit : True
    """
    document = "!Model\n" + document
    config = yaml.load(document, Loader=yaml.FullLoader)
    model = config.load()
    assert hasattr(model, "fit"), "Model must have a fit method"
    assert hasattr(model, "predict"), "Model must have a predict method"
