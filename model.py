import yaml
import collections
import importlib
from sklearn.pipeline import Pipeline


class Model(
    collections.namedtuple(
        "Model",
        ("model", "pipeline", "preprocessor", "feature_selector", "search", "grid"),
        defaults=(None, None, None, None, None),
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
            positional_arg = args[0]
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
        # Initialize model
        tup = (self.model.pop("name"), self.model)
        model = self.gen_from_tup(tup)
        pipe_list = []
        i = 0
        # Initialize pipeline
        if self.pipeline is not None:
            # if "cache" in self.pipeline:
            #     cache = self.pipeline.pop("cache")
            for name in self.pipeline:
                print(f"name is: {name}")
                component = getattr(self, name)
                print(component)
                type_ = component.pop("name")
                obj_ = self.gen_from_tup((type_, component))
                pipe_list.append((name, obj_))
                i += 0
            pipe_list.append(("model", model))
            model = Pipeline(pipe_list)
        if self.search is not None:
            print(self.search)
            print(type(self.search))
            dict_ = dict(self.search)
            name = str(dict_.pop('name'))
            if "grid" in name.lower():
                assert self.grid is not None, "if grid search is specified, param_grid must be specified."
                self.search.update({"param_grid": self.grid})
                self.search.update({"estimator": model})
                _ = self.search.pop("name")
                model = self.gen_from_tup((name, self.search))
            else:
                model = self.gen_from_tup((name, self.search))
        return model


yaml.add_constructor("!Model", Model)
if __name__ == "__main__":
    document = """\
    pipeline:
    - preprocessor
    - feature_selector
    model:
        name : sklearn.ensemble.RandomForestClassifier
        n_estimators : 10
        max_depth : 1
    preprocessor:
        name : sklearn.preprocessing.StandardScaler
        with_std : True
        with_mean : True
    feature_selector:
        name : sklearn.feature_selection.SelectKBest
        k: 30
    search:
        name : sklearn.model_selection.GridSearchCV
        refit : True
    grid:
        n_estimators : [10, 20, 30]
        k : [10, 20, 30]
    """
    document = "!Model\n" + document
    config = yaml.unsafe_load(document)
    model = config.load()
    assert hasattr(model, "fit"), "Model must have a fit method"
    assert hasattr(model, "predict"), "Model must have a predict method"
