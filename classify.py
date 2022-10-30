from experiment import load_experiment
import importlib
from pathlib import Path

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import (
    ROCAUC,
    ClassificationReport,
    ClassPredictionError,
    ConfusionMatrix,
    PrecisionRecallCurve,
)
from yellowbrick.features import rank1d, rank2d, PCA
from yellowbrick.model_selection import (
    CVScores,
    DroppingCurve,
    FeatureImportances,
    LearningCurve,
)
from yellowbrick.target import ClassBalance, FeatureCorrelation

from pre_process import ENCODING

# Default scorers
REGRESSOR_SCORERS = {
    "MAPE": mean_absolute_percentage_error,
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "EXVAR": explained_variance_score,
}
CLASSIFIER_SCORERS = {
    "F1": f1_score,
    "ACC": accuracy_score,
    "PREC": precision_score,
    "REC": recall_score,
    "AUC": roc_auc_score,
}


def gen_from_tuple(obj_tuple: list, *args) -> list:
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


if __name__ == "__main__":
    print("Loading config")
    config = dvc.api.params_show()
    print("Loading data, model, and pipeline...")
    data, model = load_experiment()
    if "X_test" in data:
        X_test = data["X_test"]
        X_train = data["X_train"]
        y_test = data["y_test"]
        y_train = data["y_train"]
    else:
        X_train = data["X_train"]
        y_train = data["y_train"]
        test_data = np.load(config['result']['test'])
        X_test = test_data["X"]
        y_test = test_data["y"]
    ####################################
    #             Science              #
    ####################################
    print("Running science...")
    model.fit(X_train, y_train)
    score_dict = {}
    for key, value in CLASSIFIER_SCORERS.items():
        print(f"Calculating {key} score...")
        try:
            score_dict.update({key: value(y_test, model.predict(X_test))})
        except ValueError as e:
            if "average=" in str(e):
                score_dict.update(
                    {key: value(y_test, model.predict(X_test), average="weighted")},
                )

    ####################################
    #             Saving               #
    ####################################
    print(f"Saving results to {config['result']['path']}...")
    result_path = Path(config["result"]["path"], config["result"]["scores"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    df = Series(score_dict)
    df.to_json(result_path)
    if hasattr(model, "cv_results_"):
        cv_df = DataFrame(model.cv_results_)
        print(cv_df.head())
        input("Press enter to continue...")
        print(f"Saving cross validation results to {config['result']['path']}...")
        cv_path = Path(config["result"]["path"], config["result"]["cv"])
        cv_df.to_csv(cv_path)
    else:
        print("No cross validation results to save")
        print(f"Model is a {type(model)}")
    ####################################
    #           Visualising            #
    ####################################
    # Balance

    # X_train = OneHotEncoder().fit_transform(X_train)
    y_train = LabelEncoder().fit_transform(y_train)
    # X_test = OneHotEncoder().fit_transform(X_test)
    y_test = LabelEncoder().fit(y_train).transform(y_test)

    tar_viz = (ClassBalance, FeatureCorrelation)
    # For Visualizing the Feature Selection
    mod_viz = (DroppingCurve, CVScores, LearningCurve, FeatureImportances)
    # For Visualizing the Model
    cls_viz = (
        ConfusionMatrix,
        ROCAUC,
        PrecisionRecallCurve,
        ClassPredictionError,
        ClassificationReport,
        ClassPredictionError,
    )
    plot_path = result_path.parent
    
    # Confusion Matrix
    print("Visualizing Confusion Matrix")
    visualizer = ConfusionMatrix(model, classes=list(ENCODING.keys()))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=str(plot_path / config["plots"]["confusion"]))
    plt.gcf().clear()
    del visualizer
    # Classification Report
    print("Visualizing Classification Report")
    visualizer = ClassificationReport(model, classes=list(ENCODING.keys()))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=str(plot_path / config["plots"]["classification"]))
    plt.gcf().clear()
    
    del visualizer
    
    cv = StratifiedKFold(n_splits=10)
    
    if isinstance(model, Pipeline):
        print("Splitting Pipeline")
        full =  Pipeline(model.steps[:])
        big_X = X_train
        big_y = y_train
        transformers = Pipeline(model.steps[:-1])
        X_train = transformers.transform(X_train)
        X_test = transformers.transform(X_test)
        features = np.array(range(X_train.shape[1]))
        model = model.steps[-1][1]
    # else:
    #     features = np.array(range(X_train.shape[1]))
    
    # PCA Visualization
    print("Visualizing PCA")
    visualizer = PCA(scale=True, classes=list(ENCODING.keys()))
    visualizer.fit_transform(X_train, y_train)
    visualizer.show(plot_path / config["plots"]["pca"])
    plt.gcf().clear()
    del visualizer
    # Fischer
    # visualizer = FeatureCorrelation(labels=False, method='mutual_info-classification', sort=True)
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(plot_path / config['plots']['information']))
    # plt.gcf().clear()
    # del visualizer
    # Feature Selection
    print("Visualizing Feature Selection")
    visualizer = DroppingCurve(model, scoring='f1_weighted', cv = cv)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(plot_path / config['plots']['dropping']))
    plt.gcf().clear()
    del visualizer
    # Learning Curve
    print("Visualizing Learning Curve")
    sizes = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    visualizer = LearningCurve(model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(plot_path / config['plots']['learning']))
    plt.gcf().clear()
    del visualizer
    # Cross Validation
    print("Visualizing Cross Validation")
    visualizer = CVScores(model, cv=cv, scoring='f1_weighted')
    visualizer.fit(X_train, y_train)
    visualizer.show(outpath = str(plot_path / config['plots']['cross_validation']))
    plt.gcf().clear()
    del visualizer
    # Feature Importance
    # print("Visualizing Feature Importance")
    # visualizer = FeatureImportances(model, labels=features)
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(plot_path / config['plots']['feature_importance']))
    # # ROCAUC
    print("Visualizing ROCAUC")
    visualizer = ROCAUC(model, classes=list(ENCODING.items())[:])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath = str(plot_path / config['plots']['roc_auc']))
    plt.gcf().clear()
    del visualizer
    # Balance
    print("Visualizing Class Balance")
    visualizer = ClassBalance(labels=list(ENCODING.keys()))
    visualizer.fit(y_train)
    visualizer.show(outpath=str(plot_path / config["plots"]["balance"]))
    plt.gcf().clear()
    del visualizer
    # # Rank1D, Rank2D <- Make this one last or debug matplotlib. Your choice.
    # fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    # rank1d(X_train, ax=axes[0])
    # rank2d(X_train, ax=axes[1])
    # fig.savefig(str(plot_path / config["plots"]["rank"]))
    # del fig
    # del axes