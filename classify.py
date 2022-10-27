from experiment import load_experiment
import importlib
from pathlib import Path

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
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

    config = dvc.api.params_show()
    data, model = load_experiment()
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_test = data["X_test"]
    X_train = data["X_train"]

    ####################################
    #             Science              #
    ####################################
    model.fit(X_train, y_train)
    score_dict = {}
    for key, value in CLASSIFIER_SCORERS.items():
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
    result_path = Path(config["result"]["path"], config["result"]["scores"])
    df = DataFrame()
    df["score"] = score_dict.values()
    df["scorer"] = score_dict.keys()
    df.to_json(result_path, orient="index")
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
    features = np.array(range(X_train.shape[1]))
    parent = Path(result_path).parent
    # Balance
    visualizer = ClassBalance(labels=list(ENCODING.keys()))
    visualizer.fit(y_train)
    visualizer.show(outpath=str(parent / config["plots"]["balance"]))
    # Confusion Matrix
    visualizer = ConfusionMatrix(model, classes=list(ENCODING.keys()))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=str(parent / config["plots"]["confusion"]))
    # Classification Report
    visualizer = ClassificationReport(model, classes=list(ENCODING.keys()))
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show(outpath=str(parent / config["plots"]["classification"]))
    # PCA Visualization
    visualizer = PCA(scale=True, classes=list(ENCODING.keys()))
    visualizer.fit_transform(X_train, y_train)
    visualizer.show(parent / config["plots"]["classification"])
    # Rank1D, Rank2D <- Make this one last or debug matplotlib. Your choice.
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    rank1d(X_train, ax=axes[0])
    rank2d(X_train, ax=axes[1])
    fig.savefig(str(parent / config["plots"]["rank"]))
    # Rank2D

    # # Fischer
    # visualizer = FeatureCorrelation(labels=features, method='mutual_info-classification', sort=True)
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(parent / config['plots']['information']))
    # # Feature Selection
    # visualizer = DroppingCurve(clf, scoring='f1_weighted')
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(parent / config['plots']['dropping']))
    # # Learning Curve
    # cv = StratifiedKFold(n_splits=12)
    # sizes = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    # visualizer = LearningCurve(clf, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(parent / config['plots']['learning']))
    # # Cross Validation
    # visualizer = CVScores(clf, cv=cv, scoring='f1_weighted')
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(parent / config['plots']['cross_validation']))
    # # Feature Importance
    # visualizer = FeatureImportances(clf)
    # visualizer.fit(X_train, y_train)
    # visualizer.show(outpath = str(parent / config['plots']['feature_importance']))
    # # ROCAUC
    # visualizer = ROCAUC(clf, classes=list(ENCODING.items())[:])
    # visualizer.fit(X_train, y_train)
    # visualizer.score(X_test, y_test)
    # visualizer.show(outpath = str(parent / config['plots']['rocauc']))
