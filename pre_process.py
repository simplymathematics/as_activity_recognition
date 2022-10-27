import zipfile
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import tsfresh

ENCODING = {"Running": 0, "Standing_still": 1, "Walking": 2}


def create_directories(dirs: list):
    directories = []
    for arg in dirs:
        directories.append(Path(arg))
        arg.mkdir(parents=True, exist_ok=True)
    return directories


def extract_feat_data_from_raw(path_raw: Path, path_out: Path, key = 'acc'):
    for path_f in path_raw.iterdir():
        path_feat = path_out / path_f.name
        with path_f.open() as fp:
            acc_data = ""
            for line in fp.readlines():
                if line.split("\t")[1] == key.upper():
                    acc_data += line
            if acc_data:
                with path_feat.open("w") as fp:
                    fp.write(acc_data)


def extract_time_windows(
    file_nrs: list,
    path_feat: Path,
    path_time_windows: Path,
    encoding: dict,
    key="acc",
    time_window_size=100,
    stride=10,
    filename="data.npz",
):
    all_data = []
    for path_f in path_feat.iterdir():
        i = int(path_f.stem.split("_")[-1])
        if i in file_nrs:
            all_data.append(
                extract_time_windows_from_feat_file(path_f, encoding, key=key, filename = filename),
            )
    all_data = np.vstack(all_data)
    nr_data_points = all_data.shape[0]
    indx = np.arange(nr_data_points)
    np.random.shuffle(indx)
    dict_ = {"X": all_data[indx, 1:], "y": all_data[indx, 0]}
    np.savez(str(Path(path_time_windows, filename)), **dict_)
    return Path(path_time_windows, filename).resolve()


def extract_time_windows_from_feat_file(
    path_f_acc: Path,
    encoding: dict,
    key:str = 'acc',
    time_window_size=100,
    stride=10,
    filename="data.npz",
):
    path_f = Path(path_f_acc)
    df = pd.read_csv(
        path_f,
        sep="\t",
        names=["time", "type", key + "_x", key + "_y", key + "_z"],
    )
    activity = path_f.stem.rsplit("_", maxsplit=1)[0]
    activity_encoded = encoding[activity]
    df[key] = np.linalg.norm(df[[key + "_x", key + "_y", key + "_z"]], axis=1)
    i = 0
    data = df[key]
    N = data.size
    all_data = []
    while (i + time_window_size) < N:
        all_data.append(data[i : i + time_window_size])  # noqa E203
        i += stride
    all_data = np.vstack(all_data)
    nr_time_windows, _ = all_data.shape
    activity_encoded_vec = np.ones((nr_time_windows, 1)) * activity_encoded
    all_data = np.hstack([activity_encoded_vec, all_data])
    return all_data


def unpack_time_windows_data(X_raw: np.ndarray):
    nr_data_points, time_window_size = X_raw.shape
    X = []
    for i, x in enumerate(X_raw):
        X.append(
            np.hstack(
                [
                    np.ones((time_window_size, 1)) * i,
                    np.arange(0, time_window_size).reshape(-1, 1),
                    x.reshape(-1, 1),
                ],
            ),
        )
    return np.vstack(X)


def create_features_from_data(inpath: Path, outpath: Path, data) -> Path:
    """
    Create features from data.
    :param inpath: path to data
    :param outpath: path to save features
    :return: path to features
    """
    assert inpath.is_file(), "Input path must be a file"
    data = np.load(str(inpath))
    X_raw = data["X"]
    y = data["y"]
    X = unpack_time_windows_data(X_raw)
    y = pd.Series(y)
    timeseries = pd.DataFrame(X, columns=("id", "time", "x"))
    timeseries = timeseries[["id", "time", "x"]]
    features_filtered_direct = tsfresh.extract_relevant_features(
        timeseries,
        y,
        column_id="id",
        column_sort="time",
    )
    np.savez(X=features_filtered_direct, y=y, file=str(outpath))
    return outpath.resolve()


if __name__ == "__main__":
    config = dvc.api.params_show()
    data_file = config["preprocessing"]["data_file"]
    key = config["preprocessing"]["key"]
    window_size = config["preprocessing"]["window"]
    stride = config["preprocessing"]["stride"]
    data_path = Path(config["preprocessing"]["root"])
    raw_path = data_path / config["preprocessing"]["raw"]
    try:
        sensor_path = data_path / config["preprocessing"][key]
    except KeyError as e:
        print(e)
        raise Exception("You must specify a location for {key} in the 'preprocessing' section of the params.yaml file.")
    windows = data_path / config["preprocessing"]["time"]
    features = data_path / config["preprocessing"]["feat"]
    input_data = config["preprocessing"]["zip"]
    print("Creating directories")
    paths = [data_path, raw_path, sensor_path, windows, features]
    dirs = create_directories(paths)
    with zipfile.ZipFile(input_data, "r") as zip_ref:
        zip_ref.extractall(raw_path)
    print(f"Extracting {key} data")
    extract_feat_data_from_raw(raw_path, sensor_path, key = key)
    print("Extracting time windows")
    train_file = config["preprocessing"]["train_key"] + str("_" + data_file)
    test_file = config["preprocessing"]["test_key"]+ str("_" + data_file)
    print("Creating training data")
    train_data = extract_time_windows(
        file_nrs=(1, 2, 3, 4),
        path_feat=sensor_path,
        path_time_windows= windows,
        encoding=ENCODING,
        time_window_size=window_size,
        stride=stride,
        key=key,
        filename = train_file,
    )
    print("Creating test data")
    test_data = extract_time_windows(
        file_nrs=(5,),
        path_feat=sensor_path,
        path_time_windows= windows,
        encoding=ENCODING,
        time_window_size=window_size,
        stride=stride,
        key=key,
        filename = test_file,
    )
    
    print("Creating training features")
    create_features_from_data(inpath=Path(windows, train_file), outpath= Path(features , train_file), data = train_data)
    print("Creating test features")
    create_features_from_data(inpath=Path(windows, test_file), outpath= Path(features , test_file), data = test_data)
