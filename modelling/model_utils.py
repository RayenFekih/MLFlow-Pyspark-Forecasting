import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pandas import DataFrame as PandasDataframe
from pyspark.sql import DataFrame as SparkDataFrame
from scipy.sparse import lil_matrix
from sklearn.metrics import mean_squared_error

from feature_engineering.engineering import featureConstructor
from modelling.models import Model, XGBoostModel, LgbmModel
from utils.utils import log

import mlflow


def loadModel(model_name: str, model_params: dict, run_id: str = None) -> Model:
    """
    This function loads the specified model in the config file
    If run_id is specified, a pretrained model (saved through the mlflow run) would be loaded
    """
    match model_name:
        case "xgboost":
            if run_id:
                return mlflow.pyfunc.load_model(
                    f"runs:/{run_id}/model"
                )  # To be changed in case of adding other mlflow flavors
            else:
                return XGBoostModel(model_params)
        case "lgbm":
            if run_id:
                return mlflow.pyfunc.load_model(
                    f"runs:/{run_id}/model"
                )
            else:
                return LgbmModel(model_params)
        case _:
            raise ValueError(f"{model_name} model in undefined")


def splitData(
    data: SparkDataFrame, model_config: dict, features_config: dict
) -> tuple[SparkDataFrame, SparkDataFrame]:

    """
    This Function splits the spark dataframe into train and inference data then selects the needed columns for modelling
    """

    # parse parameters from the config file
    train_startDate: str = model_config["train_startDate"]
    train_endDate: str = model_config["train_endDate"]
    inference_startDate: str = model_config["inference_startDate"]
    inference_endDate: str = model_config["inference_endDate"]
    hierarchy_columns: list = model_config["hierarchy_columns"]
    target: str = model_config["target"]

    features = featureConstructor(features_config)
    categorical_features = [
        f.output_column for f in features if f.output_type == "categorical"
    ]
    numerical_features = [
        f.output_column for f in features if f.output_type == "numeric"
    ]

    train_data = data.where(
        F.col("date").between(train_startDate, train_endDate)
    ).select(*hierarchy_columns + categorical_features + numerical_features, target)

    inference_data = data.where(
        F.col("date").between(inference_startDate, inference_endDate)
    ).select(*hierarchy_columns + categorical_features + numerical_features, target)

    return train_data, inference_data


def _OneHotEncode(
    data: PandasDataframe, categorical_columns: list, numerical_columns: list
):
    """
    Perform one hot encoding for categorical variables
    Null values won't be encoded
    """
    df_list = []
    for col_name in categorical_columns:
        df_oh = pd.get_dummies(
            data[col_name].astype(str), prefix=col_name, drop_first=False
        )
        df_oh.drop(
            columns=[col_name + "_None"], inplace=True, errors="ignore"
        )  # <null>'s are dropped
        df_list.append(df_oh)

    df_pandas_num = data[numerical_columns]
    df_list.append(df_pandas_num)
    ohe_data = pd.concat(df_list, axis=1)

    return ohe_data


def _data_frame_to_scipy_sparse_matrix(df, numerical_columns):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """

    arr = lil_matrix(df.shape, dtype=np.float32)

    for i, col in enumerate(df.columns):
        ix = df[col] != 0

        if col in numerical_columns:
            arr[np.where(ix), i] = df[col].to_numpy()[ix]
        else:
            arr[np.where(ix), i] = 1

    return arr.tocsr()


def prepare_data(
    pd_data: PandasDataframe,
    model_config: str,
    features_config: str,
    prefix: str,
):
    """
    This function takes a Pandas dataframe, one hot-encode it and transforms it into a CSR matrix.
    It returns the CSR matrix and the target column
    """
    hierarchy_columns: list = model_config["hierarchy_columns"]
    target: str = model_config["target"]

    features = featureConstructor(features_config)
    categorical_features = [
        f.output_column for f in features if f.output_type == "categorical"
    ]
    numerical_features = [
        f.output_column for f in features if f.output_type == "numeric"
    ]

    log(f"One-hot encoding the {prefix} dataframe")
    X_ohe = _OneHotEncode(
        pd_data, hierarchy_columns + categorical_features, numerical_features
    )

    log(f"Transforming the {prefix} one-hot encoded data into a CSR matrix")
    X_ohe_sparse = _data_frame_to_scipy_sparse_matrix(X_ohe, numerical_features)
    y = pd_data[target].values

    return X_ohe_sparse, y


def train_model(
    X_train_ohe_sparse,
    X_test_ohe_sparse,
    y_train,
    y_test,
    model_config: dict,
    run_id: str = None,
) -> list:

    """
    This function loads the model, train it using the CSR matrix then returns the prediction array.
    If a run_id was specified, a pretrained model (throught the MLFlow run) would be loaded.
    """

    # Parse parameters from the config file
    model_name: str = model_config["model"]
    model_params: dict = model_config.get("params", dict())

    # Loading the model
    if run_id:
        log(f"Loading the pretrained {model_name} model in run {run_id}")
        model = loadModel(model_name, model_params, run_id=run_id)
    else:
        log(f"Loading the {model_name} model")
        model = loadModel(model_name, model_params)

        log(f"Fitting the {model_name} model")
        model.fit(X_train=X_train_ohe_sparse, y_train=y_train)

    # Inferencing
    log("Generating predictions")
    y_pred = model.predict(X_test_ohe_sparse)
    mean_squared_error_ = mean_squared_error(y_test, y_pred)
    log(f"{mean_squared_error_ =}")

    return y_pred


def MLFlow_train_model(
    X_train_ohe_sparse,
    X_test_ohe_sparse,
    y_train,
    y_test,
    model_config: dict,
    model_params: dict,
) -> list:

    """
    A function that takes a Pandas dataframe, prepare it for modeling, fits and trains the model.
    This function returns the a prediction dataset.
    """
    with mlflow.start_run():
        # Parse parameters from the config file
        model_name: str = model_config["model"]

        run_id = mlflow.last_active_run().info.run_id
        log(f"Logged data and model in run {run_id}")

        # Loading the model
        log(f"[Run {run_id}]: Loading the {model_name} model")
        model = loadModel(model_name, model_params)

        # Enable MLFlow logs tracking
        model.mlflow_logs

        # Fitting the model
        log(f"[Run {run_id}]: Fitting the {model_name} model")
        model.fit(X_train=X_train_ohe_sparse, y_train=y_train)

        # Inferencing
        log(f"[Run {run_id}]: Generating predictions")
        y_pred = model.predict(X_test_ohe_sparse)

        # Evaluation
        mean_squared_error_ = mean_squared_error(y_test, y_pred)
        log(f"[Run {run_id}]: {mean_squared_error_=}")
        mlflow.log_metric("mse", mean_squared_error_)

        # Saving the model as an artifact.
        model.mlflow_log_model()

        return y_pred
