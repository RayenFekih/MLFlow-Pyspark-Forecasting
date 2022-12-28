import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pandas import DataFrame as PandasDataframe
from pyspark.sql import DataFrame as SparkDataFrame
from scipy.sparse import lil_matrix
from sklearn.metrics import mean_squared_error

from feature_engineering.engineering import featureConstructor
from modelling.models import Model, XGBoostModel
from utils.utils import log


def _splitData(
    data: SparkDataFrame,
    train_startDate: str,
    train_endDate: str,
    inference_startDate: str,
    inference_endDate: str,
) -> tuple[SparkDataFrame, SparkDataFrame]:

    train_data = data.where(F.col("date").between(train_startDate, train_endDate))

    inference_data = data.where(
        F.col("date").between(inference_startDate, inference_endDate)
    )

    return train_data, inference_data


def _OneHotEncode(
    data: PandasDataframe, categorical_columns: list, numerical_columns: list
):
    """
    Perform one hot encoding for categorical variables
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
    data: SparkDataFrame,
    hierarchy_columns: list,
    categorical_features: list,
    numerical_features: list,
    target: str,
    prefix: str,
):

    modelling_data = data.select(
        *hierarchy_columns + categorical_features + numerical_features, target
    )

    log(f"Saving the {prefix} SPARK dataframe")
    modelling_data.write.parquet(f"data/{prefix}_modelling_data.parquet", "overwrite")

    log(f"Reading the {prefix} dataframe using Pandas")
    pd_data = pd.read_parquet(f"data/{prefix}_modelling_data.parquet", "pyarrow")

    log(f"One-hot encoding the {prefix} dataframe")
    X_ohe = _OneHotEncode(
        pd_data, hierarchy_columns + categorical_features, numerical_features
    )

    log(f"Transforming the {prefix} one-hot encoded data into a CSR matrix")
    X_train_ohe_sparse = _data_frame_to_scipy_sparse_matrix(X_ohe, numerical_features)
    y = pd_data[target].values

    return X_train_ohe_sparse, y, pd_data


def loadModel(model_name: str, model_params: dict) -> Model:
    match model_name:
        case "xgboost":
            return XGBoostModel(model_params)
        case _:
            raise ValueError(f"{model_name} model in undefined")


def train_model(
    data: SparkDataFrame, model_config: dict, features_config: dict
) -> PandasDataframe:
    """
    A function that takes a spark data frame, prepare it for modeling, fits and trains the model.
    This function returns the a prediction dataset.
    """

    # Parse parameters from the config file
    model_name: str = model_config["model"]
    model_params: dict = model_config.get("params", dict())
    hierarchy_columns: list = model_config["hierarchy_columns"]
    target: str = model_config["target"]

    train_startDate: str = model_config["train_startDate"]
    train_endDate: str = model_config["train_endDate"]
    inference_startDate: str = model_config["inference_startDate"]
    inference_endDate: str = model_config["inference_endDate"]

    features = featureConstructor(features_config)
    categorical_features = [
        f.output_column for f in features if f.output_type == "categorical"
    ]
    numerical_features = [
        f.output_column for f in features if f.output_type == "numeric"
    ]
    train_data, inference_data = _splitData(
        data=data,
        train_startDate=train_startDate,
        train_endDate=train_endDate,
        inference_startDate=inference_startDate,
        inference_endDate=inference_endDate,
    )

    # Preparing the training data
    X_train_ohe_sparse, y_train, _ = prepare_data(
        train_data,
        hierarchy_columns,
        categorical_features,
        numerical_features,
        target,
        prefix="train",
    )

    # Loading the model
    model = loadModel(model_name, model_params)

    # Fitting the model
    model.fit(X_train=X_train_ohe_sparse, y_train=y_train)

    # preparing inferencing data
    X_test_ohe_sparse, y_test, X_test_pandas = prepare_data(
        inference_data,
        hierarchy_columns,
        categorical_features,
        numerical_features,
        target,
        prefix="test",
    )

    # Inferencing
    y_pred = model.predict(X_test_ohe_sparse)
    mean_squared_error_ = mean_squared_error(y_test, y_pred)
    log(f"{mean_squared_error_ =}")

    X_test_pandas["forecast"] = y_pred.tolist()

    return X_test_pandas
