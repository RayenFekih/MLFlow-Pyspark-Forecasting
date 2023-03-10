from abc import ABC, abstractmethod

import xgboost as xgb
import lightgbm as lgbm
import mlflow


class Model(ABC):
    """Model abstract class"""

    @abstractmethod
    def mlflow_logs(self):
        pass

    @abstractmethod
    def mlflow_log_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class XGBoostModel(Model):
    def __init__(self, params: dict = {}) -> None:
        self.params = params
        self.model = xgb.XGBRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def mlflow_logs(self):
        mlflow.xgboost.autolog()

    def mlflow_log_model(self, flavor="sklearn"):
        if flavor == "sklearn":
            mlflow.sklearn.log_model(self, "model")
        # [TODO]: Add more mlflow flavors
        # if flavor == "xgboost":
        #     mlflow.xgboost.log_model(self, "model")

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class LgbmModel(Model):
    def __init__(self, params: dict = {}) -> None:
        self.params = params
        self.model = lgbm.LGBMRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def mlflow_logs(self):
        mlflow.lightgbm.autolog()
    
    def mlflow_log_model(self, flavor="sklearn"):
        if flavor == "sklearn":
            mlflow.sklearn.log_model(self, "model")

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
