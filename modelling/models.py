from abc import ABC, abstractmethod

import xgboost as xgb


class Model(ABC):
    """Model abstract class"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class XGBoostModel(Model):
    def __init__(self, params: dict = {}) -> None:
        self._params = params
        self._model = xgb.XGBRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter for the model
        """
        return self._model

    @property
    def params(self):
        """
        Getter for model parameters
        """
        return self._params

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
