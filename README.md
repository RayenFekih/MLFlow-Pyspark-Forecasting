# Pyspark-Forecasting

[TODO]: write a proper readme file

This repo contains a demand forecasting engine that uses mainly Pyspark code alongside with some of MLFlow work (still in progress).
This engine is designed to be configurable, maintanable and extensible and aims to have a plugin architecture in almost all the pipelines steps. For example, in the model phase, adding a model class that respects the models protocol (defined in the ABC class) should be enough to be ready for production.
