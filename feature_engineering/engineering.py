# [TODO]: Refactore into a dynamic plugin architecture
from pyspark.sql import DataFrame as SparkDataFrame

from feature_engineering.features import (DiscountRate, Elasticity, Feature,
                                          MonthOfYear, PromotedPercent,
                                          PromotionCategory, WeekOfYear)


def featureConstructor(config: dict) -> list[type[Feature]]:
    """
    Construct all features
    """
    features = []
    for feature, params in config.items():
        feature = feature.lower()
        match feature:
            case "elasticity":
                features.append(Elasticity(**params))
            case "discount_rate":
                features.append(DiscountRate(**params))
            case "promoted_percent":
                features.append(PromotedPercent(**params))
            case "week_of_year":
                features.append(WeekOfYear(**params))
            case "month_of_year":
                features.append(MonthOfYear(**params))
            case "promo_category":
                features.append(PromotionCategory(**params))
            case _:
                raise ValueError("No such feature: " + feature)

    return features


def engineerFeatures(data: SparkDataFrame, config: dict) -> SparkDataFrame:
    features = featureConstructor(config)

    for f in features:
        data = f.build(data=data)  # type: ignore

    return data
