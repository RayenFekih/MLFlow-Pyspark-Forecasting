# In case of adding a new feature, creation of a subclass of Feature class is required
# [TODO]: Refactore into a dynamic plugin architecture

from abc import ABC, abstractmethod

import pyspark.sql.functions as F
from pyspark.sql import DataFrame as SparkDataFrame


class Feature(ABC):
    @abstractmethod
    def build(self, data: SparkDataFrame) -> None:
        pass


class Elasticity(Feature):
    def __init__(self) -> None:
        self.output_column = "_log_discount_rate"
        self.output_type = "numeric"

    def build(self, data: SparkDataFrame) -> SparkDataFrame:

        data = (
            data.withColumn(self.output_column, 1.0 - F.col("discount"))
            .fillna({self.output_column: 1.0})
            .withColumn(self.output_column, F.log(F.col(self.output_column)))
        )

        return data


class DiscountRate(Feature):
    def __init__(self) -> None:
        self.output_column = "_discount_rate"
        self.output_type = "numeric"

    def build(self, data: SparkDataFrame) -> SparkDataFrame:

        data = data.withColumn(self.output_column, 1.0 - F.col("discount")).fillna(
            {self.output_column: 1.0}
        )

        return data


class PromotedPercent(Feature):
    def __init__(self, promoted_hierarchy: str, group_key: str) -> None:
        self.promoted_hierarchy = promoted_hierarchy
        self.group_key = group_key
        self.output_column = f"_promoted_{promoted_hierarchy}_{self.group_key}"
        self.output_type = "numeric"

    def build(self, data: SparkDataFrame, **kwargs) -> SparkDataFrame:

        # Filter on promoted data and calculate number of promoted promoted_hierarchy per group_key
        # (i.e. number of promoted skus/stores per subclass/region to count for cannibalization)

        promoted_prods_df = (
            data.where(F.col("discount") > 0)
            .groupby(self.group_key)
            .agg(F.countDistinct(self.promoted_hierarchy).alias(self.output_column))
        )

        data = data.join(promoted_prods_df, how="inner", on=self.group_key)

        return data


class WeekOfYear(Feature):
    def __init__(self) -> None:
        self.output_column = "_week_of_year"
        self.output_type = "categorical"

    def build(self, data: SparkDataFrame) -> SparkDataFrame:

        data = (
            data.withColumn(self.output_column, F.weekofyear(F.col("date")))
            .withColumn(
                self.output_column,
                F.when(F.col(self.output_column) == 53, 52).otherwise(
                    F.col(self.output_column)
                ),
            )
            .withColumn(self.output_column, F.col(self.output_column).cast("string"))
        )
        return data


class MonthOfYear(Feature):
    def __init__(self) -> None:
        self.output_column = "_week_of_year"
        self.output_type = "categorical"

    def build(self, data: SparkDataFrame) -> SparkDataFrame:

        data = data.withColumn(self.output_column, F.month(F.col("date")))
        return data


class PromotionCategory(Feature):
    def __init__(self) -> None:
        self.output_column = "_promo_category"
        self.output_type = "categorical"

    def build(self, data: SparkDataFrame) -> SparkDataFrame:

        data = data.withColumn(self.output_column, F.col("promo_cat"))
        return data
