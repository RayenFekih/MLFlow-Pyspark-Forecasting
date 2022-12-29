import datetime as dt

import pyspark.sql.functions as F
from pyspark import SparkContext
from pyspark.sql import DataFrame as SparkDataFrame


def log(msg):
    """
    Logging function
    """
    print("[" + str(dt.datetime.now()) + "] " + str(msg))


def serialize(
    spark: SparkContext, df: SparkDataFrame, path: str, save_mode: str = "overwrite"
) -> SparkDataFrame:
    """
    This function saves the data locally then read it in order to cache it
    """

    df.write.parquet(path, save_mode)
    return spark.read.parquet(path, header="true", inferSchema="true")


def gapfilling(sales, date_column: str, product_column: str, location_column: str):
    """
    Compute domain by looking at first and last sales date for products in individual stores.
    """
    # get start and end dates assuming all products are selling for the entire life cycle
    start_end = sales.agg(
        F.max(date_column).alias("end_date"), F.min(date_column).alias("start_date")
    )

    # generate all possible intersections
    intersection = sales.select(product_column, location_column).distinct()

    domain = (
        intersection.join(start_end)
        .withColumn(
            date_column,
            F.explode(F.expr("sequence(start_date, end_date, interval 1 week)")),
        )
        .join(sales, on=[date_column, product_column, location_column], how="left")
        .drop("start_date", "end_date")
    ).fillna(
        {"units": 0}
    )  # Assuming that missing weeks are due to 0 demand

    return domain
