ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.3.1

RUN pip --no-cache-dir install scikit-learn
RUN pip --no-cache-dir install xgboost
RUN pip --no-cache-dir install pyarrow
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}
RUN pip --no-cache-dir install pandas
RUN pip --no-cache-dir install scipy
RUN pip --no-cache-dir install ipykernel

ENTRYPOINT ["bash"]