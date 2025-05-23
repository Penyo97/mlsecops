FROM apache/airflow:2.9.1

USER root

USER airflow
RUN pip install --no-cache-dir mlflow requests