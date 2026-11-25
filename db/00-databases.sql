-- Runs first on an empty Postgres volume (files run in name order).
-- MLflow and Prefect each get their own database, so their tables never mix with ours.
-- The app tables (predictions, labels, metrics) go in the main database from .env.
CREATE DATABASE mlflow;
CREATE DATABASE prefect;
