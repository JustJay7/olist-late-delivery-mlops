FROM python:3.12-slim

# Same versions as requirements-dev.txt. psycopg2 lets MLflow keep its run records in Postgres.
RUN pip install --no-cache-dir mlflow==3.18.0 psycopg2-binary==2.9.13

EXPOSE 5000
