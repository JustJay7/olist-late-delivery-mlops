FROM python:3.12-slim

# Same version as requirements-dev.txt; it already includes the Postgres driver (asyncpg).
RUN pip install --no-cache-dir prefect==3.9.4

EXPOSE 4200
