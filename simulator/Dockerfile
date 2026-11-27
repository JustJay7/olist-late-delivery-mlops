FROM python:3.12-slim

WORKDIR /app

COPY simulator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# The replay orders are built from the raw tables at start-up (about 5 seconds), so the
# image carries them (about 40 MB) and runs the same from compose or from the registry.
COPY data/raw/ data/raw/
COPY simulator/ simulator/

# Arguments after the service name are passed through: docker compose run --rm simulator --drift
ENTRYPOINT ["python", "-m", "simulator.simulate"]
