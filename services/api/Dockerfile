FROM python:3.12-slim

WORKDIR /app

COPY services/api/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

COPY services/api/ services/api/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]