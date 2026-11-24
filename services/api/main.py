from uuid import uuid4

from fastapi import FastAPI

from services.api.schemas import OrderInput

app = FastAPI(
    title="Olist Late Delivery API",
    description="Predicts late delivery risk at order approval.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: OrderInput):
    order_id = order.order_id or str(uuid4())

    # Temporary prediction until the MLflow champion is connected.
    probability = 0.18
    threshold = 0.15

    return {
        "order_id": order_id,
        "late_probability": probability,
        "flagged": probability >= threshold,
        "model_version": "stub",
    }
