import logging
from uuid import uuid4
from prometheus_fastapi_instrumentator import Instrumentator

import asyncio
from contextlib import asynccontextmanager, suppress

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from olist_mlops.schema import INPUT_COLUMNS
from services.api.model_loader import load_champion
from services.api.schemas import OrderInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def refresh_model(app: FastAPI):
    while True:
        await asyncio.sleep(60)

        try:
            current = app.state.loaded_model

            current_version = current.version if current is not None else None

            new_model = await asyncio.to_thread(load_champion, current_version)

            if new_model is not None:
                app.state.loaded_model = new_model

                logger.info(
                    "Updated to model version %s",
                    new_model.version,
                )

        except Exception:
            logger.exception("Failed to refresh champion model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_model = None

    try:
        app.state.loaded_model = await asyncio.to_thread(load_champion)

        logger.info(
            "Loaded model version %s",
            app.state.loaded_model.version,
        )

    except Exception:
        logger.exception("Failed to load champion model")

    refresh_task = asyncio.create_task(refresh_model(app))

    try:
        yield

    finally:
        refresh_task.cancel()

        with suppress(asyncio.CancelledError):
            await refresh_task


app = FastAPI(
    title="Olist Late Delivery API",
    description="Predicts late delivery risk at order approval.",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: OrderInput, request: Request):
    loaded_model = request.app.state.loaded_model

    if loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available yet",
        )

    order_id = order.order_id or str(uuid4())

    features = order.model_dump(include=set(INPUT_COLUMNS))
    X = pd.DataFrame([features], columns=INPUT_COLUMNS)

    probability = float(loaded_model.model.predict_proba(X)[0, 1])

    return {
        "order_id": order_id,
        "late_probability": probability,
        "flagged": probability >= loaded_model.threshold,
        "model_version": loaded_model.version,
    }
