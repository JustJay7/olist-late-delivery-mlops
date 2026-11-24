from fastapi import FastAPI

app = FastAPI(
    title="Olist Late Delivery API",
    description="Predicts late delivery risk at order approval.",
)


@app.get("/health")
def health():
    return {"status": "ok"}
