"""Replays unseen 2018 orders as live traffic to the model API, with a switch that injects drift.

Each order is posted to the API's /predict, like the web form does, and its true outcome is
written to the labels table in Postgres, so monitoring can join labels to logged predictions.
The orders come from the replay part of the serving split (ingest.split_serving): Apr-Jul 2018
orders that training never sees and that are not in the drift baseline.

    docker compose run --rm simulator            500 normal orders
    docker compose run --rm simulator --drift    500 drifted orders
    python -m simulator.simulate --help          all options (runs from a laptop too)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import requests

from olist_mlops.ingest import build_orders, split_serving
from olist_mlops.schema import COLUMN_SPECS, INPUT_COLUMNS, TARGET

# The drift scenario: a holiday surge handled by a new long-haul carrier. Olist starts promising
# faster delivery, payment approval backs up, and more orders ship from distant, out-of-state
# sellers at twice the freight cost. 7 of the 19 inputs move, a share of 0.37, which is above
# the 0.25 DRIFT_THRESHOLD in .env, so monitoring should raise an alert and start a retrain.
PROMISE_FACTOR = 0.6  # promises are 40% shorter
EXTRA_APPROVAL_HOURS = 36.0  # a day and a half of payment backlog
FREIGHT_FACTOR = 2.0  # freight doubles; prices stay the same
DISTANCE_FACTOR = 1.6  # sellers are 60% further away
DRIFTED_COLUMNS = (
    "promised_days_from_approval",
    "approval_lag_hours",
    "total_freight",
    "freight_to_price_ratio",
    "mean_distance_km",
    "max_distance_km",
    "same_state_item_share",
)

LABEL_BATCH = 100  # labels are written every 100 accepted orders, so a stopped run keeps most


def load_replay() -> pd.DataFrame:
    """The 16,333 replay orders, built from the committed raw tables (about 5 seconds)."""
    _, replay = split_serving(build_orders())
    return replay


def pick_orders(replay: pd.DataFrame, n: int, seed: int | None = None) -> pd.DataFrame:
    """A random sample of n orders, sent in purchase order.

    Random rather than the first n, so normal traffic covers Apr-Jul like the baseline does.
    Replaying only April would look like drift to monitoring even without --drift.
    """
    picked = replay.sample(n=min(n, len(replay)), random_state=seed)
    return picked.sort_values("order_purchase_timestamp")


def _clip(values: pd.Series, column: str) -> pd.Series:
    """Keep a drifted value inside the schema range, so the API accepts every drifted order."""
    spec = COLUMN_SPECS[column]
    return values.clip(lower=spec.min, upper=spec.max)


def inject_drift(orders: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of orders with the drift scenario applied to DRIFTED_COLUMNS only."""
    out = orders.copy()
    out["promised_days_from_approval"] = _clip(
        out["promised_days_from_approval"] * PROMISE_FACTOR, "promised_days_from_approval"
    )
    out["approval_lag_hours"] = _clip(
        out["approval_lag_hours"] + EXTRA_APPROVAL_HOURS, "approval_lag_hours"
    )
    out["total_freight"] = _clip(out["total_freight"] * FREIGHT_FACTOR, "total_freight")
    # Freight doubles and price does not, so the ratio doubles with it.
    out["freight_to_price_ratio"] = _clip(
        out["freight_to_price_ratio"] * FREIGHT_FACTOR, "freight_to_price_ratio"
    )
    for column in ("mean_distance_km", "max_distance_km"):
        out[column] = _clip(out[column] * DISTANCE_FACTOR, column)  # missing stays missing
    out["same_state_item_share"] = 0.0  # every item now ships from another state
    return out


def to_payloads(orders: pd.DataFrame, source: str) -> list[dict]:
    """One /predict request body per order: the 19 inputs, the real order_id and the source.

    Going through to_json turns NaN into null and numpy numbers into plain JSON numbers.
    """
    records = json.loads(orders[["order_id", *INPUT_COLUMNS]].to_json(orient="records"))
    return [{**record, "source": source} for record in records]


def wait_for_api(session: requests.Session, api_url: str, timeout: float) -> None:
    """Block until the API answers /health. On the first start it waits for a trained model."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if session.get(f"{api_url}/health", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"The API at {api_url} did not answer /health within {timeout:.0f}s.")
        print(f"Waiting for the API at {api_url} ...", flush=True)
        time.sleep(5)


def post_order(
    session: requests.Session, api_url: str, payload: dict, deadline: float
) -> requests.Response | None:
    """POST one order. A 503 means no model is loaded yet, so retry until the deadline."""
    while True:
        try:
            response = session.post(f"{api_url}/predict", json=payload, timeout=10)
        except requests.RequestException:
            return None
        if response.status_code != 503 or time.monotonic() > deadline:
            return response
        print("The API has no model loaded yet; retrying in 10s ...", flush=True)
        time.sleep(10)


def connect_db():
    """Connect to the app database with the same settings the other services read from .env."""
    import psycopg2  # imported here so --no-labels runs without a database driver

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def write_labels(conn, labels: list[tuple[str, bool]]) -> None:
    """Insert true outcomes. An order replayed again keeps its first label, which is the same."""
    if not labels:
        return
    from psycopg2.extras import execute_values

    with conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO labels (order_id, is_late) VALUES %s ON CONFLICT (order_id) DO NOTHING",
            labels,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drift", action="store_true", help="apply the drift scenario")
    parser.add_argument("-n", type=int, default=500, help="orders to send (default 500)")
    parser.add_argument(
        "--rate", type=float, default=10.0, help="orders per second, 0 for no pause (default 10)"
    )
    parser.add_argument("--seed", type=int, default=None, help="fix the sample, for a rerun")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("API_URL", "http://localhost:8000"),
        help="model API address (default $API_URL, else http://localhost:8000)",
    )
    parser.add_argument(
        "--wait", type=float, default=600, help="seconds to wait for the API (default 600)"
    )
    parser.add_argument(
        "--no-labels", action="store_true", help="post orders only, without writing labels"
    )
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("-n must be at least 1")
    if args.rate < 0:
        parser.error("--rate cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = "simulator-drift" if args.drift else "simulator"

    orders = pick_orders(load_replay(), args.n, args.seed)
    if args.drift:
        orders = inject_drift(orders)
    payloads = to_payloads(orders, source)
    is_late = dict(zip(orders["order_id"], orders[TARGET].astype(bool)))

    session = requests.Session()
    wait_for_api(session, args.api_url, args.wait)
    conn = None if args.no_labels else connect_db()
    deadline = time.monotonic() + args.wait

    print(f"Sending {len(payloads)} {source} orders to {args.api_url}/predict", flush=True)
    accepted, rejected, failed, flagged = 0, 0, 0, 0
    pending: list[tuple[str, bool]] = []
    interval = 1 / args.rate if args.rate else 0.0
    next_send = time.monotonic()
    for i, payload in enumerate(payloads, start=1):
        response = post_order(session, args.api_url, payload, deadline)
        if response is None or response.status_code >= 500:
            failed += 1
        elif response.status_code == 422:
            rejected += 1
            print(f"Rejected {payload['order_id']}: {response.text[:200]}", flush=True)
        elif response.ok:
            accepted += 1
            flagged += bool(response.json().get("flagged"))
            pending.append((payload["order_id"], is_late[payload["order_id"]]))
        else:
            failed += 1
        if conn is not None and len(pending) >= LABEL_BATCH:
            write_labels(conn, pending)
            pending = []
        if i % 100 == 0:
            print(f"  {i}/{len(payloads)} sent", flush=True)
        next_send += interval
        time.sleep(max(0.0, next_send - time.monotonic()))

    if conn is not None:
        write_labels(conn, pending)
        conn.close()

    flagged_share = flagged / accepted if accepted else 0.0
    print(
        f"Done: {accepted} accepted, {rejected} rejected (422), {failed} failed. "
        f"{flagged_share:.1%} of accepted orders flagged as high risk."
    )
    if not args.no_labels:
        print(f"Labels written to Postgres for the {accepted} accepted orders.")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
