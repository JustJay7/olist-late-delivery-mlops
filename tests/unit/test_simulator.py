"""The simulator sends valid orders, drifts only the columns it names, and labels what it sends."""

import pandas as pd
import pytest

from olist_mlops import config
from olist_mlops.ingest import SAMPLE_DIR
from olist_mlops.schema import INPUT_COLUMNS, NULLABLE_COLUMNS, TARGET
from services.api.schemas import OrderInput
from simulator import simulate


@pytest.fixture(scope="module")
def sample():
    return pd.read_parquet(SAMPLE_DIR / "orders_sample.parquet")


@pytest.fixture(scope="module")
def replay():
    return simulate.load_replay()


def drift_threshold() -> float:
    for line in (config.ROOT / ".env.example").read_text().splitlines():
        if line.startswith("DRIFT_THRESHOLD="):
            return float(line.split("=", 1)[1])
    raise AssertionError("DRIFT_THRESHOLD missing from .env.example")


def test_drift_changes_only_the_declared_columns(sample):
    drifted = simulate.inject_drift(sample)
    changed = {
        c for c in INPUT_COLUMNS if not drifted[c].equals(sample[c])
    }  # NaN == NaN counts as equal here
    assert changed == set(simulate.DRIFTED_COLUMNS)
    assert drifted[TARGET].equals(sample[TARGET])  # true outcomes are never altered
    assert drifted["order_id"].equals(sample["order_id"])


def test_drift_moves_enough_columns_to_cross_the_threshold():
    assert len(simulate.DRIFTED_COLUMNS) / len(INPUT_COLUMNS) > drift_threshold()


def test_drift_keeps_missing_distances_missing(sample):
    drifted = simulate.inject_drift(sample)
    for column in ("mean_distance_km", "max_distance_km"):
        assert drifted[column].isna().equals(sample[column].isna())


def test_payloads_are_plain_json_with_nulls_for_missing(sample):
    with_missing = sample[sample[NULLABLE_COLUMNS].isna().any(axis=1)].head(5)
    assert len(with_missing) > 0
    payloads = simulate.to_payloads(with_missing, "simulator")
    assert len(payloads) == len(with_missing)
    for payload, (_, row) in zip(payloads, with_missing.iterrows()):
        assert payload["order_id"] == row["order_id"]
        assert payload["source"] == "simulator"
        assert set(payload) == {"order_id", "source", *INPUT_COLUMNS}
        assert type(payload["purchase_hour"]) is int
        for column in NULLABLE_COLUMNS:
            assert (payload[column] is None) == pd.isna(row[column])


@pytest.mark.parametrize("drift", [False, True])
def test_every_replay_order_passes_the_api_validation(replay, drift):
    orders = simulate.inject_drift(replay) if drift else replay
    source = "simulator-drift" if drift else "simulator"
    for payload in simulate.to_payloads(orders, source):
        OrderInput(**payload)  # raises on any value the API would answer with a 422


def test_pick_orders_is_a_seeded_sample_in_purchase_order(replay):
    first = simulate.pick_orders(replay, 300, seed=1)
    assert len(first) == 300
    assert (
        first["order_id"].tolist() == simulate.pick_orders(replay, 300, seed=1)["order_id"].tolist()
    )
    assert first["order_purchase_timestamp"].is_monotonic_increasing
    assert first["order_purchase_timestamp"].min() >= config.TEST_START
    # Spread over the whole replay period, not only its first weeks
    assert first["order_purchase_timestamp"].dt.month.nunique() >= 3
    assert len(simulate.pick_orders(replay, 10**6)) == len(replay)


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code, self._body, self.text = status_code, body or {}, "bad input"
        self.ok = status_code < 400

    def json(self):
        return self._body


class FakeSession:
    """Accepts every order except the ones in reject, and flags every second one."""

    def __init__(self, reject=()):
        self.posted, self.reject = [], set(reject)

    def get(self, url, timeout):
        return FakeResponse(200)

    def post(self, url, json, timeout):
        self.posted.append(json)
        if json["order_id"] in self.reject:
            return FakeResponse(422)
        return FakeResponse(200, {"flagged": len(self.posted) % 2 == 0})


def run_main(monkeypatch, sample, argv, reject=()):
    session, written = FakeSession(reject), []
    monkeypatch.setattr(simulate, "load_replay", lambda: sample)
    monkeypatch.setattr(simulate.requests, "Session", lambda: session)
    monkeypatch.setattr(simulate, "connect_db", lambda: _FakeConn())
    monkeypatch.setattr(simulate, "write_labels", lambda conn, labels: written.extend(labels))
    code = simulate.main(argv)
    return code, session.posted, written


class _FakeConn:
    def close(self):
        pass


def test_main_posts_orders_and_labels_only_the_accepted_ones(monkeypatch, sample):
    reject = set(sample["order_id"].head(3))
    code, posted, labels = run_main(
        monkeypatch, sample, ["-n", "150", "--rate", "0", "--seed", "0"], reject
    )
    assert code == 0
    assert len(posted) == 150
    assert {p["source"] for p in posted} == {"simulator"}
    sent_ids = {p["order_id"] for p in posted}
    truth = dict(zip(sample["order_id"], sample[TARGET].astype(bool)))
    assert {order_id for order_id, _ in labels} == sent_ids - reject
    assert all(truth[order_id] == is_late for order_id, is_late in labels)


def test_main_with_drift_sends_drifted_values(monkeypatch, sample):
    _, posted, _ = run_main(monkeypatch, sample, ["--drift", "-n", "20", "--rate", "0"])
    assert {p["source"] for p in posted} == {"simulator-drift"}
    assert all(p["same_state_item_share"] == 0.0 for p in posted)


def test_main_without_labels_never_touches_the_database(monkeypatch, sample):
    def no_db():
        raise AssertionError("connect_db should not be called with --no-labels")

    monkeypatch.setattr(simulate, "connect_db", no_db)
    monkeypatch.setattr(simulate, "load_replay", lambda: sample)
    monkeypatch.setattr(simulate.requests, "Session", lambda: FakeSession())
    assert simulate.main(["--no-labels", "-n", "5", "--rate", "0"]) == 0


def test_bad_arguments_are_refused():
    with pytest.raises(SystemExit):
        simulate.parse_args(["-n", "0"])
    with pytest.raises(SystemExit):
        simulate.parse_args(["--rate", "-1"])
