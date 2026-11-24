"""Training pipeline as a Prefect flow: ingest, validate, split, cross-validate, register.

Reproduces the INSY 674 model selection (locked decision 5): the B1 baseline and the three
tuned candidates are scored with the same time-series CV, each logged as its own MLflow run,
and the winner is registered as olist-late-delivery with the alias champion.

Run locally:  python flows/training_flow.py
"""

import os

import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from mlflow import MlflowClient
from mlflow.models import infer_signature
from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from olist_mlops import config
from olist_mlops.data import time_split
from olist_mlops.evaluation import cv_splitter, recall_at_budget, scorers
from olist_mlops.features import make_preprocess
from olist_mlops.ingest import build_orders
from olist_mlops.schema import COLUMN_SPECS, INPUT_COLUMNS

EXPERIMENT = "olist-late-delivery"
MODEL_NAME = "olist-late-delivery"
ALIAS = "champion"
BASELINE = "b1_baseline"
REFERENCE = "logistic_regression"
RS = config.RANDOM_STATE


# ---------------------------------------------------------------- candidates (decision 5)
def make_candidates() -> dict[str, Pipeline]:
    """The four models of 674 section 5, with the settings the tuning selected there."""
    b1_pre = ColumnTransformer(
        [
            (
                "days",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["promised_days_from_approval"],
            ),
            (
                "state",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                ["customer_state"],
            ),
        ],
        sparse_threshold=0,
    )
    return {
        # B1: the two most obvious signals. Every model has to beat it.
        BASELINE: Pipeline(
            [("pre", b1_pre), ("clf", LogisticRegression(max_iter=1000, random_state=RS))]
        ),
        # l1_ratio=1 is the scikit-learn 1.9 spelling of penalty="l1" (decision 2).
        REFERENCE: Pipeline(
            [
                ("pre", make_preprocess(min_frequency=20)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.15728326259078687,
                        l1_ratio=1,
                        solver="liblinear",
                        max_iter=5000,
                        random_state=RS,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", make_preprocess(min_frequency=100)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=18,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RS,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("pre", make_preprocess(min_frequency=100)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.01351182947645082,
                        max_iter=395,
                        max_leaf_nodes=27,
                        min_samples_leaf=227,
                        l2_regularization=4.86964094152089,
                        max_features=0.4371880036934016,
                        early_stopping=False,
                        random_state=RS,
                    ),
                ),
            ]
        ),
    }


def passes_rule(challenger_folds: np.ndarray, reference_folds: np.ndarray) -> bool:
    """674 champion rule: better on at least 4 of 5 folds, and mean gain above one SE.

    Paired by fold, so a hard fold counts against both models equally.
    """
    gain = np.asarray(challenger_folds) - np.asarray(reference_folds)
    se = gain.std(ddof=1) / np.sqrt(len(gain))
    return bool((gain > 0).sum() >= 4 and gain.mean() > se)


def cv_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-fold PR-AUC, lift and recall, with the same folds for every model."""
    cv = cv_splitter()
    res = cross_validate(model, X, y, cv=cv, scoring=scorers(), n_jobs=1)
    val_rate = np.array([y.iloc[va].mean() for _, va in cv.split(X)])
    folds = pd.DataFrame({k.removeprefix("test_"): v for k, v in res.items() if "test_" in k})
    # Lift divides by each fold's own late rate, which moves a lot over time (D8).
    folds["lift"] = folds["pr_auc"] / val_rate
    folds.index = pd.RangeIndex(1, len(folds) + 1, name="fold")
    return folds


# ---------------------------------------------------------------- tasks
@task
def ingest() -> pd.DataFrame:
    """One row per order, built from the committed raw tables, no notebook involved."""
    return build_orders()


@task
def validate(orders: pd.DataFrame) -> pd.DataFrame:
    """Stop the flow before training if the data breaks the shared schema."""
    missing = [c for c in INPUT_COLUMNS + [config.TARGET] if c not in orders.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if orders[config.TARGET].isna().any() or not orders[config.TARGET].isin([0, 1]).all():
        raise ValueError(f"{config.TARGET} must be 0 or 1 on every row.")
    for col, spec in COLUMN_SPECS.items():
        values = orders[col]
        if not spec.nullable and values.isna().any():
            raise ValueError(f"{col} has missing values but is not nullable.")
        if spec.kind == "numeric":
            outside = values.notna() & ~values.between(spec.min, spec.max)
            if outside.any():
                raise ValueError(f"{col} has {int(outside.sum())} values outside its range.")
        elif spec.allowed is not None:
            unknown = set(values.dropna()) - set(spec.allowed)
            if unknown:
                raise ValueError(f"{col} has unknown values: {sorted(unknown)}")
    return orders


@task
def split(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on purchases before 1 Apr 2018. Later orders are serving traffic, never trained on."""
    orders = orders.sort_values("order_purchase_timestamp", kind="stable")
    return time_split(orders.reset_index(drop=True))


@task
def evaluate_candidate(name: str, model: Pipeline, train: pd.DataFrame, flow_run: str) -> dict:
    """Cross-validate one candidate and log it as its own MLflow run."""
    X, y = train[INPUT_COLUMNS], train[config.TARGET]
    folds = cv_scores(model, X, y)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"candidate": name, "prefect_flow_run": flow_run})
        mlflow.log_params(
            {f"clf__{k}": v for k, v in model.named_steps["clf"].get_params().items()}
            | {"n_train_orders": len(X), "cv_splits": config.N_CV_SPLITS, "cv_gap": config.CV_GAP}
        )
        mean, std = folds.mean(), folds.std()
        for metric in ["pr_auc", "lift", "recall_top10", "roc_auc", "recall_top5", "recall_top20"]:
            mlflow.log_metric(f"cv_{metric}", mean[metric])
            mlflow.log_metric(f"cv_{metric}_std", std[metric])
        for fold, value in folds["pr_auc"].items():
            mlflow.log_metric("fold_pr_auc", value, step=fold)
        mlflow.log_table(folds.reset_index(), artifact_file="cv_folds.json")
    return {"name": name, "run_id": run.info.run_id, "folds": folds}


@task
def select_champion(results: list[dict]) -> dict:
    """Apply decision 5: beat B1, and replace logistic regression only by passing the rule."""
    logger = get_run_logger()
    by_name = {r["name"]: r for r in results}
    b1 = by_name[BASELINE]["folds"]["pr_auc"].mean()
    eligible = {
        n: r for n, r in by_name.items() if n != BASELINE and r["folds"]["pr_auc"].mean() > b1
    }
    if REFERENCE not in eligible:
        raise RuntimeError("Logistic regression does not beat B1; something changed in the data.")
    ref = eligible[REFERENCE]["folds"]["pr_auc"]
    challengers = [
        r for n, r in eligible.items() if n != REFERENCE and passes_rule(r["folds"]["pr_auc"], ref)
    ]
    winner = max(
        challengers, key=lambda r: r["folds"]["pr_auc"].mean(), default=eligible[REFERENCE]
    )
    logger.info("B1 CV PR-AUC %.4f. Winner: %s", b1, winner["name"])
    return winner


@task
def fit_and_register(
    winner: dict, candidates: dict[str, Pipeline], train: pd.DataFrame, test: pd.DataFrame
) -> str:
    """Refit the winner, register it, and move the alias only if it beats the current champion."""
    logger = get_run_logger()
    # Leave out the last CV_GAP orders: their outcomes were not yet known on 1 Apr (D6).
    fit = train.iloc[: len(train) - config.CV_GAP]
    model = clone(candidates[winner["name"]]).fit(fit[INPUT_COLUMNS], fit[config.TARGET])

    train_scores = model.predict_proba(fit[INPUT_COLUMNS])[:, 1]
    # The API flags an order when its score is above this: the riskiest 10% of training orders.
    flag_threshold = float(np.quantile(train_scores, 1 - config.FLAG_BUDGET))
    y_test, s_test = test[config.TARGET], model.predict_proba(test[INPUT_COLUMNS])[:, 1]

    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.log_metric("flag_threshold", flag_threshold)
        # Apr to Jul 2018 is never trained on, so this is an honest one-off check.
        mlflow.log_metric("holdout_pr_auc", average_precision_score(y_test, s_test))
        mlflow.log_metric("holdout_lift", average_precision_score(y_test, s_test) / y_test.mean())
        mlflow.log_metric("holdout_recall_top10", recall_at_budget(y_test, s_test))
        example = fit[INPUT_COLUMNS].head(5)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            # MLflow 3 saves with skops, which only loads types listed here. We list the types
            # of the model we just trained, so nothing else can be smuggled into the file.
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)),
        )
    version = str(info.registered_model_version)

    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except mlflow.exceptions.MlflowException:
        current = None
    if current is None:
        promote = True
    else:
        # Same data, same folds: the current champion is re-scored so the comparison is fair.
        old = clone(mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}"))
        old_folds = cv_scores(old, train[INPUT_COLUMNS], train[config.TARGET])["pr_auc"]
        promote = passes_rule(winner["folds"]["pr_auc"], old_folds)
    if promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    logger.info(
        "Registered version %s; champion alias %s.", version, "moved" if promote else "kept"
    )
    return version


# ---------------------------------------------------------------- flow
@flow(name="training-flow")
def training_flow() -> str:
    """Train the four candidates, log them to MLflow and register the champion."""
    mlflow.set_experiment(EXPERIMENT)
    orders = validate(ingest())
    train, test = split(orders)
    candidates = make_candidates()
    results = [
        evaluate_candidate(name, model, train, flow_run.get_id() or "local")
        for name, model in candidates.items()
    ]
    winner = select_champion(results)
    return fit_and_register(winner, candidates, train, test)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
    training_flow()
