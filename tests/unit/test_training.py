"""Model vs baseline, and the champion rule that decides which model is served."""

import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes flows/ importable

from flows.training_flow import BASELINE, REFERENCE, make_candidates, passes_rule  # noqa: E402
from olist_mlops import config  # noqa: E402
from olist_mlops.data import time_split  # noqa: E402
from olist_mlops.ingest import build_orders  # noqa: E402
from olist_mlops.schema import INPUT_COLUMNS  # noqa: E402


def test_rule_needs_four_of_five_folds():
    reference = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
    assert passes_rule(reference + [0.02, 0.02, 0.02, 0.02, -0.01], reference)
    assert not passes_rule(reference + [0.02, 0.02, 0.02, -0.01, -0.01], reference)


def test_rule_needs_gain_above_one_standard_error():
    reference = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
    # Better on 4 folds, but the one bad fold is so bad that the mean gain is noise.
    assert not passes_rule(reference + [0.001, 0.001, 0.001, 0.001, -0.05], reference)


def test_identical_model_does_not_replace_the_champion():
    folds = np.array([0.15, 0.12, 0.30, 0.08, 0.20])
    assert not passes_rule(folds, folds)


def test_model_beats_b1_on_unseen_orders():
    # Trained on Jan 2017 to Mar 2018, scored on Apr to Jul 2018, which it never saw.
    train, test = time_split(build_orders())
    fit = train.iloc[: len(train) - config.CV_GAP]
    candidates = make_candidates()
    pr_auc = {}
    for name in (BASELINE, REFERENCE):
        model = clone(candidates[name]).fit(fit[INPUT_COLUMNS], fit[config.TARGET])
        scores = model.predict_proba(test[INPUT_COLUMNS])[:, 1]
        pr_auc[name] = average_precision_score(test[config.TARGET], scores)
    assert pr_auc[REFERENCE] > pr_auc[BASELINE] > test[config.TARGET].mean()
