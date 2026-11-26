-- Store model predictions for monitoring.
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    order_id TEXT NOT NULL,
    features JSONB NOT NULL,
    late_probability DOUBLE PRECISION NOT NULL
        CHECK (late_probability BETWEEN 0 AND 1),
    flagged BOOLEAN NOT NULL,
    model_version TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'frontend'
);

-- Support efficient retrieval of recent predictions.
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_order_id
    ON predictions (order_id);

-- Store ground-truth delivery outcomes.
CREATE TABLE IF NOT EXISTS labels (
    order_id TEXT PRIMARY KEY,
    is_late BOOLEAN NOT NULL,
    labelled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Store data drift results for monitoring and retraining.
CREATE TABLE IF NOT EXISTS drift_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    n_rows INTEGER NOT NULL CHECK (n_rows >= 0),
    share_drifted_columns DOUBLE PRECISION NOT NULL
        CHECK (share_drifted_columns BETWEEN 0 AND 1),
    dataset_drift BOOLEAN NOT NULL,
    retrain_triggered BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_drift_metrics_run_at
    ON drift_metrics (run_at DESC);


-- Store model performance metrics once labels are available.
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    n_labelled INTEGER NOT NULL CHECK (n_labelled >= 0),
    pr_auc DOUBLE PRECISION
        CHECK (pr_auc BETWEEN 0 AND 1),
    recall_top10 DOUBLE PRECISION
        CHECK (recall_top10 BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_run_at
    ON performance_metrics (run_at DESC);