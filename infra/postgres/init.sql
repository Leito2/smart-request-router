CREATE TABLE IF NOT EXISTS messages (message_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, lang TEXT,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS decisions (message_id TEXT PRIMARY KEY, route TEXT NOT NULL, priority TEXT NOT NULL,
  decided_by TEXT NOT NULL, confidence DOUBLE PRECISION, model_version TEXT, threshold_set TEXT,
  decided_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS escalations (message_id TEXT PRIMARY KEY, reason TEXT NOT NULL, laya JSONB,
  resolved_route TEXT, rationale TEXT, resolved_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS feedback (message_id TEXT PRIMARY KEY, correct_route TEXT NOT NULL, source TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS alerts (id BIGSERIAL PRIMARY KEY, intent TEXT, volume INT, zscore DOUBLE PRECISION,
  examples JSONB, raised_at TIMESTAMPTZ NOT NULL DEFAULT now());

CREATE OR REPLACE VIEW v_coverage AS
SELECT decided_by, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM decisions GROUP BY decided_by;

CREATE OR REPLACE VIEW v_retrain_dataset AS
SELECT m.message_id, f.correct_route FROM feedback f JOIN messages m USING (message_id);
