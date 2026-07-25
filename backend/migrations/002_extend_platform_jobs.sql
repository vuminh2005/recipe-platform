-- Run this once in the Neon SQL Editor connected to the Recipe Platform DB.
-- SQLAlchemy create_all() does not add columns to an existing table.

ALTER TABLE platform_jobs
    ADD COLUMN IF NOT EXISTS best_params JSONB,
    ADD COLUMN IF NOT EXISTS best_metric DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mlflow_final_run_id VARCHAR(128),
    ADD COLUMN IF NOT EXISTS model_uri VARCHAR(500),
    ADD COLUMN IF NOT EXISTS registered_model_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS registered_model_version VARCHAR(64),
    ADD COLUMN IF NOT EXISTS final_metrics JSONB;

CREATE INDEX IF NOT EXISTS idx_platform_jobs_status_created_at
    ON platform_jobs (status, created_at);

-- Inspect the status column. If data_type is character varying/text, no enum
-- migration is needed. If udt_name points to a PostgreSQL enum, add the new
-- values using the generated ALTER TYPE statements shown by the second query.
SELECT data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'platform_jobs'
  AND column_name = 'status';

SELECT format(
    'ALTER TYPE %I ADD VALUE IF NOT EXISTS %L;',
    c.udt_name,
    value_to_add
) AS statement_to_run
FROM information_schema.columns AS c
CROSS JOIN (
    VALUES
        ('PENDING'),
        ('CLAIMED'),
        ('TUNING'),
        ('TRAINING'),
        ('REGISTERING'),
        ('SUCCEEDED'),
        ('FAILED')
) AS values_list(value_to_add)
WHERE c.table_schema = 'public'
  AND c.table_name = 'platform_jobs'
  AND c.column_name = 'status'
  AND c.data_type = 'USER-DEFINED';
