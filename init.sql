-- init.sql
-- ─────────────────────────────────────────────────────────────
-- Runs automatically the FIRST time the MySQL container starts
-- (via Docker's docker-entrypoint-initdb.d mechanism).
-- Creates the table that logs every /predict call.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    request_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Input fields (what the user sent)
    gene VARCHAR(255) NOT NULL,
    variation VARCHAR(255) NOT NULL,
    clinical_text TEXT NOT NULL,

    -- Output fields (what the model predicted)
    predicted_class INT NOT NULL,
    predicted_class_name VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL,

    -- Metadata — useful once you have multiple model versions
    model_name VARCHAR(255) DEFAULT 'cancer-classifier',
    model_stage VARCHAR(50) DEFAULT 'Production',

    INDEX idx_timestamp (request_timestamp)
);