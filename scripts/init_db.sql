CREATE TABLE IF NOT EXISTS colleges (
    college_id    SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    state         TEXT NOT NULL,
    ownership     TEXT,
    type          TEXT,
    aliases       TEXT[],
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, state)
);

CREATE TABLE IF NOT EXISTS cutoffs (
    cutoff_id     SERIAL PRIMARY KEY,
    college_id    INT REFERENCES colleges(college_id),
    year          INT NOT NULL,
    round         INT NOT NULL,
    quota         TEXT NOT NULL,
    category      TEXT NOT NULL,
    course        TEXT NOT NULL,
    opening_rank  INT,
    closing_rank  INT,
    seat_type     TEXT,
    source_file   TEXT,
    confidence    FLOAT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS seat_matrix (
    seat_id       SERIAL PRIMARY KEY,
    college_id    INT REFERENCES colleges(college_id),
    year          INT NOT NULL,
    course        TEXT NOT NULL,
    category      TEXT NOT NULL,
    seat_count    INT NOT NULL,
    quota         TEXT,
    source_file   TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cutoffs_college_year ON cutoffs(college_id, year);
CREATE INDEX IF NOT EXISTS idx_cutoffs_closing_rank ON cutoffs(closing_rank);
CREATE INDEX IF NOT EXISTS idx_cutoffs_quota ON cutoffs(quota);
CREATE INDEX IF NOT EXISTS idx_cutoffs_category ON cutoffs(category);
CREATE INDEX IF NOT EXISTS idx_seat_matrix_college_year ON seat_matrix(college_id, year);
CREATE INDEX IF NOT EXISTS idx_colleges_state ON colleges(state);
