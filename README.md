# NEET College Predictor

Rule-based NEET All India College Predictor. Shows **all eligible colleges** for a given rank, category, and quota.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run CLI
python scripts/predict.py --rank 10000 --category General --quota "All India"

# Run Web UI
streamlit run app/streamlit_app.py
```

## How It Works

**Chance Logic:**
- 🟢 **High** — Rank ≤ closing rank (strong admission chance)
- 🟡 **Good** — Rank ≤ closing rank × 1.07 (possible, 7% margin)
- 🔴 **Low** — Rank > closing rank × 1.07 (borderline)

## Data Collection

Data was collected from **official MCC counselling notifications** and state counselling portals across India.

### Sources
- **MCC (Medical Counselling Committee)** — All India Quota counselling data (2019-2025)
- **State Portals** — Gujarat, Karnataka state counselling data
- **Allotment Lists** — Phase 1, 2, 3 closing ranks from official PDF notifications

### Pipeline
1. Crawled MCC archive pages to discover PDF links (456 PDFs)
2. Downloaded and extracted tables from PDFs using pdfplumber/tabula
3. Normalized college names, categories, quotas, and state mappings
4. Merged with structured allotment data from counselling portals
5. Built rule-based predictor using closing rank patterns

### Data Coverage

| Metric | Count |
|---|---|
| Institutes | 599 |
| States/UTs | 33 |
| Quotas | 28 |
| Categories | 11 |
| Courses | 3 (MBBS, BDS, B.Sc. Nursing) |
| Records | 3,910 |

## Project Structure

```
neet-pipeline/
├── app/
│   └── streamlit_app.py          # Web UI
├── scripts/
│   ├── predict.py                # CLI predictor
│   ├── merge_brilliantpala.py    # Data merge script
│   ├── prepare_training_data.py  # Training data prep
│   ├── train_model.py            # Model training
│   └── download_state_portals.py # State portal downloader
├── data/
│   ├── raw/
│   │   └── allotment_data.csv    # Source data (3,910 records)
│   └── cleaned/
│       └── merged_dataset_v2.csv # Merged dataset
├── config/
│   ├── settings.yaml
│   ├── sources.yaml
│   └── state_sources.yaml
├── src/
│   ├── coordinator.py            # Pipeline orchestrator
│   ├── agents/
│   │   ├── link_discovery.py     # MCC link crawler
│   │   ├── downloader.py         # PDF downloader
│   │   ├── pdf_extractor.py      # PDF table extraction
│   │   ├── ocr_agent.py          # OCR fallback
│   │   ├── normalizer.py         # Data normalizer
│   │   ├── validator.py          # Data validation
│   │   ├── db_ingestion.py       # PostgreSQL ingestion
│   │   ├── qa_agent.py           # ML dataset builder
│   │   └── state_link_discovery.py # State portal crawler
│   ├── models/
│   │   ├── schemas.py            # Pydantic models
│   │   └── db_models.py          # SQLAlchemy ORM
│   ├── parsers/
│   │   ├── mcc_parser.py
│   │   └── table_detector.py
│   └── utils/
│       ├── http_client.py
│       ├── checkpoint.py
│       ├── rate_limiter.py
│       ├── checksum.py
│       └── text_clean.py
├── docker-compose.yml            # PostgreSQL
├── requirements.txt
└── README.md
```

## CLI Usage

```bash
# Basic prediction
python scripts/predict.py --rank 50000 --category General --quota "All India"

# With state filter
python scripts/predict.py --rank 10000 --category OBC --quota "All India" --state Maharashtra

# With course filter
python scripts/predict.py --rank 5000 --category SC --course MBBS

# With phase filter
python scripts/predict.py --rank 20000 --category ST --phase 3

# Show top 20 only
python scripts/predict.py --rank 10000 --category General --top 20
```

## Web UI Features

- **Sidebar:** NEET AIR, Category, Quota (multi-select), State, Course (multi-select)
- **Results:** Color-coded table with Institute, Chance, State, Quota, Course, Allotted Category, Phase, Closing Rank
- **Refine Results:** Secondary filters for State (region-grouped), Course, Allotted Category, Quota, Phase, Chance

## API

```python
from scripts.predict import predict

results = predict(
    rank=10000,
    category="General",
    quota="All India",
    state="Maharashtra",
    course="MBBS",
)
# Returns DataFrame with columns: S.No, Institute, Chance, State, Quota, Course, Allotted Category, Phase, Closing Rank
```

## License

Data collected from official MCC counselling notifications and state counselling portals.
