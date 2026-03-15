# Evidence-to-Journal Matching (Audit Rollup Mini)

An end-to-end prototype that ingests journal entries (XLSX) and evidence invoices (PDF), stores them in a local vector store, and matches invoices to the correct journal entry group(s) with explainable output.

## Quick Start (One Command)

After cloning the repo:

```bash
bash run_demo.sh
```
Also check the run_demo.sh file to be aware about steps and change conda environment's name.

This single command creates the conda environment, installs all dependencies, configures the project, and launches the Streamlit dashboard. The app opens at **http://localhost:8501**.

## Manual Setup (Alternative)

```bash
conda create -n osama-env python=3.11 -y
conda activate osama-env
pip install -r requirements.txt
.env   # edit with your OpenAI API key
streamlit run app/dashboard/streamlit_app.py
```

## Usage

1. **Upload XLSX** — Upload `sample_journal_entries.xlsx` (or `_v2_noisy.xlsx`), preview groups, click "Add to Vector Store"
2. **Upload PDFs** — Upload invoice PDFs, review extracted fields, click "Run Matching"
3. **View Results** — See per-invoice outcome (Matched/No Match), confidence, explanation, field evidence, and top-3 candidates

## Provided Test Data

| Invoice | Expected Outcome |
|---------|-----------------|
| `invoice_INV-ACME-1001.pdf` | MATCHED |
| `invoice_NIMBUS-778.pdf` | MATCHED |
| `invoice_ORION-404.pdf` | NO MATCH |

## Matching Pipeline

```
Invoice PDF
    |
    v
[1] Vector Search (text-embedding-3-small + ChromaDB)
    Top-10 candidates by cosine similarity
    |
    v
[2] Deterministic Reranking (field-by-field scoring)
    invoice_no (0.35) + vendor (0.25) + amount (0.20) + date (0.10) + VAT (0.10)
    Candidates ranked by deterministic score
    |
    v
[3] LLM Decision (gpt-4o-mini, 1 call per invoice)
    Structured JSON: decision + confidence + explanation
    |
    v
Result: MATCHED / NO MATCH + evidence
```

## Project Structure

```
audit-evidence-matcher/
├── run_demo.sh                # One-click setup & launch
├── app/
│   ├── config.py              # Configuration (OpenAI keys, thresholds)
│   ├── cost_logger.py         # API cost tracking
│   ├── models.py              # Data classes
│   ├── parsers/
│   │   ├── xlsx_parser.py     # XLSX -> grouped JournalGroups
│   │   └── pdf_parser.py      # PDF -> structured Invoice objects
│   ├── indexer/
│   │   ├── embedder.py        # OpenAI embedding wrapper
│   │   └── vector_store.py    # ChromaDB + duplicate detection
│   ├── matcher/
│   │   ├── deterministic.py   # Field-level scoring (5 fields)
│   │   ├── reranker.py        # Passthrough (reranking done in deterministic)
│   │   ├── llm_judge.py       # gpt-4o-mini decision + explanation
│   │   └── pipeline.py        # 3-stage orchestrator
│   └── dashboard/
│       └── streamlit_app.py   # Streamlit UI
├── docs/
│   └── technical_writeup.pdf  # Architecture & design write-up
├── requirements.txt
├── .env.example
└── README.md
```

## Cost

| Stage | Model | 3 Invoices | Per Invoice |
|-------|-------|-----------|------------|
| Embedding | text-embedding-3-small | $0.000024 | $0.000008 |
| LLM Judge | gpt-4o-mini | $0.000837 | $0.000279 |
| **Total** | | **$0.000861** | **$0.000287** |

Budget: $5.00 — supports ~17,000 invoices.
