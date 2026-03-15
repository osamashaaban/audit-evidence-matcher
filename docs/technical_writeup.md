# Technical Write-Up: Evidence-to-Journal Matching

## 1. Architecture

3-step Streamlit dashboard backed by a 3-stage matching pipeline using OpenAI APIs.

**Pipeline:** Vector Search -> Deterministic Reranking -> LLM Decision

## 2. Chunking & Indexing

- XLSX grouped by journal_ref, one embedding document per group
- Metadata stored alongside: journal_ref, vendor, invoice_no, total, date, VAT ID
- PDFs parsed with PyMuPDF, fields normalized for noisy data

## 3. Matching Strategy

**Stage 1:** Vector search (text-embedding-3-small + ChromaDB, top-10)

**Stage 2:** Deterministic reranking:
```
det_score = 0.35 x invoice_no + 0.25 x vendor + 0.20 x amount + 0.10 x date + 0.10 x vat_id
ranking = det_score only (vector similarity used for retrieval, not ranking)
threshold = 0.60
```

Gating rule: mismatched invoice numbers -> score capped at 0.25

**Stage 3:** LLM decision (gpt-4o-mini, 1 call per invoice, structured JSON)

Deterministic decides, LLM explains. If they disagree, deterministic wins.

## 4. Results (3 provided invoices)

| Invoice | v1 Clean | v2 Noisy |
|---------|----------|----------|
| INV-ACME-1001 | MATCHED (0.805) | MATCHED (0.861) |
| NIMBUS-778 | MATCHED (0.892) | MATCHED (0.932) |
| ORION-404 | NO MATCH (0.354) | NO MATCH (0.559) |

6/6 correct.

## 5. Cost Controls

3 invoices: ~$0.0009 total. Budget $5 supports ~17,000 invoices.
