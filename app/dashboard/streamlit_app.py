"""
Streamlit Dashboard — Evidence-to-Journal Matching.

Clean 3-step workflow:
  Step 1: Upload XLSX(s) → preview → append to vector store
  Step 2: Upload PDFs → extract → match
  Step 3: View results per invoice with full explainability

Run with: streamlit run app/dashboard/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import streamlit as st
import pandas as pd
from io import BytesIO

from app.config import get_config, VectorStoreConfig, VECTORDB_DIR
from app.cost_logger import cost_logger
from app.models import MatchResult
from app.parsers.xlsx_parser import parse_and_group
from app.parsers.pdf_parser import extract_and_parse
from app.indexer.embedder import get_embedder
from app.indexer.vector_store import VectorStore
from app.matcher.pipeline import MatchingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Page Config & Styles
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Audit Evidence Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 6px solid;
    }
    .result-card.matched {
        background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(16,185,129,0.02));
        border-left-color: #10b981;
    }
    .result-card.no-match {
        background: linear-gradient(135deg, rgba(239,68,68,0.06), rgba(239,68,68,0.02));
        border-left-color: #ef4444;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-match { background: rgba(16,185,129,0.15); color: #059669; }
    .badge-no    { background: rgba(239,68,68,0.15);  color: #dc2626; }

    .invoice-header {
        background: rgba(99,102,241,0.08);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6366f1;
    }

    .ingested-file {
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        background: rgba(16,185,129,0.08);
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

for key, default in {
    "journal_groups": [],
    "journal_df": None,
    "ingested_files": [],
    "ingested_group_count": 0,
    "invoices": [],
    "match_results": [],
    "matching_done": False,
    "store": None,
    "pipeline": None,
    "settings_snapshot": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")

        st.caption(
            "**Models:** OpenAI text-embedding-3-small + gpt-4o-mini  \n"
            "All costs tracked against $5 budget."
        )

        use_llm = st.toggle("Use LLM for explanations", value=True,
                             help="Off = deterministic scoring only (faster, cheaper)")

        # ── Cost Tracker ──
        st.divider()
        st.header("💰 Cost Tracker")

        if cost_logger.total_calls() > 0:
            for stage, info in cost_logger.summary_by_stage().items():
                tokens = info['input_tokens'] + info['output_tokens']
                cost = info['openai_equivalent_usd']
                st.markdown(
                    f"**{stage}** ☁️  \n"
                    f"`{info['num_calls']}` calls · `{tokens:,}` tokens · "
                    f"`${cost:.6f}`"
                )

            st.divider()
            total = cost_logger.total_openai_equivalent()
            st.metric("Total Cost", f"${total:.6f}",
                      delta=f"${5.0 - total:.4f} remaining", delta_color="normal")

            if st.button("💾 Export cost log"):
                cost_logger.save_to_file()
                st.success("Saved to `logs/cost_log.json`")
        else:
            st.caption("No API calls yet.")

        # ── Indexed files status ──
        if st.session_state.ingested_files:
            st.divider()
            st.header("📂 Indexed Files")
            st.caption(f"{st.session_state.ingested_group_count} groups from {len(st.session_state.ingested_files)} file(s)")
            for fname in st.session_state.ingested_files:
                st.markdown(f'<div class="ingested-file">✅ {fname}</div>', unsafe_allow_html=True)

        # ── Reset ──
        st.divider()
        if st.button("🗑️ Reset everything"):
            if st.session_state.store:
                st.session_state.store.reset_all()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            cost_logger.clear()
            st.rerun()

    return {"use_llm": use_llm}


# ──────────────────────────────────────────────
# Pipeline Factory
# ──────────────────────────────────────────────

def build_pipeline(settings: dict) -> tuple[VectorStore, MatchingPipeline]:
    snapshot = json.dumps(settings, sort_keys=True)
    if st.session_state.pipeline and st.session_state.settings_snapshot == snapshot:
        return st.session_state.store, st.session_state.pipeline

    config = get_config(use_llm_judge=settings["use_llm"])

    embedder = get_embedder(config.embedding)
    store = VectorStore(VectorStoreConfig(persist_dir=str(VECTORDB_DIR)), embedder)
    pipeline = MatchingPipeline(config, store)

    st.session_state.store = store
    st.session_state.pipeline = pipeline
    st.session_state.settings_snapshot = snapshot
    return store, pipeline


# ──────────────────────────────────────────────
# Step 1 — Journal Upload (always multi-file, append mode)
# ──────────────────────────────────────────────

def render_step1(settings: dict):
    st.markdown("## 📊 Step 1 — Upload Journal Entries")
    st.caption("Upload one or more XLSX files. Each file is parsed, previewed, and **appended** to the vector store. Upload additional files at any time.")

    uploaded = st.file_uploader(
        "Choose XLSX file(s)",
        type=["xlsx"],
        accept_multiple_files=True,
        key="xlsx_up",
    )

    if not uploaded:
        if st.session_state.ingested_files:
            st.info(
                f"✅ **{st.session_state.ingested_group_count} groups** indexed "
                f"from {len(st.session_state.ingested_files)} file(s). "
                f"Upload more files or proceed to Step 2."
            )
        return

    # ── Classify files: new vs re-upload ──
    new_files = []
    reupload_files = []
    for f in uploaded:
        if f.name in st.session_state.ingested_files:
            reupload_files.append(f)
        else:
            new_files.append(f)

    # ── Parse ALL files (new + re-uploads) ──
    all_files = new_files + reupload_files
    try:
        all_groups = []
        all_dfs = []

        for file in all_files:
            buf = BytesIO(file.read())
            df, groups = parse_and_group(buf)
            all_dfs.append(df)
            all_groups.extend(groups)

        combined_df = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]
    except Exception as e:
        st.error(f"Could not parse file(s): {e}")
        return

    # ── Preview ──
    all_names = ", ".join(f.name for f in all_files)
    st.success(f"Parsed **{len(combined_df)} rows** → **{len(all_groups)} groups** from: {all_names}")

    if reupload_files:
        reup_names = ", ".join(f"`{f.name}`" for f in reupload_files)
        st.info(
            f"📄 **Re-upload detected:** {reup_names} was previously indexed. "
            f"Data from this file will be **updated** in the vector store (overwrite)."
        )

    summary = pd.DataFrame([
        {
            "Group": g.journal_ref,
            "Vendor": g.vendor_name if g.vendor_name != "—" else "—",
            "Invoice #": g.invoice_no or "—",
            "Date": g.invoice_date or "—",
            "Total": f"{g.total_amount:,.2f}" if g.total_amount else "—",
            "Lines": g.line_count,
        }
        for g in all_groups
    ])
    st.dataframe(summary, width='stretch', hide_index=True)

    with st.expander("View raw data"):
        st.dataframe(combined_df, width='stretch', hide_index=True)

    # ── Ingest button ──
    button_label = "Update Vector Store" if reupload_files and not new_files else "Add to Vector Store"
    if reupload_files and new_files:
        button_label = "Add & Update Vector Store"

    if st.button(button_label, type="primary", icon="🔄"):
        with st.spinner("Generating embeddings and indexing..."):
            store, _ = build_pipeline(settings)
            n = store.ingest_journal_groups(all_groups)

        for f in all_files:
            if f.name not in st.session_state.ingested_files:
                st.session_state.ingested_files.append(f.name)

        st.session_state.journal_groups = all_groups
        st.session_state.ingested_group_count = store.get_journal_count()
        st.session_state.matching_done = False
        st.session_state.match_results = []

        if reupload_files:
            reup_names = ", ".join(f.name for f in reupload_files)
            st.success(f"✅ Updated data from **{reup_names}**. Total in store: **{st.session_state.ingested_group_count} groups**")
        else:
            st.success(f"✅ Added **{n} groups**. Total in store: **{st.session_state.ingested_group_count}**")
        st.rerun()

    # ── Show current status ──
    if st.session_state.ingested_files:
        total = st.session_state.ingested_group_count
        count = len(st.session_state.ingested_files)
        st.info(f"✅ **{total} groups** indexed from {count} file(s). Upload more or proceed to Step 2.")


# ──────────────────────────────────────────────
# Step 2 — Evidence Upload & Matching
# ──────────────────────────────────────────────

def render_step2(settings: dict):
    st.markdown("## 📄 Step 2 — Upload Evidence Invoices")

    if not st.session_state.ingested_files:
        st.warning("Complete Step 1 first — upload and index journal entries.")
        return

    st.caption("Upload one or more PDF invoices. Fields are extracted automatically and matched against the indexed journal groups.")

    uploaded_pdfs = st.file_uploader(
        "Choose PDF invoice(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_up",
    )

    if not uploaded_pdfs:
        return

    # Detect if uploaded PDFs changed
    current_pdf_key = "|".join(sorted(f.name + str(f.size) for f in uploaded_pdfs))
    if st.session_state.get("_last_pdf_key") != current_pdf_key:
        st.session_state.matching_done = False
        st.session_state.match_results = []
        st.session_state._last_pdf_key = current_pdf_key

    # ── Parse invoices ──
    invoices = []
    for pdf in uploaded_pdfs:
        try:
            buf = BytesIO(pdf.read())
            inv = extract_and_parse(buf, source_file=pdf.name)
            invoices.append(inv)
        except Exception as e:
            st.error(f"Error reading **{pdf.name}**: {e}")

    if not invoices:
        return

    st.session_state.invoices = invoices
    st.success(f"Extracted fields from **{len(invoices)} invoice(s)**")

    # ── Extracted fields table ──
    inv_df = pd.DataFrame([
        {
            "File": inv.source_file,
            "Vendor": inv.vendor_name,
            "Invoice #": inv.invoice_no,
            "Date": inv.invoice_date,
            "Subtotal": f"{inv.subtotal:,.2f}" if inv.subtotal else "—",
            "VAT": f"{inv.vat_amount:,.2f}" if inv.vat_amount else "—",
            "Total": f"{inv.total_amount:,.2f}" if inv.total_amount else "—",
            "VAT ID": inv.vat_id or "—",
        }
        for inv in invoices
    ])
    st.dataframe(inv_df, width='stretch', hide_index=True)

    # ── Raw OCR text (Req 4.3) ──
    with st.expander("View raw extracted text"):
        for inv in invoices:
            st.markdown(f"**{inv.source_file}**")
            st.code(inv.raw_text, language=None)

    # ── Duplicate invoice detection (audit control) ──
    seen = {}
    unique_invoices = []
    duplicates = []
    for inv in invoices:
        key = inv.invoice_no
        if key in seen:
            duplicates.append((inv.source_file, seen[key], key))
        else:
            seen[key] = inv.source_file
            unique_invoices.append(inv)

    if duplicates:
        st.warning(
            f"⚠️ **Duplicate invoice(s) detected** — "
            f"in real audit, this could indicate a double-payment risk."
        )
        for dup_file, orig_file, inv_no in duplicates:
            st.markdown(
                f"- **{inv_no}**: `{dup_file}` is a duplicate of `{orig_file}` — skipped"
            )

    # ── Match ──
    if st.button("Run Matching", type="primary", icon="🎯"):
        store, pipeline = build_pipeline(settings)
        n_stored, store_warnings = store.ingest_invoices(unique_invoices)

        for w in store_warnings:
            st.warning(f"⚠️ **{w['type']}**: {w['message']}")

        with st.spinner("Running matching pipeline..."):
            results = pipeline.match_invoices(unique_invoices)

        # Add DUPLICATE result for each duplicate
        for dup_file, orig_file, inv_no in duplicates:
            dup_inv = next(inv for inv in invoices if inv.source_file == dup_file)
            from app.models import MatchResult
            results.append(MatchResult(
                invoice=dup_inv,
                outcome="DUPLICATE",
                confidence=1.0,
                explanation=(
                    f"Duplicate invoice detected: {dup_file} has the same invoice "
                    f"number ({inv_no}) as {orig_file}. In audit, this is flagged as "
                    f"a potential double-payment risk. The original was matched separately."
                ),
            ))

        st.session_state.match_results = results
        st.session_state.matching_done = True
        st.rerun()


# ──────────────────────────────────────────────
# Step 3 — Results (per invoice)
# ──────────────────────────────────────────────

def render_evidence_table(evidence: dict):
    fields = {
        "invoice_no_match": "📝 Invoice Number",
        "vendor_match":     "🏢 Vendor Name",
        "amount_match":     "💰 Amount",
        "date_match":       "📅 Date",
        "vat_id_match":     "🆔 VAT ID",
    }
    rows = []
    for key, label in fields.items():
        ev = evidence.get(key, {})
        if not ev:
            continue
        matched = ev.get("matched", False)
        icon = "✅" if matched else "❌"
        inv_val = str(ev.get("invoice_value", "—"))
        jnl_val = str(ev.get("journal_value", "—"))

        detail = ""
        if ev.get("similarity") is not None:
            detail = f"{ev['similarity']:.0%} match"
        elif ev.get("difference_pct") is not None:
            detail = f"{ev['difference_pct']:.1f}% diff"
        elif ev.get("days_diff") is not None:
            detail = f"{ev['days_diff']}d apart"

        rows.append({"": f"{icon} {label}", "Invoice": inv_val, "Journal": jnl_val, "Detail": detail})

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_result_card(result: MatchResult, invoice_index: int, total_invoices: int):
    inv = result.invoice
    is_matched = result.outcome == "MATCHED"
    is_duplicate = result.outcome == "DUPLICATE"

    if is_matched:
        css_class = "matched"
        badge = '<span class="badge badge-match">MATCHED</span>'
    elif is_duplicate:
        css_class = "no-match"
        badge = '<span class="badge badge-no">DUPLICATE</span>'
    else:
        css_class = "no-match"
        badge = '<span class="badge badge-no">NO MATCH</span>'

    conf_pct = int(result.confidence * 100)

    st.markdown(f"""
    <div class="invoice-header">
        <strong>Invoice {invoice_index} of {total_invoices}</strong> —
        <code>{inv.source_file}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <h3 style="margin:0;">{inv.invoice_no} — {inv.vendor_name}</h3>
            {badge}
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        if is_matched:
            st.markdown(f"**Matched to:** `{result.best_match}`")
        if is_duplicate:
            st.error("⚠️ **Duplicate invoice** — risk of double payment. Auditor should review.", icon="🚨")
        st.markdown(f"**Confidence:** {conf_pct}%")
        st.progress(result.confidence)

        st.markdown("##### Invoice Fields")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Vendor:** {inv.vendor_name}")
        c2.markdown(f"**Invoice #:** {inv.invoice_no}")
        c1.markdown(f"**Date:** {inv.invoice_date}")
        c2.markdown(f"**Total:** {inv.total_amount:,.2f} {inv.currency}" if inv.total_amount else "**Total:** —")
        c1.markdown(f"**VAT:** {inv.vat_amount:,.2f}" if inv.vat_amount else "**VAT:** —")
        c2.markdown(f"**VAT ID:** {inv.vat_id}" if inv.vat_id else "**VAT ID:** —")

    with right:
        st.markdown("##### Why this decision?")
        st.info(result.explanation, icon="💡")

        if result.evidence:
            st.markdown("##### Field-by-Field Evidence")
            render_evidence_table(result.evidence)

    with st.expander(f"View top {len(result.top_candidates)} candidates"):
        for i, c in enumerate(result.top_candidates):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"**#{i+1}**"

            cols = st.columns([0.8, 2.5, 1, 5])
            cols[0].markdown(medal)
            cols[1].code(c.journal_ref, language=None)
            cols[2].markdown(f"**{c.combined_score:.3f}**")
            cols[3].caption(c.rationale)

            fs = c.evidence.get("field_scores", {})
            if fs:
                score_cols = st.columns(5)
                for j, field in enumerate(["invoice_no", "vendor_name", "amount", "date", "vat_id"]):
                    val = fs.get(field, 0)
                    score_cols[j].progress(val, text=f"{field} {val:.2f}")

            if i < len(result.top_candidates) - 1:
                st.divider()


def render_step3():
    st.markdown("## 🎯 Step 3 — Matching Results")

    if not st.session_state.matching_done:
        st.info("Upload invoices and run matching in Step 2 to see results here.")
        return

    results = st.session_state.match_results
    if not results:
        st.warning("No results available.")
        return

    matched_count = sum(1 for r in results if r.outcome == "MATCHED")
    dup_count = sum(1 for r in results if r.outcome == "DUPLICATE")
    no_match_count = len(results) - matched_count - dup_count

    if dup_count > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invoices Processed", len(results))
        c2.metric("Matched", matched_count)
        c3.metric("No Match", no_match_count)
        c4.metric("Duplicates", dup_count, delta="audit flag", delta_color="inverse")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Invoices Processed", len(results))
        c2.metric("Matched", matched_count)
        c3.metric("No Match", no_match_count)

    st.divider()

    for i, result in enumerate(results):
        render_result_card(result, invoice_index=i + 1, total_invoices=len(results))
        if i < len(results) - 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

    st.divider()
    export_data = json.dumps([r.to_dict() for r in results], indent=2, default=str)
    st.download_button(
        "📥 Download results as JSON",
        data=export_data,
        file_name="match_results.json",
        mime="application/json",
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    st.title("🔍 Evidence-to-Journal Matching")
    st.caption("Match invoices to journal entries with explainable AI-powered evidence analysis.")

    settings = render_sidebar()

    render_step1(settings)
    st.divider()
    render_step2(settings)
    st.divider()
    render_step3()


if __name__ == "__main__":
    main()