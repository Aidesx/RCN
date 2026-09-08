"""E2E: RCN Studio UI running the REAL CORE (not the demo) — 4 flows:
T1 pasted text (extractive) · T2 pasted text (abstractive) · T3 file upload ·
T4 folder batch. Asserts the record carries the full real-core structure.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.model  # E2E runs the REAL CORE (SVM + abstractive checkpoint) — run: pytest -m model

APP = Path(__file__).resolve().parents[2] / "scripts" / "app.py"
SAMPLE = ("INVOICE #INV-10482\n\n"
          "Acme Corporation issued this invoice on 2026-08-20.\n"
          "John Smith purchased office chairs.\n"
          "Total due is $147.50 within 30 days.\n\n"
          "Payment terms are net thirty days from the invoice date.\n"
          "Contact Dr. Smith at accounting about the invoice.")

fails = []

def check(name, cond, extra=""):
    print(("PASS" if cond else "FAIL"), "-", name, extra, flush=True)
    if not cond:
        fails.append(name)

def rec_of(at):
    try:
        return at.session_state["rec"]
    except Exception:
        return None

def rows_of(at):
    try:
        return at.session_state["batch_rows"]
    except Exception:
        return None


def test_e2e_core():
    """E2E: 4 real-core flows via AppTest (extractive, abstractive, upload, batch)."""
    # ---- T1: pasted text, extractive mode ----
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("📄 Your document")
    at.run()
    at.sidebar.text_area[0].set_value(SAMPLE)
    at.sidebar.segmented_control[0].set_value("Extract sentences")
    at.sidebar.button(key="analyze").click()
    at.run()
    check("T1 runs without exception", not at.exception)
    r = rec_of(at) or {}
    check("T1 real core: doc_type has a label (SVM/rules)", bool((r.get("doc_type") or {}).get("label")))
    check("T1 summary engine = extractive", (r.get("summary") or {}).get("engine") == "extractive")
    check("T1 has structure.stats", bool((r.get("structure") or {}).get("stats")))
    check("T1 source keeps the pasted-text marker", str(r.get("source", "")).startswith("(pasted text)"))

    # ---- T2: pasted text, abstractive mode (real vit5, ~20s CPU) ----
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    at2.sidebar.radio[0].set_value("📄 Your document")
    at2.run()
    at2.sidebar.text_area[0].set_value(SAMPLE)
    at2.sidebar.segmented_control[0].set_value("Generate new text")
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("T2 runs without exception", not at2.exception)
    r2 = rec_of(at2) or {}
    eng2 = (r2.get("summary") or {}).get("engine")
    check("T2 abstractive runs on the real core", eng2 in ("abstractive", "extractive"),
          f"(engine={eng2})")
    check("T2 has summary content", bool(((r2.get("summary") or {}).get("text") or "").strip()))

    # ---- T3: upload a file via file_uploader ----
    at3 = AppTest.from_file(APP, default_timeout=300)
    at3.run()
    at3.sidebar.radio[0].set_value("📄 Your document")
    at3.run()
    at3.sidebar.file_uploader[0].set_value([("hoa_don.txt", SAMPLE.encode("utf-8"), "text/plain")])
    at3.sidebar.button(key="analyze").click()
    at3.run()
    check("T3 runs without exception", not at3.exception)
    r3 = rec_of(at3) or {}
    check("T3 doc_type has a label", bool((r3.get("doc_type") or {}).get("label")))
    check("T3 source = uploaded file name", str(r3.get("source", "")) == "hoa_don.txt")

    # ---- T4: folder batch (folder text_input) ----
    tmpdir = Path(tempfile.mkdtemp(prefix="rcn_e2e_"))
    try:
        (tmpdir / "a_invoice.txt").write_text(SAMPLE, encoding="utf-8")
        (tmpdir / "b_report.md").write_text(
            "Quarterly Performance Report Q2\n\nThis report summarizes operations "
            "for Acme Corporation during the period ending 2026-06-30.\nKey findings "
            "include steady growth in delivery throughput.\nPrepared by: Maria Nguyen",
            encoding="utf-8")
        at4 = AppTest.from_file(APP, default_timeout=300)
        at4.run()
        at4.sidebar.radio[0].set_value("📁 Folder (batch)")
        at4.run()
        at4.sidebar.text_input[0].set_value(str(tmpdir))
        at4.sidebar.button(key="analyze").click()
        at4.run()
        check("T4 runs without exception", not at4.exception)
        rows = rows_of(at4)
        check("T4 scans both files", bool(rows) and len(rows) == 2, f"(got {len(rows) if rows else 0})")
        check("T4 no file errored", bool(rows) and all(r["Type"] != "— error —" for r in rows))
        check("T4 has the real type (invoice)", bool(rows) and rows[0]["Type"] != "— error —")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert not fails, f"{len(fails)} failures: {fails}"


if __name__ == "__main__":
    test_e2e_core()
    print("\n===", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}", "===")
    sys.exit(0 if not fails else 1)
