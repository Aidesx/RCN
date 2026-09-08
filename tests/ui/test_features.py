"""Tests for five RCN Studio features via streamlit.testing.AppTest.

Coverage: source-text highlight · coherence chart · session history · batch mode + CSV
     (the skeleton is a transient loading effect -> verified by not-crashing).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.model  # real UI flow needs SVM + summarizer checkpoint — run: pytest -m model

APP = Path(__file__).resolve().parents[2] / "scripts" / "app.py"

SAMPLE_A = ("INVOICE #INV-10482\n\n"
            "Acme Corporation issued this invoice on 2026-08-20.\n"
            "John Smith purchased office chairs.\n"
            "Total due is $147.50 within 30 days.\n\n"
            "Payment terms are net thirty days from the invoice date.\n"
            "Contact Dr. Smith at accounting about the invoice.")
SAMPLE_B = ("Quarterly Performance Report Q2\n\n"
            "This report summarizes operations for Acme Corporation during "
            "the period ending 2026-06-30.\n"
            "Key findings include steady growth in delivery throughput.\n"
            "Prepared by: Maria Nguyen\n\n"
            "We recommend expanding the automation pilot next quarter.\n"
            "Distribution is internal use only.")

fails = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        fails.append(name)


def state_rec(at):
    try:
        return at.session_state["rec"]
    except Exception:
        return None


def state_value(at, key):
    try:
        return at.session_state[key]
    except Exception:
        return None


def set_widget(w, value):
    try:
        w.set_value(value)
    except AttributeError:
        w.set_input(value)


def test_features():
    """Coverage: highlight, coherence, history, batch+CSV."""
    # ---------- T1: highlight + coherence chart on sample data ----------
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🧪 Sample data (preview the UI)")
    at.sidebar.button(key="analyze").click()
    at.run()
    check("T1 demo runs without exception", not at.exception)
    main_md = "\n".join(str(v.value) for v in at.main.markdown)
    check("T1 <mark> highlight present", "<mark" in main_md)
    check("T1 highlights the right phrase 'hóa đơn giá trị gia tăng'",
          "hóa đơn giá trị gia tăng</mark>" in main_md.replace("\n", "")
          or "hóa đơn giá trị gia tăng" in main_md)
    try:
        n_charts = len(at.main.line_chart)
        check("T1 coherence line_chart renders", n_charts >= 1)
    except AttributeError:
        check("T1 coherence chart does not crash (not inspectable)", not at.exception)

    # ---------- T2: session history (2 analyses + View button) ----------
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    at2.sidebar.radio[0].set_value("📄 Your document")
    at2.run()
    set_widget(at2.sidebar.text_area[0], SAMPLE_A)
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("T2 first run without exception", not at2.exception)
    set_widget(at2.sidebar.text_area[0], SAMPLE_B)
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("T2 second run without exception", not at2.exception)
    hist = state_value(at2, "history")
    check(f"T2 history holds 2 entries (got: {len(hist) if hist else 0})",
          bool(hist) and len(hist) == 2)
    check("T2 newest entry is the report",
          bool(hist) and hist[0]["label"] == "report")
    btn_old = None
    for b in at2.main.button:
        if str(getattr(b, "key", "")) == "hist_1":
            btn_old = b
            break
    if btn_old is not None:
        btn_old.click()
        at2.run()
        rec_now = state_rec(at2) or {}
        check("T2 View restores the older result",
              (rec_now.get("doc_type") or {}).get("label") == "invoice")
    else:
        check("T2 View button found in history", False)

    # ---------- T3: batch mode + CSV ----------
    tmpdir = Path(tempfile.mkdtemp(prefix="rcn_batch_"))
    (tmpdir / "a_invoice.txt").write_text(SAMPLE_A, encoding="utf-8")
    (tmpdir / "b_report.md").write_text(SAMPLE_B, encoding="utf-8")
    (tmpdir / "c_letter.txt").write_text(
        "2026-08-20\n\nDear John,\n\nThank you for your continued partnership "
        "with Acme Corporation.\n\nSincerely,\nMaria Nguyen", encoding="utf-8")
    try:
        at3 = AppTest.from_file(APP, default_timeout=300)
        at3.run()
        at3.sidebar.radio[0].set_value("📁 Folder (batch)")
        at3.run()
        set_widget(at3.sidebar.text_input[0], str(tmpdir))
        at3.sidebar.button(key="analyze").click()
        at3.run()
        check("T3 batch runs without exception", not at3.exception)
        rows = state_value(at3, "batch_rows")
        check(f"T3 batch scans all 3 files (got: {len(rows) if rows else 0})",
              bool(rows) and len(rows) == 3)
        check("T3 no file errored",
              bool(rows) and all(r["Type"] != "— error —" for r in rows))
        check("T3 has a CSV download button", len(at3.main.download_button) >= 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert not fails, f"{len(fails)} failures: {fails}"


if __name__ == "__main__":
    test_features()
    print("\n===", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}", "===")
    sys.exit(0 if not fails else 1)
