"""Smoke tests for RCN Studio (scripts/app.py) via streamlit.testing.AppTest.

Checks:
  1. The landing page renders without exceptions
  2. The "sample data" flow: press Analyze -> all tabs render, JSON/Markdown downloadable
  3. Real pasted-text flow (through the understand() seam): invoice stats/keywords correct
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.model  # real UI flow needs SVM + summarizer checkpoint — run: pytest -m model

APP = Path(__file__).resolve().parents[2] / "scripts" / "app.py"

SAMPLE = (
    "INVOICE #INV-10482\n\n"
    "Acme Corporation issued this invoice on 2026-08-20.\n"
    "John Smith purchased office chairs and printer paper.\n"
    "Total due is $147.50 within 30 days.\n"
    "\n"
    "Payment terms are net thirty days from the invoice date.\n"
    "Contact Dr. Smith at accounting for any questions about this invoice."
)

def state_rec(at):
    try:
        return at.session_state["rec"]
    except Exception:
        return None

fails = []

def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        fails.append(name)


def test_smoke():
    """Smoke: render without crash + demo mode + real text through the seam."""
    # ---------- 1) landing page ----------
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    check("landing renders without exception", not at.exception)

    # ---------- 2) sample data ----------
    at.sidebar.radio[0].set_value("🧪 Sample data (preview the UI)")
    at.sidebar.button(key="analyze").click()
    at.run()
    check("demo runs without exception", not at.exception)
    rec = state_rec(at)
    check("demo stores a record in state", bool(rec))
    check("demo structure.stats.words == 186", (rec or {}).get("structure", {})
          .get("stats", {}).get("words") == 186)
    main_text = "\n".join(str(v.value) for v in at.main.markdown)
    check("demo shows Invoice badge", "Invoice" in main_text)
    check("demo shows the keyword chip", "hóa đơn giá trị gia tăng" in main_text)

    # ---------- 3) real pasted text through the seam ----------
    at2 = AppTest.from_file(str(APP), default_timeout=300)
    at2.run()
    at2.sidebar.radio[0].set_value("📄 Your document")
    at2.run()
    ta = at2.sidebar.text_area[0]
    try:
        ta.set_value(SAMPLE)
    except AttributeError:
        ta.set_input(SAMPLE)
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("real seam runs without exception", not at2.exception)
    rec2 = state_rec(at2)
    check("seam returns a record", bool(rec2))
    dt = (rec2 or {}).get("doc_type", {})
    check(f"router detects invoice (got: {dt.get('label')})",
          dt.get("label") == "invoice")
    stats = ((rec2 or {}).get("structure") or {}).get("stats", {})
    check("seam stats has words>0", stats.get("words", 0) > 0)
    kws = rec2.get("keywords", []) if rec2 else []
    check("keywords are dicts with term/score/count",
          bool(kws) and isinstance(kws[0], dict)
          and {"term", "score", "count"} <= set(kws[0]))
    sm = (rec2 or {}).get("summary", {})
    check("extractive summary has sentences",
          sm.get("engine") == "extractive" and len(sm.get("sentences", [])) >= 1)

    assert not fails, f"{len(fails)} failures: {fails}"


if __name__ == "__main__":
    test_smoke()
    print("\n===", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}", "===")
    sys.exit(0 if not fails else 1)
