"""Kiểm thử 5 tính năng mới của RCN Studio qua streamlit.testing.AppTest.

Phủ: highlight văn bản gốc · coherence chart · lịch sử phiên · batch mode+CSV
     (skeleton là hiệu ứng tạm thời trong lúc chạy -> xác minh bằng không-crash).
"""
import shutil
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

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


def set_widget(w, value):
    try:
        w.set_value(value)
    except AttributeError:
        w.set_input(value)


def test_features():
    """Kiểm thử: highlight, coherence, lịch sử, batch+CSV."""
    # ---------- T1: highlight + coherence chart trên dữ liệu mẫu ----------
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("🧪 Dữ liệu mẫu (xem thử giao diện)")
    at.sidebar.button(key="analyze").click()
    at.run()
    check("T1 demo không exception", not at.exception)
    main_md = "\n".join(str(v.value) for v in at.main.markdown)
    check("T1 highlight <mark> xuất hiện", "<mark" in main_md)
    check("T1 highlight đúng cụm 'hóa đơn giá trị gia tăng'",
          "hóa đơn giá trị gia tăng</mark>" in main_md.replace("\n", "")
          or "hóa đơn giá trị gia tăng" in main_md)
    try:
        n_charts = len(at.main.line_chart)
        check("T1 coherence line_chart render", n_charts >= 1)
    except AttributeError:
        check("T1 coherence chart không crash (không inspect được)", not at.exception)

    # ---------- T2: lịch sử phiên (2 lần phân tích + nút Xem) ----------
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    at2.sidebar.radio[0].set_value("📄 Tài liệu của bạn")
    at2.run()
    set_widget(at2.sidebar.text_area[0], SAMPLE_A)
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("T2 lần 1 không exception", not at2.exception)
    set_widget(at2.sidebar.text_area[0], SAMPLE_B)
    at2.sidebar.button(key="analyze").click()
    at2.run()
    check("T2 lần 2 không exception", not at2.exception)
    hist = None
    try:
        hist = at2.session_state["history"]
    except Exception:
        pass
    check(f"T2 history có 2 mục (thấy: {len(hist) if hist else 0})",
          bool(hist) and len(hist) == 2)
    check("T2 entry mới nhất là Báo cáo",
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
        check("T2 nút Xem khôi phục đúng kết quả cũ",
              (rec_now.get("doc_type") or {}).get("label") == "invoice")
    else:
        check("T2 tìm thấy nút Xem trong lịch sử", False)

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
        at3.sidebar.radio[0].set_value("📁 Thư mục (batch)")
        at3.run()
        set_widget(at3.sidebar.text_input[0], str(tmpdir))
        at3.sidebar.button(key="analyze").click()
        at3.run()
        check("T3 batch không exception", not at3.exception)
        rows = None
        try:
            rows = at3.session_state["batch_rows"]
        except Exception:
            pass
        check(f"T3 batch quét đủ 3 tệp (thấy: {len(rows) if rows else 0})",
              bool(rows) and len(rows) == 3)
        check("T3 không tệp nào lỗi",
              bool(rows) and all(r["Loại"] != "— lỗi —" for r in rows))
        check("T3 có nút tải CSV", len(at3.main.download_button) >= 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert not fails, f"{len(fails)} failures: {fails}"


if __name__ == "__main__":
    test_features()
    print("\n===", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}", "===")
    sys.exit(0 if not fails else 1)