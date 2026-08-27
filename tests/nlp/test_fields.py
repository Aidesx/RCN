"""L4 field extraction tests: per-class schemas + generic fallback + report wiring."""
import pytest

from docproc.nlp.fields import extract_fields

INVOICE = ("INVOICE #INV-10482\nDate: 2025-03-14\nFrom: Acme Corporation\n"
           "Bill to:\n  Maria Nguyen\nItems:\n  2 x desk lamp @ 12.50\n"
           "TOTAL DUE: $147.50")

RECEIPT = "BLUEPEAK CONSULTING\n2025-05-15\nlamp        25.00\ntotal       50.00\nTOTAL    50.00\nCARD"

LETTER = ("2025-02-01\n\nDear Maria,\n\nThank you for your partnership with Acme Corp.\n\n"
          "Sincerely,\nJohn Smith")


class TestInvoice:
    def test_all_core_fields(self):
        out = extract_fields(INVOICE, "invoice")
        f = out["fields"]
        assert out["doc_type"] == "invoice" and out["matched"] >= 4
        assert f["invoice_number"] == "INV-10482"
        assert f["date"] == "2025-03-14"
        assert f["total_due"] == "147.50"
        assert f["vendor"] == "Acme Corporation"
        assert f["buyer"] == "Maria Nguyen"


class TestReceipt:
    def test_merchant_total_payment(self):
        out = extract_fields(RECEIPT, "receipt")
        f = out["fields"]
        assert f["merchant"] == "BLUEPEAK CONSULTING"
        assert f["total"] == "50.00"
        assert f["payment_method"] == "CARD"


class TestLetter:
    def test_recipient_and_date(self):
        out = extract_fields(LETTER, "letter")
        f = out["fields"]
        assert f["date"] == "2025-02-01"
        assert f["recipient_first_name"] == "Maria"


class TestFallbackAndContract:
    def test_unknown_type_uses_generic(self):
        out = extract_fields("met on 2025-01-02 about $30.00", None)
        assert out["doc_type"] == "generic"
        assert "2025-01-02" in out["fields"]["dates"]

    def test_unavailable_label_falls_back_generic(self):
        out = extract_fields(INVOICE, "unavailable")
        assert out["doc_type"] == "generic"

    def test_no_match_returns_nones(self):
        out = extract_fields("nothing useful here at all.", "form")
        assert all(v is None for v in out["fields"].values())
        assert out["matched"] == 0

    def test_deterministic(self):
        a = extract_fields(INVOICE, "invoice")
        b = extract_fields(INVOICE, "invoice")
        assert a == b

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            extract_fields(None, "invoice")


class TestSchemaConfig:
    def test_override_changes_behavior(self, tmp_path):
        cfg = tmp_path / "fields.yaml"
        cfg.write_text(r"""invoice:
  required: [invoice_number, total_due]
  fields:
    invoice_number:
      patterns: ['REF-(\d+)']
""", encoding="utf-8")
        # pattern mới thắng; total_due không khai → giữ built-in
        out = extract_fields("REF-9991 and also #INV-10482", "invoice",
                             config_path=cfg)
        assert out["fields"]["invoice_number"] == "9991"
        out2 = extract_fields(INVOICE, "invoice", config_path=cfg)
        assert out2["fields"]["total_due"] == "147.50"
        assert out2["fields"]["date"] == "2025-03-14"

    def test_required_reports_missing(self, tmp_path):
        cfg = tmp_path / "fields.yaml"
        cfg.write_text(
            "invoice:\n"
            "  required: [invoice_number, total_due, vendor]\n",
            encoding="utf-8")
        out = extract_fields(INVOICE, "invoice", config_path=cfg)
        assert out["missing_required"] == []
        out2 = extract_fields("no invoice here", "invoice", config_path=cfg)
        assert out2["missing_required"] == \
            ["invoice_number", "total_due", "vendor"]

    def test_normalize_money_and_date(self, tmp_path):
        cfg = tmp_path / "fields.yaml"
        cfg.write_text(r"""invoice:
  fields:
    total_due:
      patterns: ['TOTAL\s*:?\s*\$?\s*((?:\$|USD\s*)?\d[\d,]*(?:\.\d{1,2})?)']
      normalize: money
    date:
      patterns: ['Date:\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})']
      normalize: date
""", encoding="utf-8")
        out = extract_fields("TOTAL: $2,008.40\nDate: 30/04/2021", "invoice",
                             config_path=cfg)
        assert out["fields"]["total_due"] == "2008.40"
        assert out["fields"]["date"] == "2021-04-30"
        # ambiguous 08/09/2021 (cả 2 ≤ 12) → giữ nguyên, không đoán
        out2 = extract_fields("TOTAL: 5\nDate: 08/09/2021", "invoice",
                              config_path=cfg)
        assert out2["fields"]["date"] == "08/09/2021"

    def test_bad_yaml_falls_back(self, tmp_path):
        cfg = tmp_path / "fields.yaml"
        cfg.write_text("invoice: [unclosed", encoding="utf-8")
        out = extract_fields(INVOICE, "invoice", config_path=cfg)
        assert out["fields"]["invoice_number"] == "INV-10482"

    def test_project_config_active(self):
        # khi project có configs/fields.yaml, invoice dùng override + required
        out = extract_fields(INVOICE, "invoice")
        assert out["missing_required"] == []
        assert out["fields"]["total_due"] == "147.50"
        assert out["fields"]["date"] == "2025-03-14"


class TestReportIntegration:
    def test_understand_includes_fields_section(self):
        from docproc.nlp.report import render_markdown, understand

        r = understand(INVOICE, source="t")
        assert "fields" in r
        md = render_markdown(r)
        assert "## Fields" in md and "INV-10482" in md