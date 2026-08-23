"""L4 field extraction: regex schemas per document class; generic fallback."""
from __future__ import annotations

import re

_DATE_ISO = r"\d{4}-\d{2}-\d{2}"
_DATE_DMY = r"\d{1,2}/\d{1,2}/\d{2,4}"
_MONEY = r"(?:\$|USD\s*)?\d[\d,]*(?:\.\d{1,2})?"

_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "invoice": {
        "invoice_number": [r"(?:INVOICE|Invoice)\s*(?:#|number|#\s*|No\.?[:\s]*)\s*([A-Z]*-?\d+)",
                           r"#(INV-\d+)"],
        "date": [rf"Date:\s*({_DATE_ISO}|{_DATE_DMY})", rf"issue[d]?\s*(?:date)?[:\s]*({_DATE_ISO}|{_DATE_DMY})", rf"({_DATE_ISO}|{_DATE_DMY})"],
        "total_due": [rf"(?:TOTAL DUE|Total due|Total)\s*:?\s*\$?\s*({_MONEY})"],
        "vendor": [r"From:\s*(.+)", r"issued by\s+(.+?)\."],
        "buyer": [r"(?:Bill to|Billed to|Customer)[:\s]*\n?\s*(.+)", r"purchased by\s+(.+?)\b"],
    },
    "receipt": {
        "merchant": [r"^\s*([A-Z][A-Z0-9 &.'\-]{3,})\s*$"],
        "date": [rf"({_DATE_ISO}|{_DATE_DMY})"],
        "total": [rf"(?:TOTAL|Total)\s*:?\s*\$?\s*({_MONEY})"],
        "payment_method": [r"\b(CASH|CARD|VISA|MASTERCARD|AMEX)\b"],
    },
    "report": {
        "title": [r"^\s*(Quarterly .+|Annual .+|.+\bReport\b.*)$", ],
        "period_end_date": [rf"(?:ending|period)\s*(?:on)?\s*({_DATE_ISO}|{_DATE_DMY})"],
        "prepared_by": [r"Prepared by:\s*(.+)"],
        "organization": [r"for\s+([A-Z][A-Za-z&. ]+)\s+during", r"summarizes operations for (.+?) during"],
    },
    "letter": {
        "date": [rf"^\s*({_DATE_ISO}|{_DATE_DMY})\s*$"],
        "recipient_first_name": [r"Dear\s+(\w+)"],
        "sender": [r"Sincerely,\s*\n(.+)"],
        "organization": [r"partnership with (.+?)\.", r"regarding your (?:account|service).*(?:with\s+)?([A-Z][A-Za-z ]+)"],
    },
    "form": {
        "reference_number": [r"(?:Reference No\.?|Ref\.?)\s*:?\s*([A-Z]{2,4}-?\d+)"],
        "submission_date": [rf"Submission date:\s*({_DATE_ISO}|{_DATE_DMY})"],
        "requested_item": [r"Requested item:\s*(.+)"],
        "status_options": [r"Office use only:\s*(.+)"],
    },
    "article": {
        "headline": [r"^\s*(.+)$"],
        "author": [r"By\s+(.+?)(?:\s+—|\s+-|\s*$)", r"By ([A-Z]\w+ \w+)"],
        "published_date": [rf"published\s+({_DATE_ISO}|{_DATE_DMY})"],
    },
}

_GENERIC: dict[str, list[str]] = {
    "dates": [rf"({_DATE_ISO}|{_DATE_DMY})"],
    "amounts": [r"(?:\$|USD)\s*([\d,]+(?:\.\d{1,2})?)",
                r"\b(\d{1,3}(?:,\d{3})*\.\d{2})\b"],
}

_ALL_FIELDS_CACHE: dict[str, re.Pattern] = {}


def _compiled(pattern: str) -> re.Pattern:
    if pattern not in _ALL_FIELDS_CACHE:
        _ALL_FIELDS_CACHE[pattern] = re.compile(pattern, re.MULTILINE)
    return _ALL_FIELDS_CACHE[pattern]


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = _compiled(pat).search(text)
        if m:
            value = (m.group(1) if m.groups() else m.group(0)).strip().strip("|").strip()
            value = value.lstrip("# ").strip() or None
            return value
    return None


def extract_fields(text: str, doc_type: str | None = None) -> dict:
    """Extract fields per class schema; unknown types use the generic schema."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    schema_key = doc_type if doc_type in _SCHEMAS else "generic"
    schema = _SCHEMAS.get(schema_key, _GENERIC)

    fields = {name: _first_match(text, pats) for name, pats in schema.items()}
    matched = sum(1 for v in fields.values() if v)
    return {"doc_type": schema_key, "fields": fields, "matched": matched}