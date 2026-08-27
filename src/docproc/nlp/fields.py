"""L4 field extraction: regex schemas per document class; generic fallback.

Schemas are configurable: a `configs/fields.yaml` (or an explicit path) can
override per-class patterns/normalizers/required fields; classes not listed
keep the built-in defaults below. Deterministic and offline.
"""
from __future__ import annotations

import re
from pathlib import Path

from docproc.paths import CONFIG_DIR

_FIELDS_YAML = "fields.yaml"
_OVERRIDE_CACHE: dict[str, dict] = {}

_DATE_ISO = r"\d{4}-\d{2}-\d{2}"
_DATE_DMY = r"\d{1,2}/\d{1,2}/\d{2,4}"
_MONEY = r"(?:\$|USD\s*)?\d[\d,]*(?:\.\d{1,2})?"

_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "invoice": {
        "invoice_number": [r"(?:INVOICE|Invoice)\s*(?:#|number|#\s*|No\.?[:\s]*)\s*([A-Z]{0,5}-?\d+(?:-\d+)*)",
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


def _load_overrides(path: Path | None) -> dict:
    """Load per-class schema overrides from YAML; graceful on any error."""
    key = str(path) if path else _FIELDS_YAML
    if key in _OVERRIDE_CACHE:
        return _OVERRIDE_CACHE[key]
    data: dict = {}
    target = path if path is not None else CONFIG_DIR / _FIELDS_YAML
    if target.exists():
        try:
            import yaml

            raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            data = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            data = {}
    _OVERRIDE_CACHE[key] = data
    return data


def _normalize(value: str, kind: str | None) -> str | None:
    """Deterministic value normalization: date/money/upper."""
    if value is None or not kind or kind == "none":
        return value
    if kind == "upper":
        return value.upper()
    if kind == "money":
        return re.sub(r"[$,USD\s]", "", value)
    if kind == "date":
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if m:
            return value
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})", value)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
            year = f"20{y}" if len(y) == 2 else y
            if a > 12 >= b:      # day-first (d/m/y)
                return f"{year}-{b:02d}-{a:02d}"
            if b > 12 >= a:      # month-first (m/d/y)
                return f"{year}-{a:02d}-{b:02d}"
            # ambiguous (both ≤ 12): keep raw, never guess
        return value
    return value


def extract_fields(text: str, doc_type: str | None = None,
                   config_path: str | Path | None = None) -> dict:
    """Extract fields per class schema; unknown types use the generic schema.

    ``config_path`` (optional) points at a YAML override file; when omitted the
    project ``configs/fields.yaml`` is used if present, else built-in defaults.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    path = Path(config_path) if config_path else None
    overrides = _load_overrides(path)

    schema_key = doc_type if (doc_type in _SCHEMAS or doc_type in overrides) \
        else "generic"
    if schema_key == "generic":
        schema, normalizers, required = _GENERIC, {}, []
    else:
        ov = overrides.get(schema_key, {}) or {}
        spec_fields = ov.get("fields", {}) or {}
        default = _SCHEMAS.get(schema_key, {})
        schema: dict[str, list[str]] = {}
        normalizers: dict[str, str] = {}
        for name in sorted(set(default) | set(spec_fields)):
            if name in spec_fields:
                spec = spec_fields[name] or {}
                schema[name] = spec.get("patterns") or default.get(name) or []
                nrm = spec.get("normalize", "none")
                if nrm not in (None, "none"):
                    normalizers[name] = nrm
            else:
                schema[name] = default[name]
        required = list(ov.get("required", []))

    fields = {}
    for name, pats in schema.items():
        raw = _first_match(text, pats)
        fields[name] = _normalize(raw, normalizers.get(name)) \
            if raw is not None else None
    matched = sum(1 for v in fields.values() if v)
    missing_required = [name for name in required if not fields.get(name)]
    return {"doc_type": schema_key, "fields": fields, "matched": matched,
            "missing_required": missing_required}