"""Stage 8: generate the synthetic text corpus for the text baseline (E0b).

The image dataset (RVL-CDIP/SROIE) is scanned pages with no text layer and
OCR is OUT_OF_SCOPE, so 04 §1(b)/§6-E0 routes the text branch to a synthetic
corpus of realistic class-templated documents (marked EXTENSION per 04 §1
"synthetic data allowed as rendered variants"). Deterministic: seed 42.

Output: datasets/text/<class>/<doc_id>.{txt|md|html} + datasets/text/PROVENANCE_TEXT.csv
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc import paths  # noqa: E402

SEED = 42
PER_CLASS = 60
rng = random.Random(SEED)

COMPANIES = ["Acme Corporation", "Global Traders Ltd", "Sunrise Logistics",
             "Northwind Solutions", "BluePeak Consulting", "Metro Supplies"]
NAMES = ["John Smith", "Maria Nguyen", "David Tran", "Lan Pham", "Peter Le", "Anna Vo"]
ITEMS = ["office chair", "printer paper", "laptop stand", "USB cable", "desk lamp",
         "whiteboard marker", "notebook pack", "monitor arm"]

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def _date():
    return f"20{rng.randint(21, 25)}-{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}"


def _money():
    return f"{rng.uniform(5, 2000):.2f}"


def _invoice(i):
    lines = [f"INVOICE #INV-{10000 + i}", f"Date: {_date()}",
             f"From: {rng.choice(COMPANIES)}", "Bill to:", f"  {rng.choice(NAMES)}",
             "", "Items:"]
    total = 0.0
    for _ in range(rng.randint(1, 4)):
        q = rng.randint(1, 5)
        price = float(_money())
        item = rng.choice(ITEMS)
        lines.append(f"  {q} x {item} @ {price:.2f}")
        total += q * price
    lines += ["", f"TOTAL DUE: ${total:.2f}",
              "Payment terms: net 30 days.", "Please reference the invoice number."]
    return "\n".join(lines)


def _receipt(i):
    lines = [rng.choice(COMPANIES).upper(), "*** SALES RECEIPT ***",
             _date(), "-" * 24]
    for _ in range(rng.randint(2, 5)):
        lines.append(f"{rng.choice(ITEMS)[:14]:<14} {float(_money()):>8.2f}")
    lines += ["-" * 24, f"TOTAL       {float(_money()):>8.2f}",
              rng.choice(["CASH", "CARD"]), "THANK YOU - COME AGAIN!"]
    return "\n".join(lines)


def _report(i):
    title = f"Quarterly Performance Report Q{rng.randint(1, 4)}"
    body = [
        f"{title}", "", "Abstract:",
        f"This report summarizes operations for {rng.choice(COMPANIES)} during the period ending {_date()}.",
        "Key findings include steady growth in delivery throughput and a reduction in processing latency.",
        "We recommend expanding the automation pilot and reviewing vendor contracts next quarter.",
        "", f"Prepared by: {rng.choice(NAMES)}", "Distribution: internal use only.",
    ]
    return "\n".join(body)


def _letter(i):
    name = rng.choice(NAMES)
    body = [
        f"{_date()}", "",
        f"Dear {name.split()[0]},",
        "",
        f"Thank you for your continued partnership with {rng.choice(COMPANIES)}.",
        "We are writing regarding your account status and upcoming service changes.",
        "Please contact our office if you have any questions about this letter.",
        "", "Sincerely,", rng.choice(NAMES), "Customer Relations",
    ]
    return "\n".join(body)


def _form(i):
    fields = [
        "APPLICATION FORM", "=" * 18,
        f"Reference No.: FRM-{2000 + i}",
        f"Full name: ________ (e.g., {rng.choice(NAMES)})",
        "Date of birth: ____-__-__",
        f"Submission date: {_date()}",
        "Section A - Contact information",
        "  Phone: ___-___-____   Email: ____________",
        "Section B - Request details",
        f"  Requested item: {rng.choice(ITEMS)}",
        "  Justification: ______________________________",
        "Signature: ____________   Date: ________",
        "Office use only: approved / rejected / pending",
    ]
    return "\n".join(fields)


def _article(i):
    head = rng.choice([
        "City Approves New Recycling Program",
        "Local Team Wins Championship Final",
        "Study Finds Remote Work Boosts Output",
        "Council Debates Transit Expansion Plan",
    ])
    paras = [
        f"{head}",
        "",
        f"By {rng.choice(NAMES)} — published {_date()}.",
        f"Residents reacted this week after officials confirmed the decision affecting {rng.choice(COMPANIES)} and surrounding districts.",
        "\"We expected debate, but the outcome surprised many observers,\" one analyst said, noting similar moves elsewhere.",
        "Officials say implementation begins next month, with a public review scheduled within ninety days.",
    ]
    return "\n".join(paras)


GENERATORS = {
    "invoice": _invoice, "receipt": _receipt, "report": _report,
    "letter": _letter, "form": _form, "article": _article,
}
FORMATS = ["txt", "md", "html"]


def _wrap(fmt: str, body: str) -> str:
    if fmt == "html":
        return ("<html><body><pre>" + body.replace("&", "&amp;")
                .replace("<", "&lt;") + "</pre></body></html>")
    if fmt == "md":
        first, _, rest = body.partition("\n")
        return f"# {first}\n{rest}"
    return body


OUT = paths.DATASETS_DIR / "text"
rows = []
for cls, gen in GENERATORS.items():
    d = OUT / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(PER_CLASS):
        doc_id = f"{cls}_{i:04d}"
        fmt = FORMATS[i % len(FORMATS)]
        p = d / f"{doc_id}.{fmt}"
        p.write_text(_wrap(fmt, gen(i)), encoding="utf-8")
        rows.append([cls, doc_id, p.name])

with open(OUT / "PROVENANCE_TEXT.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["project_class", "doc_id", "file_name"])
    w.writerows(rows)

print(f"text corpus: {len(rows)} docs -> {OUT}")
from collections import Counter

print(dict(Counter(r[0] for r in rows)))