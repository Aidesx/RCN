"""Document understanding CLI (v2): file / folder / demo -> JSON + Markdown report.

Usage:
  python scripts/understand_text.py <file>            one file (any supported type)
  python scripts/understand_text.py <folder>          recurse: one report per file
  python scripts/understand_text.py --demo            built-in sample
Options:
  --out DIR      write reports here (default: alongside the input file)
  --k-keywords N top-k keyphrases (default 10)
Exit codes: 0 = all reports produced · 2 = nothing understood / usage error
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc.nlp import render_markdown, understand_file  

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf", ".png", ".jpg", ".jpeg"}


def _process(path: Path, out_dir: Path, k_keywords: int) -> tuple[bool, str]:
    try:
        t0 = time.perf_counter()
        record = understand_file(path, k_keywords=k_keywords)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"[skip ] {path}: {exc}")
        return False, f"{path}: {exc}"

    json_path = out_dir / f"{path.name}.understanding.json"
    md_path = out_dir / f"{path.name}.understanding.md"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(record), encoding="utf-8")

    label = record.get("doc_type", {}).get("label", "?")
    n_para = record.get("structure", {}).get("stats", {}).get("paragraphs")
    summary = f"label={label}"
    if n_para is not None:
        summary += f", paragraphs={n_para}, keywords={len(record.get('keywords', []))}"
    print(f"[ok   ] {path} -> {json_path.name} + {md_path.name} ({elapsed:.2f}s) {summary}")
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", help="file or folder; omit with --demo")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None, help="output directory for reports")
    ap.add_argument("--k-keywords", type=int, default=10)
    args = ap.parse_args()

    if args.demo:
        from docproc.nlp import analyze_structure

        demo_invoice = ("ACME Corporation - INVOICE #INV-10482\n\n"
                        "Nguyen Van A purchased 3 x office chair @ 45.00.\n"
                        "Total due is $147.50 within 30 days.\n"
                        "Contact Dr. Smith at accounting.")
        structure = analyze_structure(demo_invoice)
        print(json.dumps(structure["stats"], indent=2, ensure_ascii=False))
        return 0

    if not args.target:
        ap.print_help()
        return 2

    target = Path(args.target)
    if not target.exists():
        print(f"not found: {target}")
        return 2

    if target.is_dir():
        # include extensionless files too — detection content-sniffs them;
        # truly unsupported types surface as transparent [skip] lines below.
        files = sorted(f for f in target.rglob("*") if f.is_file())
    else:
        files = [target]

    out_dir = Path(args.out) if args.out else (target.parent if target.is_file() else target)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for f in files:
        success, _ = _process(f, out_dir, args.k_keywords)
        ok += int(success)

    if not files:
        print("no supported files found")
        return 2
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())