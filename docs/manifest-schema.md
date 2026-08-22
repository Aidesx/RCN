# Dataset Manifest Schema (Stage 1 / Stage 2)

Two artifacts are produced by Stage 2 and consumed by every later stage:

## 1. Provenance (exists already: `datasets/raw/PROVENANCE.csv`)

| Column | Type | Meaning |
|--------|------|---------|
| project_class | str | Project class name (`letter`, `form`, `report`, `article`, `invoice`, `receipt`) |
| local_file | str | File name inside `datasets/raw/<class>/` (e.g. `letter_0000.png`) |
| source_split | str | Split inside the source dataset the page came from (external sources only) |
| rvlcdip_filename | str | Original file name in the source dataset (= document id) |
| label_id | int | Label id in the source dataset (external sources only) |

## 2. Split manifest (created in Stage 2: `datasets/splits/manifest.csv`)

One row per page image. Used for train/val/test loading, leak checks, and metrics.

| Column | Type | Meaning |
|--------|------|---------|
| doc_id | str | Document identifier — pages of one document share it (from source filename, else generated) |
| class | str | Project class name |
| page_file | str | Relative path to the page image (`datasets/pages/<class>/<file>.png` or `datasets/raw/<class>/<file>.png`) |
| source | str | Provenance: external dataset / rendered / collected |
| split | str | `train` \| `validation` \| `test` |

## Rules (per 04 §1)

- Split by **document** (`doc_id`), never by page: all pages of one `doc_id` land in the same split.
- Stratified by class; ratios 70/15/15.
- Test split frozen after first build (no re-split without a spec update per 06 §14).
- No duplicate `doc_id` across splits; every class present in every split (≥50 pages/class recommended).
- `split.by: document` and `split.seed: 42` in `configs/dataset.yaml` drive reproducibility.