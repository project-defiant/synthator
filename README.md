# synthator

Annotate a genomic variant index with [AlphaGenome](https://github.com/google-deepmind/alphagenome_research) variant effect predictions and write the results as Parquet files.

## Overview

`synthator` takes a variant index (Parquet), groups variants into genomic batches, calls the AlphaGenome DNA client to score each batch, and writes tidy Parquet output — one file per batch.

```text
variant_index.parquet
        │
        ▼
VariantBatchGenerator   ← groups nearby variants into batches
        │
        ▼
  annotate_batch        ← calls AlphaGenome DNA client
        │
        ▼
  transform_batch       ← tidy_scores → Polars DataFrame
        │
        ▼
   write_batch          ← batch_{id}.parquet  (local or GCS)
```

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# runtime only
uv sync

# runtime + dev tools (pytest, ruff, ty)
uv sync --group dev
```

## Usage

```bash
synthator alpha-genome \
  --variant-index-path data/variants.parquet \
  --api-key YOUR_ALPHAGENOME_KEY \
  --output gs://my-bucket/results \
  --context-window 1048576 \
  --batch-window 10 \
  --no-test-mode
```

| Option | Default | Description |
| --- | --- | --- |
| `--variant-index-path` | *(required)* | Path to variant index Parquet file |
| `--api-key` | *(required)* | AlphaGenome API key |
| `--output` | `data/alphagenome` | Output directory (local or `gs://`) |
| `--context-window` | `1048576` | Sequence context length in bp |
| `--batch-window` | `10` | Variants per genomic batch |
| `--test-mode / --no-test-mode` | `--test-mode` | Stop after 2 batches (for debugging) |

Output files are written as `{output}/batch_{batch_id}.parquet`.

## Development

### Running tests

```bash
uv run pytest
```

Coverage is reported in the terminal and written to `coverage.xml`.

### Linting and formatting

```bash
uv run ruff check src/ tests/   # lint
uv run ruff format src/ tests/  # auto-format
```

### Type checking

```bash
uv run ty check src/
```

## Architecture

| Module | Purpose |
| --- | --- |
| `synthator/__init__.py` | Typer CLI entry point |
| `synthator/context.py` | `ContextualizedVariant` — pairs a `Variant` with its genomic `Interval` |
| `synthator/batch.py` | `ContextualizedVariantBatch`, `VariantBatchGenerator`, scoring and I/O pipeline |
| `synthator/transform.py` | Polars expression helpers for parsing scorer strings, variant IDs, and intervals |
| `synthator/input.py` | Polars schema definition for the variant index |

## CI

GitHub Actions runs on every push and pull request to `main`:

1. **Lint** — `ruff format --check` + `ruff check`
2. **Type check** — `ty check src/`
3. **Test** — `pytest` with coverage (requires lint + typecheck to pass first)

Coverage is uploaded as a workflow artifact (`coverage.xml`).
