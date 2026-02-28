from __future__ import annotations

from typing import Annotated

import polars as pl
import typer
from alphagenome.models import dna_client
from loguru import logger

from synthator.batch import VariantBatchGenerator, batch_output_exists, process_batch
from synthator.input import VariantSchema

app = typer.Typer()


@app.command("alpha-genome")
def cli(
    variant_index_path: Annotated[
        str,
        typer.Option(help="Path to variant index. Supports local paths and GCS globs (gs://bucket/path/*.parquet)."),
    ],
    api_key: Annotated[str | None, typer.Option(help="API key. If provided the scoring will be done via api instead of local model.")] = None, 
    context_window: Annotated[int, typer.Option(help="Sequence length to use for predictions.")] = 2**20,
    output: Annotated[
        str,
        typer.Option(help="Output path. Supports local paths and GCS (gs://bucket/path)."),
    ] = "data/alphagenome",
    test_mode: Annotated[
        bool,
        typer.Option(help="Whether to run in test mode (process only 2 batches)."),
    ] = True,
    batch_window: Annotated[int, typer.Option(help="Number of variants to process in each batch.")] = 10,
    resume: Annotated[
        bool,
        typer.Option(help="Skip batches whose output file already exists."),
    ] = False,
    score_with_model: Annotated[bool, typer.Option(help="Whether to score variants with the model (requires API key).")] = True,
) -> None:

    logger.info(f"Using variant index from {variant_index_path}")
    logger.info(f"Using output path {output}")
    _cw = dna_client.SUPPORTED_SEQUENCE_LENGTHS.get("SEQUENCE_LENGTH_1MB", context_window)
    _v = pl.scan_parquet(variant_index_path, schema=VariantSchema.schema)

    logger.success("Loaded variant index.")

    logger.info(f"Using sequence context of {_cw}bp.")

    _iter = VariantBatchGenerator.batch_variant_index(variant_index=_v, context_window=_cw, batch_window=batch_window)
    for i, _batch in enumerate(_iter):
        if resume and batch_output_exists(output, _batch.batch_id):
            logger.info(f"Skipping batch {i} (batch_id={_batch.batch_id}): output already exists.")
            continue
        logger.info(f"Processing batch {i}.")
        logger.debug(f"Batch {i} contains {_batch.n_variants} variants.")
        logger.debug(f"Batch {i} has batch ID: {_batch.batch_id}")
        if api_key is not None:
            logger.info("Scoring variants with API key.")
            process_batch(api_key=api_key, c_variants=_batch, output_path=output)
        else:
            logger.info("Scoring variants with local model.")
            process_batch(c_variants=_batch, output_path=output, api_key=None)
        logger.success(f"Finished processing batch {i}.")
        if i >= 1 and test_mode:
            logger.info("Stopping after 2 batches for testing purposes.")
            break
