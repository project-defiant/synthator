from __future__ import annotations

from typing import Annotated

import polars as pl
import typer
from alphagenome.models import dna_client
from loguru import logger

from synthator.batch import VariantBatchGenerator, process_batch
from synthator.input import VariantSchema

app = typer.Typer()


@app.command("alpha-genome")
def cli(
    variant_index_path: Annotated[
        str,
        typer.Option(
            help="Path to variant index. Supports local paths and GCS globs (gs://bucket/path/*.parquet)."
        ),
    ],
    api_key: Annotated[str, typer.Option(help="API key.")],
    context_window: Annotated[int, typer.Option(help="Sequence length to use for predictions.")] = 2
    ** 20,
    output: Annotated[
        str,
        typer.Option(help="Output path. Supports local paths and GCS (gs://bucket/path)."),
    ] = "data/alphagenome",
    test_mode: Annotated[
        bool,
        typer.Option(help="Whether to run in test mode (process only 2 batches)."),
    ] = True,
    batch_window: Annotated[
        int, typer.Option(help="Number of variants to process in each batch.")
    ] = 10,
) -> None:

    logger.info(f"Using variant index from {variant_index_path}")
    logger.info(f"Using output path {output}")
    _cw = dna_client.SUPPORTED_SEQUENCE_LENGTHS.get("SEQUENCE_LENGTH_1MB", context_window)
    _v = pl.scan_parquet(variant_index_path, schema=VariantSchema.schema)

    logger.success("Loaded variant index.")

    logger.info(f"Using sequence context of {_cw}bp.")

    _iter = VariantBatchGenerator.batch_variant_index(
        variant_index=_v, context_window=_cw, batch_window=batch_window
    )
    for i, _batch in enumerate(_iter):
        logger.info(f"Processing batch {i}.")
        logger.debug(f"Batch {i} contains {_batch.n_variants} variants.")
        logger.debug(f"Batch {i} has batch ID: {_batch.batch_id}")
        process_batch(api_key=api_key, c_variants=_batch, output_path=output)
        logger.success(f"Finished processing batch {i}.")
        if i >= 1 and test_mode:
            logger.info("Stopping after 2 batches for testing purposes.")
            break
