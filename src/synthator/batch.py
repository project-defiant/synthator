from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import polars as pl
from alphagenome.models import variant_scorers
from loguru import logger

from synthator.context import ContextualizedVariant
from synthator.transform import ensembl_to_ucsc


@dataclass
class ContextualizedVariantBatch:
    """Batch of interval-variant pairs."""

    interval_variants: list[ContextualizedVariant]
    """Single batch of contextualized variants."""
    batch_id: str
    """The id for the batch"""
    n_variants: int = 0
    """Number of variants in the batch."""

    def append_variant(self, variant: ContextualizedVariant) -> None:
        """Append a variant to the batch.

        :param variant: ContextualizedVariant to append to the batch.

        :return: None
        """
        self.interval_variants.append(variant)
        self.n_variants += 1


class VariantBatchGenerator:
    """Generator for batches of interval-variant pairs."""

    @staticmethod
    def _batch_id(row_number: pl.Expr, batch_window: int) -> pl.Expr:
        """Generate a batch ID for a variant based on its row number.

        :param row_number: Row number of the variant.
        :param batch_window: Size of the window to batch variants together.

        :return: Batch ID as a string expression.

        :example:
        For a batch window of 10, variants with row numbers 0-9 will have batch ID "0", variants with row numbers 10-19 will have batch ID "1", and so on.

        """
        return (row_number // batch_window).alias("batchId")

    @classmethod
    def _aggregate_variants_by_batch(
        cls, variant_index: pl.LazyFrame, batch_window: int
    ) -> pl.LazyFrame:
        """Aggregate variants into batches based on their row number.

        :param variant_index: LazyFrame containing the variant index.
        :param batch_window: Size of the window to batch variants together.

        :return: LazyFrame with variants aggregated into batches.
        """
        return (
            variant_index.sort("chromosome", "position", "referenceAllele", "alternateAllele")
            .with_row_index(name="rowNumber")
            .with_columns(
                batchId=cls._batch_id(
                    pl.col("rowNumber"),
                    batch_window,
                )
            )
            .group_by("batchId")
            .agg(
                pl.struct(
                    "chromosome",
                    "position",
                    "referenceAllele",
                    "alternateAllele",
                ).alias("variants")
            )
        )

    @classmethod
    def batch_variant_index(
        cls, variant_index: pl.LazyFrame, context_window: int, batch_window: int
    ) -> Generator[ContextualizedVariantBatch, None, None]:
        """Constructor for ContextualizedVariantBatch from a variant index.

        :param variant_index: LazyFrame containing the variant index.
        :param context_window: Size of the context window around the variant.
        :param batch_window: Size of the window to batch variants together.

        :return: Generator yielding IntervalVariantBatch instances.
        """

        collected = cls._aggregate_variants_by_batch(variant_index, batch_window).collect()
        assert isinstance(collected, pl.DataFrame)
        for batch_id, variant_list in collected.iter_rows():
            # slowest part of the code, we could consider parallelizing this in the future
            logger.debug(f"Processing batch {batch_id} with {len(variant_list)} variants.")
            interval_variants = []
            for v in variant_list:
                iv = ContextualizedVariant.from_variant(
                    chromosome=ensembl_to_ucsc(v["chromosome"]),
                    position=v["position"],
                    reference_bases=v["referenceAllele"],
                    alternate_bases=v["alternateAllele"],
                    window_size=context_window,
                    batch_id=batch_id,
                )
                interval_variants.append(iv)
            yield ContextualizedVariantBatch(interval_variants=interval_variants, batch_id=batch_id)


def annotate_batch(api_key: str, c_variants: ContextualizedVariantBatch) -> list[list[ad.Anndata]]:
    """Annotate a batch of variants with DNA client scores.

    :param api_key: API key for the DNA client.
    :param c_variants: Batch of contextualized variants to annotate.

    :return: List of lists of Anndata objects containing the annotations for each variant.
    """
    from alphagenome.models import dna_client

    _model_version = dna_client.ModelVersion.ALL_FOLDS
    client = dna_client.create(api_key=api_key, model_version=_model_version)
    variants = [cv.variant for cv in c_variants.interval_variants]
    intervals = [cv.interval for cv in c_variants.interval_variants]
    results = client.score_variants(variants=variants, intervals=intervals)
    return results


def transform_batch(annotation_result: list[list[ad.AnnData]]) -> pl.DataFrame:
    """Transform the annotated batch of variants into a Polars DataFrame.

    :param annotation_result: List of lists of Anndata objects containing the annotations for each variant.

    :return: Polars DataFrame containing the transformed annotations for each variant.
    """
    data = variant_scorers.tidy_scores(annotation_result)
    assert data is not None, "No data returned from variant scorers."
    data["variant_id"] = data["variant_id"].astype(str)
    data["scored_interval"] = data["scored_interval"].astype(str)
    d = pl.DataFrame(data)
    return d


def write_batch(transformed_batch: pl.DataFrame, output_path: str, batch_id: str) -> None:
    """Write the transformed batch of annotations to a specified output path.

    :param transformed_batch: Polars DataFrame containing the transformed annotations for each variant.
    :param output_path: Path to write the output data. Supports local paths and GCS (gs://bucket/path).
    :param batch_id: ID of the batch being processed.

    :return: None
    """
    if "://" not in output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
    transformed_batch.write_parquet(f"{output_path}/batch_{batch_id}.parquet")


def batch_output_exists(output_path: str, batch_id: str) -> bool:
    """Check whether the output parquet file for a batch already exists.

    :param output_path: Base output path. Supports local paths and GCS (gs://bucket/path).
    :param batch_id: Batch ID used when writing the file.

    :return: True if the output file already exists, False otherwise.
    """
    path = f"{output_path}/batch_{batch_id}.parquet"
    if "://" not in output_path:
        return Path(path).exists()
    try:
        pl.scan_parquet(path).limit(0).collect()
        return True
    except Exception:
        return False


def process_batch(api_key: str, c_variants: ContextualizedVariantBatch, output_path: str) -> None:
    """Process a batch of contextualized variants by annotating them and transforming the results.

    :param api_key: API key for the DNA client.
    :param c_variants: Batch of contextualized variants to process.
    :param output_path: Path to write the output data. Supports local paths and GCS (gs://bucket/path).

    :return: None
    """
    annotation_result = annotate_batch(api_key, c_variants)
    transformed_result = transform_batch(annotation_result)
    write_batch(transformed_result, output_path, c_variants.batch_id)
