"""Shared fixtures for the synthator test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest

from synthator.batch import ContextualizedVariantBatch
from synthator.context import ContextualizedVariant


@pytest.fixture()
def mock_interval() -> MagicMock:
    interval = MagicMock()
    interval.chromosome = "chr1"
    interval.start = 1000
    interval.end = 2000
    return interval


@pytest.fixture()
def mock_variant(mock_interval: MagicMock) -> MagicMock:
    variant = MagicMock()
    variant.chromosome = "chr1"
    variant.position = 1500
    variant.reference_bases = "A"
    variant.alternate_bases = "T"
    variant.name = "chr1_1500_A_T"
    variant.info = {}
    variant.reference_interval.resize.return_value = mock_interval
    return variant


@pytest.fixture()
def mock_contextualized_variant(
    mock_interval: MagicMock, mock_variant: MagicMock
) -> ContextualizedVariant:
    return ContextualizedVariant(interval=mock_interval, variant=mock_variant)


@pytest.fixture()
def mock_batch(mock_contextualized_variant: ContextualizedVariant) -> ContextualizedVariantBatch:
    batch = ContextualizedVariantBatch(
        interval_variants=[mock_contextualized_variant],
        batch_id="chr1_15",
        n_variants=1,
    )
    return batch


@pytest.fixture()
def sample_scorer_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "variant_scorer": [
                "GeneMaskActiveScorer(requested_output=RNA_SEQ)",
                "GeneMaskLFCScorer(requested_output=RNA_SEQ)",
                "CenterMaskScorer(requested_output=CHIP_TF, width=501, aggregation_type=ACTIVE_SUM)",
                "CenterMaskScorer(requested_output=CHIP_HISTONE, width=2001, aggregation_type=DIFF_LOG2_SUM)",
                "SpliceJunctionScorer()",
                "PolyadenylationScorer()",
                "GeneMaskSplicingScorer(requested_output=SPLICE_SITE_USAGE, width=None)",
                "ContactMapScorer()",
            ]
        }
    )


@pytest.fixture()
def sample_tidy_df() -> pl.DataFrame:
    """Minimal DataFrame matching transform_output's expected input columns."""
    return pl.DataFrame(
        {
            "variant_id": ["chr1:12345:A>T"],
            "scored_interval": ["chr1:1000-2000"],
            "variant_scorer": [
                "CenterMaskScorer(requested_output=CHIP_TF, width=501, aggregation_type=ACTIVE_SUM)"
            ],
            "raw_score": [0.5],
            "quantile_score": [0.8],
            "data_source": ["ENCODE"],
            "gene_id": ["ENSG00000001"],
            "ontology_curie": ["CL:0000001"],
            "gene_name": ["TP53"],
            "gene_type": ["protein_coding"],
            "gene_strand": ["+"],
            "junction_Start": [None],
            "junction_End": [None],
            "track_name": ["track_1"],
            "track_strand": ["+"],
            "Assay title": ["ChIP-seq"],
            "biosample_name": ["K562"],
            "biosample_type": ["cell line"],
            "biosample_life_stage": ["adult"],
            "endedness": ["single-ended"],
            "genetically_modified": [False],
            "transcription_factor": ["CTCF"],
            "histone_mark": [None],
            "gtex_tissue": [None],
        }
    )
