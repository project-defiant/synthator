"""Tests for synthator.transform module."""

from __future__ import annotations

import polars as pl
import pytest

from synthator.transform import (
    ensembl_to_ucsc,
    parse_scorer,
    parse_variant_id,
    scored_interval_to_interval_struct,
    transform_output,
    ucsc_to_ensembl,
)


# ---------------------------------------------------------------------------
# Chromosome conversion
# ---------------------------------------------------------------------------


class TestEnsemblToUcsc:
    def test_regular_chromosome(self) -> None:
        assert ensembl_to_ucsc("1") == "chr1"

    def test_x_chromosome(self) -> None:
        assert ensembl_to_ucsc("X") == "chrX"

    def test_y_chromosome(self) -> None:
        assert ensembl_to_ucsc("Y") == "chrY"

    def test_mt_becomes_chrm(self) -> None:
        assert ensembl_to_ucsc("MT") == "chrM"

    def test_numeric_string(self) -> None:
        assert ensembl_to_ucsc("22") == "chr22"


class TestUcscToEnsembl:
    def test_regular_chromosome(self) -> None:
        assert ucsc_to_ensembl("chr1") == "1"

    def test_x_chromosome(self) -> None:
        assert ucsc_to_ensembl("chrX") == "X"

    def test_y_chromosome(self) -> None:
        assert ucsc_to_ensembl("chrY") == "Y"

    def test_chrm_becomes_mt(self) -> None:
        assert ucsc_to_ensembl("chrM") == "MT"

    def test_chr22(self) -> None:
        assert ucsc_to_ensembl("chr22") == "22"

    def test_roundtrip(self) -> None:
        for chrom in ["1", "X", "MT", "22"]:
            assert ucsc_to_ensembl(ensembl_to_ucsc(chrom)) == chrom


# ---------------------------------------------------------------------------
# parse_scorer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scorer_str, expected_name, expected_output, expected_width, expected_agg",
    [
        (
            "CenterMaskScorer(requested_output=CHIP_TF, width=501, aggregation_type=ACTIVE_SUM)",
            "CenterMaskScorer",
            "CHIP_TF",
            501,
            "ACTIVE_SUM",
        ),
        (
            "CenterMaskScorer(requested_output=CHIP_HISTONE, width=2001, aggregation_type=DIFF_LOG2_SUM)",
            "CenterMaskScorer",
            "CHIP_HISTONE",
            2001,
            "DIFF_LOG2_SUM",
        ),
        (
            "CenterMaskScorer(requested_output=CAGE, width=501, aggregation_type=ACTIVE_SUM)",
            "CenterMaskScorer",
            "CAGE",
            501,
            "ACTIVE_SUM",
        ),
        (
            "CenterMaskScorer(requested_output=DNASE, width=501, aggregation_type=DIFF_LOG2_SUM)",
            "CenterMaskScorer",
            "DNASE",
            501,
            "DIFF_LOG2_SUM",
        ),
        (
            "CenterMaskScorer(requested_output=ATAC, width=501, aggregation_type=ACTIVE_SUM)",
            "CenterMaskScorer",
            "ATAC",
            501,
            "ACTIVE_SUM",
        ),
        (
            "GeneMaskActiveScorer(requested_output=RNA_SEQ)",
            "GeneMaskActiveScorer",
            "RNA_SEQ",
            None,
            None,
        ),
        (
            "GeneMaskLFCScorer(requested_output=RNA_SEQ)",
            "GeneMaskLFCScorer",
            "RNA_SEQ",
            None,
            None,
        ),
        (
            "GeneMaskSplicingScorer(requested_output=SPLICE_SITE_USAGE, width=None)",
            "GeneMaskSplicingScorer",
            "SPLICE_SITE_USAGE",
            None,  # "None" string doesn't match \d+
            None,
        ),
        ("SpliceJunctionScorer()", "SpliceJunctionScorer", None, None, None),
        ("PolyadenylationScorer()", "PolyadenylationScorer", None, None, None),
        ("ContactMapScorer()", "ContactMapScorer", None, None, None),
    ],
)
def test_parse_scorer(
    scorer_str: str,
    expected_name: str,
    expected_output: str | None,
    expected_width: int | None,
    expected_agg: str | None,
) -> None:
    df = pl.DataFrame({"s": [scorer_str]})
    result = df.select(parse_scorer(pl.col("s")).alias("parsed"))["parsed"][0]

    assert result["scorerName"] == expected_name
    assert result["requestedOutput"] == expected_output
    assert result["width"] == expected_width
    assert result["aggregationType"] == expected_agg


def test_parse_scorer_null_input() -> None:
    df = pl.DataFrame({"s": [None]}, schema={"s": pl.String})
    result = df.select(parse_scorer(pl.col("s")).alias("parsed"))["parsed"][0]
    assert result["scorerName"] is None
    assert result["requestedOutput"] is None
    assert result["width"] is None
    assert result["aggregationType"] is None


# ---------------------------------------------------------------------------
# scored_interval_to_interval_struct
# ---------------------------------------------------------------------------


class TestScoredIntervalToIntervalStruct:
    def _parse(self, interval_str: str) -> dict:
        df = pl.DataFrame({"si": [interval_str]})
        return df.select(scored_interval_to_interval_struct(pl.col("si")))["interval"][0]

    def test_standard_interval(self) -> None:
        result = self._parse("chr1:1000-2000")
        assert result["chromosome"] == "1"
        assert result["start"] == 1000
        assert result["end"] == 2000

    def test_chrm_converted_to_mt(self) -> None:
        result = self._parse("chrM:500-1500")
        assert result["chromosome"] == "MT"

    def test_x_chromosome(self) -> None:
        result = self._parse("chrX:0-1048576")
        assert result["chromosome"] == "X"

    def test_start_end_types(self) -> None:
        result = self._parse("chr22:10000-20000")
        assert isinstance(result["start"], int)
        assert isinstance(result["end"], int)

    def test_large_coordinates(self) -> None:
        result = self._parse("chr1:100000000-101048576")
        assert result["start"] == 100000000
        assert result["end"] == 101048576


# ---------------------------------------------------------------------------
# parse_variant_id
# ---------------------------------------------------------------------------


class TestParseVariantId:
    def _parse(self, variant_id: str) -> str:
        df = pl.DataFrame({"vid": [variant_id]})
        return df.select(parse_variant_id(pl.col("vid")))["variant_id"][0]

    def test_standard_snv(self) -> None:
        assert self._parse("chr1:12345:A>T") == "1_12345_A_T"

    def test_chrm_converted_to_mt(self) -> None:
        assert self._parse("chrM:999:C>G") == "MT_999_C_G"

    def test_x_chromosome(self) -> None:
        assert self._parse("chrX:5000:G>A") == "X_5000_G_A"

    def test_multichar_alleles(self) -> None:
        assert self._parse("chr2:300:ACT>A") == "2_300_ACT_A"


# ---------------------------------------------------------------------------
# transform_output — integration
# ---------------------------------------------------------------------------


class TestTransformOutput:
    def test_returns_dataframe(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert isinstance(result, pl.DataFrame)

    def test_variant_id_column_present(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert "variantId" in result.columns

    def test_interval_struct_present(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert "interval" in result.columns

    def test_scorer_struct_present(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert "scorer" in result.columns

    def test_variant_id_value(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert result["variantId"][0] == "1_12345_A_T"

    def test_scorer_name_extracted(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert result["scorer"][0]["scorerName"] == "CenterMaskScorer"

    def test_interval_chromosome(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        assert result["interval"][0]["chromosome"] == "1"

    def test_camel_case_columns(self, sample_tidy_df: pl.DataFrame) -> None:
        result = transform_output(sample_tidy_df)
        expected = {
            "variantId", "interval", "scorer", "rawScore", "quantileScore",
            "dataSource", "geneId", "ontologyCurie", "geneSymbol", "geneType",
            "geneStrand", "junctionStart", "junctionEnd", "trackName", "trackStrand",
            "assayTitle", "biosampleName", "biosampleType", "biosampleLifeStage",
            "endedness", "geneticallyModified", "transcriptionFactor",
            "histoneMark", "gtexTissue",
        }
        assert expected == set(result.columns)
