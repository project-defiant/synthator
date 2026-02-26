"""Tests for synthator.batch — batching, scoring, and I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pandas as pd
import polars as pl
import pytest

from synthator.batch import (
    ContextualizedVariantBatch,
    VariantBatchGenerator,
    annotate_batch,
    batch_output_exists,
    process_batch,
    transform_batch,
    write_batch,
)
from synthator.context import ContextualizedVariant


# ---------------------------------------------------------------------------
# ContextualizedVariantBatch
# ---------------------------------------------------------------------------


class TestContextualizedVariantBatch:
    def test_initial_n_variants_zero(self) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        assert batch.n_variants == 0

    def test_batch_id_stored(self) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chrX_5")
        assert batch.batch_id == "chrX_5"

    def test_interval_variants_stored(self) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        assert batch.interval_variants == []

    def test_append_variant_increments_count(
        self, mock_contextualized_variant: ContextualizedVariant
    ) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        batch.append_variant(mock_contextualized_variant)
        assert batch.n_variants == 1

    def test_append_variant_twice_increments_twice(
        self, mock_contextualized_variant: ContextualizedVariant
    ) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        batch.append_variant(mock_contextualized_variant)
        batch.append_variant(mock_contextualized_variant)
        assert batch.n_variants == 2

    def test_append_variant_adds_to_list(
        self, mock_contextualized_variant: ContextualizedVariant
    ) -> None:
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        batch.append_variant(mock_contextualized_variant)
        assert mock_contextualized_variant in batch.interval_variants

    def test_append_variant_preserves_order(self) -> None:
        iv1 = MagicMock(spec=ContextualizedVariant)
        iv2 = MagicMock(spec=ContextualizedVariant)
        batch = ContextualizedVariantBatch(interval_variants=[], batch_id="chr1_0")
        batch.append_variant(iv1)
        batch.append_variant(iv2)
        assert batch.interval_variants == [iv1, iv2]


# ---------------------------------------------------------------------------
# VariantBatchGenerator
# ---------------------------------------------------------------------------


class TestVariantBatchGeneratorBatchId:
    def test_produces_integer_column(self) -> None:
        df = pl.DataFrame({"rowNumber": [0, 1]})
        result = df.select(VariantBatchGenerator._batch_id(pl.col("rowNumber"), 10))
        assert result["batchId"].dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)

    def test_rows_in_same_window_produce_same_id(self) -> None:
        # Rows 0 and 9 are both in batch 0 for window=10
        df = pl.DataFrame({"rowNumber": [0, 9]})
        result = df.select(VariantBatchGenerator._batch_id(pl.col("rowNumber"), 10))
        assert result["batchId"][0] == result["batchId"][1]

    def test_rows_in_different_windows_produce_different_ids(self) -> None:
        # Row 0 → batch 0, row 10 → batch 1 for window=10
        df = pl.DataFrame({"rowNumber": [0, 10]})
        result = df.select(VariantBatchGenerator._batch_id(pl.col("rowNumber"), 10))
        assert result["batchId"][0] != result["batchId"][1]

    def test_batch_id_is_floor_division(self) -> None:
        # batch = row_number // batch_window
        df = pl.DataFrame({"rowNumber": [0, 5, 10, 15, 20]})
        result = df.select(VariantBatchGenerator._batch_id(pl.col("rowNumber"), 10))
        assert list(result["batchId"]) == [0, 0, 1, 1, 2]


class TestVariantBatchGeneratorBatchVariantIndex:
    @pytest.fixture()
    def small_variant_index(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "chromosome": ["1", "1", "1", "1"],
                "position": [100, 105, 200, 205],
                "referenceAllele": ["A", "C", "G", "T"],
                "alternateAllele": ["T", "G", "A", "C"],
            }
        ).lazy()

    def test_yields_contextualized_variant_batch(self, small_variant_index: pl.LazyFrame) -> None:
        mock_cv = MagicMock(spec=ContextualizedVariant)

        with patch("synthator.batch.ContextualizedVariant.from_variant", return_value=mock_cv):
            batches = list(
                VariantBatchGenerator.batch_variant_index(
                    variant_index=small_variant_index,
                    context_window=1048576,
                    batch_window=100,
                )
            )

        assert len(batches) > 0
        assert all(isinstance(b, ContextualizedVariantBatch) for b in batches)

    def test_total_variants_match_input(self, small_variant_index: pl.LazyFrame) -> None:
        mock_cv = MagicMock(spec=ContextualizedVariant)

        with patch("synthator.batch.ContextualizedVariant.from_variant", return_value=mock_cv):
            batches = list(
                VariantBatchGenerator.batch_variant_index(
                    variant_index=small_variant_index,
                    context_window=1048576,
                    batch_window=100,
                )
            )

        total_variants = sum(len(b.interval_variants) for b in batches)
        assert total_variants == 4

    def test_from_variant_called_for_each_row(self, small_variant_index: pl.LazyFrame) -> None:
        mock_cv = MagicMock(spec=ContextualizedVariant)

        with patch(
            "synthator.batch.ContextualizedVariant.from_variant", return_value=mock_cv
        ) as mock_from:
            list(
                VariantBatchGenerator.batch_variant_index(
                    variant_index=small_variant_index,
                    context_window=1048576,
                    batch_window=100,
                )
            )

        assert mock_from.call_count == 4


# ---------------------------------------------------------------------------
# annotate_batch
# ---------------------------------------------------------------------------


class TestAnnotateBatch:
    @pytest.fixture()
    def mock_client(self):
        mock_result = [[MagicMock()]]
        client = MagicMock()
        client.score_variants.return_value = mock_result
        return client, mock_result

    def test_calls_score_variants(
        self, mock_batch: ContextualizedVariantBatch, mock_client
    ) -> None:
        client, _ = mock_client
        with patch("alphagenome.models.dna_client.create", return_value=client):
            annotate_batch(api_key="test-key", c_variants=mock_batch)

        expected_variants = [cv.variant for cv in mock_batch.interval_variants]
        expected_intervals = [cv.interval for cv in mock_batch.interval_variants]
        client.score_variants.assert_called_once_with(
            variants=expected_variants, intervals=expected_intervals
        )

    def test_returns_score_variants_result(
        self, mock_batch: ContextualizedVariantBatch, mock_client
    ) -> None:
        client, mock_result = mock_client
        with patch("alphagenome.models.dna_client.create", return_value=client):
            result = annotate_batch(api_key="test-key", c_variants=mock_batch)

        assert result is mock_result

    def test_creates_client_with_api_key(
        self, mock_batch: ContextualizedVariantBatch, mock_client
    ) -> None:
        client, _ = mock_client
        with patch("alphagenome.models.dna_client.create", return_value=client) as mock_create:
            annotate_batch(api_key="my-secret-key", c_variants=mock_batch)

        mock_create.assert_called_once_with(api_key="my-secret-key", model_version=ANY)


# ---------------------------------------------------------------------------
# transform_batch
# ---------------------------------------------------------------------------


class TestTransformBatch:
    @pytest.fixture()
    def tidy_pd(self) -> pd.DataFrame:
        """Pandas DataFrame matching real tidy_scores output with all columns required by transform_output."""
        return pd.DataFrame(
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

    def test_returns_polars_dataframe(self, tidy_pd: pd.DataFrame) -> None:
        with patch("synthator.batch.variant_scorers.tidy_scores", return_value=tidy_pd):
            result = transform_batch([[MagicMock()]])

        assert isinstance(result, pl.DataFrame)

    def test_calls_tidy_scores(self, tidy_pd: pd.DataFrame) -> None:
        annotation = [[MagicMock()]]
        with patch(
            "synthator.batch.variant_scorers.tidy_scores", return_value=tidy_pd
        ) as mock_tidy:
            transform_batch(annotation)

        mock_tidy.assert_called_once_with(annotation)

    def test_dataframe_columns_from_transform_output(self, tidy_pd: pd.DataFrame) -> None:
        with patch("synthator.batch.variant_scorers.tidy_scores", return_value=tidy_pd):
            result = transform_batch([[MagicMock()]])

        expected_columns = {
            "variantId",
            "interval",
            "scorer",
            "rawScore",
            "quantileScore",
            "dataSource",
            "geneId",
            "ontologyCurie",
            "geneSymbol",
            "geneType",
            "geneStrand",
            "junctionStart",
            "junctionEnd",
            "trackName",
            "trackStrand",
            "assayTitle",
            "biosampleName",
            "biosampleType",
            "biosampleLifeStage",
            "endedness",
            "geneticallyModified",
            "transcriptionFactor",
            "histoneMark",
            "gtexTissue",
        }
        assert set(result.columns) == expected_columns

    def test_raises_if_tidy_scores_returns_none(self) -> None:
        with patch("synthator.batch.variant_scorers.tidy_scores", return_value=None):
            with pytest.raises(AssertionError, match="No data returned"):
                transform_batch([[MagicMock()]])

    def test_variant_id_parsed_to_string(self, tidy_pd: pd.DataFrame) -> None:
        with patch("synthator.batch.variant_scorers.tidy_scores", return_value=tidy_pd):
            result = transform_batch([[MagicMock()]])

        assert result["variantId"].dtype == pl.String

    def test_scored_interval_parsed_to_struct(self, tidy_pd: pd.DataFrame) -> None:
        with patch("synthator.batch.variant_scorers.tidy_scores", return_value=tidy_pd):
            result = transform_batch([[MagicMock()]])

        assert isinstance(result["interval"].dtype, pl.Struct)


# ---------------------------------------------------------------------------
# write_batch
# ---------------------------------------------------------------------------


class TestWriteBatch:
    def test_creates_parquet_file(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"x": [1, 2, 3]})
        write_batch(df, str(tmp_path), "chr1_0")
        assert (tmp_path / "batch_chr1_0.parquet").exists()

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        df = pl.DataFrame({"x": [1]})
        write_batch(df, str(nested), "test_id")
        assert (nested / "batch_test_id.parquet").exists()

    def test_written_file_is_readable(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"a": [10, 20], "b": ["x", "y"]})
        write_batch(df, str(tmp_path), "my_batch")
        loaded = pl.read_parquet(tmp_path / "batch_my_batch.parquet")
        assert loaded.equals(df)

    def test_batch_id_in_filename(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"x": [1]})
        batch_id = "chrY_special_42"
        write_batch(df, str(tmp_path), batch_id)
        assert (tmp_path / f"batch_{batch_id}.parquet").exists()

    def test_gcs_path_skips_mkdir(self) -> None:
        df = pl.DataFrame({"x": [1]})
        with patch.object(df, "write_parquet"), patch("synthator.batch.Path") as mock_path_cls:
            write_batch(df, "gs://my-bucket/output", "chr1_0")

        mock_path_cls.assert_not_called()

    def test_gcs_path_writes_to_correct_uri(self) -> None:
        df = pl.DataFrame({"x": [1]})
        with patch.object(df, "write_parquet") as mock_write:
            write_batch(df, "gs://my-bucket/output", "chr1_0")

        mock_write.assert_called_once_with("gs://my-bucket/output/batch_chr1_0.parquet")


# ---------------------------------------------------------------------------
# batch_output_exists
# ---------------------------------------------------------------------------


class TestBatchOutputExists:
    def test_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"x": [1]})
        write_batch(df, str(tmp_path), "42")
        assert batch_output_exists(str(tmp_path), "42") is True

    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        assert batch_output_exists(str(tmp_path), "99") is False

    def test_gcs_returns_true_on_success(self) -> None:
        mock_lf = MagicMock()
        mock_lf.limit.return_value.collect.return_value = MagicMock()
        with patch("synthator.batch.pl.scan_parquet", return_value=mock_lf):
            assert batch_output_exists("gs://my-bucket/output", "42") is True

    def test_gcs_returns_false_on_error(self) -> None:
        with patch("synthator.batch.pl.scan_parquet", side_effect=Exception("not found")):
            assert batch_output_exists("gs://my-bucket/output", "42") is False

    def test_gcs_checks_correct_path(self) -> None:
        mock_lf = MagicMock()
        mock_lf.limit.return_value.collect.return_value = MagicMock()
        with patch("synthator.batch.pl.scan_parquet", return_value=mock_lf) as mock_scan:
            batch_output_exists("gs://bucket/out", "7")
        mock_scan.assert_called_once_with("gs://bucket/out/batch_7.parquet")


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_calls_annotate_transform_write_in_order(
        self, mock_batch: ContextualizedVariantBatch
    ) -> None:
        mock_df = pl.DataFrame({"x": [1]})
        call_order: list[str] = []

        def fake_annotate(*args, **kwargs):
            call_order.append("annotate")
            return [[MagicMock()]]

        def fake_transform(*args, **kwargs):
            call_order.append("transform")
            return mock_df

        def fake_write(*args, **kwargs):
            call_order.append("write")

        with (
            patch("synthator.batch.annotate_batch", side_effect=fake_annotate),
            patch("synthator.batch.transform_batch", side_effect=fake_transform),
            patch("synthator.batch.write_batch", side_effect=fake_write),
        ):
            process_batch(api_key="key", c_variants=mock_batch, output_path="/tmp/out")

        assert call_order == ["annotate", "transform", "write"]

    def test_annotate_called_with_correct_args(
        self, mock_batch: ContextualizedVariantBatch
    ) -> None:
        mock_df = pl.DataFrame({"x": [1]})
        with (
            patch("synthator.batch.annotate_batch", return_value=[[MagicMock()]]) as mock_ann,
            patch("synthator.batch.transform_batch", return_value=mock_df),
            patch("synthator.batch.write_batch"),
        ):
            process_batch(api_key="secret", c_variants=mock_batch, output_path="/out")

        mock_ann.assert_called_once_with("secret", mock_batch)

    def test_write_called_with_batch_id(self, mock_batch: ContextualizedVariantBatch) -> None:
        mock_df = pl.DataFrame({"x": [1]})
        with (
            patch("synthator.batch.annotate_batch", return_value=[[MagicMock()]]),
            patch("synthator.batch.transform_batch", return_value=mock_df),
            patch("synthator.batch.write_batch") as mock_write,
        ):
            process_batch(api_key="key", c_variants=mock_batch, output_path="/output")

        mock_write.assert_called_once_with(mock_df, "/output", mock_batch.batch_id)
