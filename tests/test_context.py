"""Tests for synthator.context — ContextualizedVariant."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from synthator.context import ContextualizedVariant


class TestContextualizedVariantDataclass:
    def test_has_interval_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(ContextualizedVariant)}
        assert "interval" in fields

    def test_has_variant_field(self) -> None:
        fields = {f.name for f in dataclasses.fields(ContextualizedVariant)}
        assert "variant" in fields

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ContextualizedVariant)

    def test_stores_interval_and_variant(self) -> None:
        interval = MagicMock()
        variant = MagicMock()
        cv = ContextualizedVariant(interval=interval, variant=variant)
        assert cv.interval is interval
        assert cv.variant is variant


class TestContextualizedVariantFromVariant:
    """Tests for the from_variant factory classmethod."""

    @pytest.fixture()
    def mock_variant_cls(self):
        """Patch the Variant class used inside context.py."""
        mock_interval = MagicMock()
        mock_variant_instance = MagicMock()
        mock_variant_instance.reference_interval.resize.return_value = mock_interval

        with patch("synthator.context.Variant", return_value=mock_variant_instance) as mock_cls:
            mock_cls._mock_interval = mock_interval
            mock_cls._mock_instance = mock_variant_instance
            yield mock_cls

    def test_returns_contextualized_variant(self, mock_variant_cls: MagicMock) -> None:
        result = ContextualizedVariant.from_variant(
            chromosome="chrX",
            position=100,
            reference_bases="A",
            alternate_bases="T",
            window_size=1048576,
        )
        assert isinstance(result, ContextualizedVariant)

    def test_variant_and_interval_are_set(self, mock_variant_cls: MagicMock) -> None:
        result = ContextualizedVariant.from_variant(
            chromosome="chrX",
            position=100,
            reference_bases="A",
            alternate_bases="T",
            window_size=1048576,
        )
        assert result.variant is mock_variant_cls._mock_instance
        assert result.interval is mock_variant_cls._mock_interval

    def test_variant_name_format(self, mock_variant_cls: MagicMock) -> None:
        ContextualizedVariant.from_variant(
            chromosome="chrX",
            position=100,
            reference_bases="A",
            alternate_bases="T",
            window_size=1048576,
        )
        _, kwargs = mock_variant_cls.call_args
        assert kwargs["name"] == "chrX_100_A_T"

    def test_with_batch_id_sets_info(self, mock_variant_cls: MagicMock) -> None:
        ContextualizedVariant.from_variant(
            chromosome="chr1",
            position=500,
            reference_bases="C",
            alternate_bases="G",
            window_size=1048576,
            batch_id="batch_42",
        )
        _, kwargs = mock_variant_cls.call_args
        assert kwargs["info"] == {"batchId": "batch_42"}

    def test_without_batch_id_empty_info(self, mock_variant_cls: MagicMock) -> None:
        ContextualizedVariant.from_variant(
            chromosome="chr1",
            position=500,
            reference_bases="C",
            alternate_bases="G",
            window_size=1048576,
        )
        _, kwargs = mock_variant_cls.call_args
        assert kwargs["info"] == {}

    def test_resize_called_with_window_size(self, mock_variant_cls: MagicMock) -> None:
        window = 524288
        ContextualizedVariant.from_variant(
            chromosome="chr1",
            position=500,
            reference_bases="C",
            alternate_bases="G",
            window_size=window,
        )
        mock_variant_cls._mock_instance.reference_interval.resize.assert_called_once_with(width=window)

    def test_chromosome_passed_to_variant(self, mock_variant_cls: MagicMock) -> None:
        ContextualizedVariant.from_variant(
            chromosome="chr2",
            position=200,
            reference_bases="A",
            alternate_bases="C",
            window_size=1048576,
        )
        _, kwargs = mock_variant_cls.call_args
        assert kwargs["chromosome"] == "chr2"

    def test_position_passed_to_variant(self, mock_variant_cls: MagicMock) -> None:
        ContextualizedVariant.from_variant(
            chromosome="chr2",
            position=99999,
            reference_bases="A",
            alternate_bases="C",
            window_size=1048576,
        )
        _, kwargs = mock_variant_cls.call_args
        assert kwargs["position"] == 99999
