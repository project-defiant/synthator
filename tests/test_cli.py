"""Tests for the synthator CLI (synthator/__init__.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from synthator import app
from synthator.batch import ContextualizedVariantBatch, VariantBatchGenerator

# Default runner (catches exceptions) for error-state tests.
runner = CliRunner()
# Strict runner: surfaces real exceptions so failures are easier to diagnose.
strict_runner = CliRunner(catch_exceptions=False)


def _make_mock_batches(n: int) -> list[ContextualizedVariantBatch]:
    batches = []
    for i in range(n):
        b = MagicMock(spec=ContextualizedVariantBatch)
        b.n_variants = 2
        b.batch_id = f"chr1_{i}"
        batches.append(b)
    return batches


# ---------------------------------------------------------------------------
# Help / basic invocation
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_help_shows_options() -> None:
    # In Typer 0.12+ a single @app.command is the root command; no subcommand prefix.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "variant-index-path" in result.output


def test_cli_missing_required_args_exits_nonzero() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_patches(tmp_path: Path):
    """Patch all external I/O dependencies of the CLI command."""
    mock_lazyframe = MagicMock()
    dummy_index = tmp_path / "variants.parquet"
    dummy_index.touch()

    with (
        patch("synthator.pl.scan_parquet", return_value=mock_lazyframe),
        patch("synthator.process_batch") as mock_process,
    ):
        yield dummy_index, mock_process


def _invoke(dummy_index: Path, extra_args: list[str] | None = None, *, strict: bool = True):
    args = ["--variant-index-path", str(dummy_index), "--api-key", "test-key"]
    if extra_args:
        args.extend(extra_args)
    r = strict_runner if strict else runner
    return r.invoke(app, args)


# ---------------------------------------------------------------------------
# test_mode behaviour
# ---------------------------------------------------------------------------


class TestCliTestMode:
    def test_test_mode_processes_exactly_two_batches(self, cli_patches: tuple) -> None:
        dummy_index, mock_process = cli_patches
        batches = _make_mock_batches(5)

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter(batches),
        ):
            result = _invoke(dummy_index, ["--test-mode"])

        assert result.exit_code == 0, f"CLI error: {result.exception!r}\n{result.output}"
        assert mock_process.call_count == 2

    def test_no_test_mode_processes_all_batches(self, cli_patches: tuple) -> None:
        dummy_index, mock_process = cli_patches
        batches = _make_mock_batches(4)

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter(batches),
        ):
            result = _invoke(dummy_index, ["--no-test-mode"])

        assert result.exit_code == 0, f"CLI error: {result.exception!r}\n{result.output}"
        assert mock_process.call_count == 4

    def test_fewer_than_two_batches_processes_all(self, cli_patches: tuple) -> None:
        dummy_index, mock_process = cli_patches
        batches = _make_mock_batches(1)

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter(batches),
        ):
            _invoke(dummy_index, ["--test-mode"])

        assert mock_process.call_count == 1


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


class TestCliArguments:
    def test_api_key_passed_to_process_batch(self, cli_patches: tuple) -> None:
        dummy_index, mock_process = cli_patches
        batches = _make_mock_batches(1)

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter(batches),
        ):
            strict_runner.invoke(
                app,
                [
                    "--variant-index-path",
                    str(dummy_index),
                    "--api-key",
                    "my-real-key",
                    "--no-test-mode",
                ],
            )

        assert mock_process.call_args.kwargs["api_key"] == "my-real-key"

    def test_output_path_passed_to_process_batch(self, cli_patches: tuple) -> None:
        dummy_index, mock_process = cli_patches
        batches = _make_mock_batches(1)
        custom_output = "/custom/output/path"

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter(batches),
        ):
            strict_runner.invoke(
                app,
                [
                    "--variant-index-path",
                    str(dummy_index),
                    "--api-key",
                    "key",
                    "--output",
                    custom_output,
                    "--no-test-mode",
                ],
            )

        assert mock_process.call_args.kwargs["output_path"] == custom_output

    def test_batch_window_passed_to_generator(self, cli_patches: tuple) -> None:
        dummy_index, _ = cli_patches

        with patch.object(
            VariantBatchGenerator,
            "batch_variant_index",
            side_effect=lambda *a, **kw: iter([]),
        ) as mock_gen:
            strict_runner.invoke(
                app,
                [
                    "--variant-index-path",
                    str(dummy_index),
                    "--api-key",
                    "key",
                    "--batch-window",
                    "25",
                    "--no-test-mode",
                ],
            )

        assert mock_gen.call_args.kwargs["batch_window"] == 25
