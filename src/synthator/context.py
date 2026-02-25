from __future__ import annotations

from dataclasses import dataclass

from alphagenome.data.genome import Interval, Variant


@dataclass
class ContextualizedVariant:
    """Interval-variant pair."""

    interval: Interval
    """Interval representing the genomic context around the variant."""
    variant: Variant
    """Variant information."""

    @classmethod
    def from_variant(
        cls,
        chromosome: str,
        position: int,
        reference_bases: str,
        alternate_bases: str,
        window_size: int,
        batch_id: str | None = None,
    ) -> ContextualizedVariant:
        """Create an IntervalVariant from variant information.

        :param chromosome: Chromosome of the variant.
        :param position: Position of the variant.
        :param reference_bases: Reference bases of the variant.
        :param alternate_bases: Alternate bases of the variant.
        :param window_size: Size of the context window around the variant.
        :param batch_id: Optional batch ID for the variant.

        :return: ContextualizedVariant instance.
        """
        variant = Variant(
            chromosome=chromosome,
            position=position,
            reference_bases=reference_bases,
            alternate_bases=alternate_bases,
            name=f"{chromosome}_{position}_{reference_bases}_{alternate_bases}",
            info={"batchId": batch_id} if batch_id else {},
        )
        interval = variant.reference_interval.resize(width=window_size)
        return cls(interval=interval, variant=variant)
