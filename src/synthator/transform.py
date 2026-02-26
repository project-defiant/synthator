"""Transformation utilities for synthator."""

import polars as pl
from alphagenome.data.genome import Variant

from synthator.output import TidyAnndataSchema


def variant_to_variant_id(v: Variant) -> str:
    """Convert a Variant object to a unique variant ID string.

    The variant ID is constructed in the format: "chr{chromosome}_{position}_{reference_bases}_{alternate_bases}".

    :param v: Variant object containing the variant information.

    :return: A string representing the unique variant ID.
    """
    return f"{ucsc_to_ensembl(v.chromosome)}_{v.position}_{v.reference_bases}_{v.alternate_bases}"


def ensembl_to_ucsc(chromosome: str) -> str:
    """Convert an Ensembl chromosome name to UCSC format.

    Ensembl uses bare names (1, 2, ..., X, Y, MT).
    UCSC uses a chr prefix (chr1, chr2, ..., chrX, chrY, chrM).

    :param chromosome: Ensembl chromosome name.
    :return: UCSC chromosome name.
    """
    if chromosome == "MT":
        return "chrM"
    return f"chr{chromosome}"


def ucsc_to_ensembl(chromosome: str) -> str:
    """Convert a UCSC chromosome name to Ensembl format.

    UCSC uses a chr prefix (chr1, chr2, ..., chrX, chrY, chrM).
    Ensembl uses bare names (1, 2, ..., X, Y, MT).

    :param chromosome: UCSC chromosome name.
    :return: Ensembl chromosome name.
    """
    if chromosome == "chrM":
        return "MT"
    return chromosome.removeprefix("chr")


def scored_interval_to_interval_struct(scored_interval: pl.Expr) -> pl.Expr:
    """Convert a scored interval string to an Interval struct.

    The scored interval string is in the format: "chr{chromosome}_{start}_{end}".

    :param scored_interval: Polars expression containing the scored interval string.

    :return: Polars expression containing the Interval struct.
    """
    regexp = r"(chr\w+):(-?\d+)-(\d+)"
    return pl.struct(
        chromosome=scored_interval.str.extract(regexp, 1).str.replace("chr", "").str.replace("^M$", "MT").cast(pl.Categorical()),
        start=scored_interval.str.extract(regexp, 2).cast(pl.Int64),
        end=scored_interval.str.extract(regexp, 3).cast(pl.Int64),
    ).alias("interval")


def parse_variant_id(variant_id: pl.Expr) -> pl.Expr:
    """Parse a variant ID string into its components.

    The variant ID is in the format: "chr{chromosome}_{position}_{reference_bases}_{alternate_bases}".
    :param variant_id: Polars expression containing the variant ID string.

    :return: Polars expression containing the parsed components of the variant ID.
    """
    regexp = r"(chr\w+):(\d+):(\w+)>(\w+)"
    chr = variant_id.str.extract(regexp, 1).str.replace("chr", "").str.replace("^M$", "MT")
    pos = variant_id.str.extract(regexp, 2)
    ref = variant_id.str.extract(regexp, 3)
    alt = variant_id.str.extract(regexp, 4)
    return pl.concat_str(chr, pos, ref, alt, separator="_").alias("variant_id")


def parse_scorer(scorer: pl.Expr) -> pl.Expr:
    """Parse a variant scorer string into its components.

    Handles patterns like:
      CenterMaskScorer(requested_output=CHIP_TF, width=501, aggregation_type=ACTIVE_SUM)
      GeneMaskActiveScorer(requested_output=RNA_SEQ)
      SpliceJunctionScorer()

    :param scorer: Polars expression containing the scorer string.

    :return: Polars expression containing a struct with fields:
             scorerName, requestedOutput, width, aggregationType.
    """
    scorer_name = scorer.str.extract(r"^(\w+)\(", 1)
    requested_output = scorer.str.extract(r"requested_output=(\w+)", 1)
    width = scorer.str.extract(r"width=(\d+)", 1).cast(pl.Int32)
    aggregation_type = scorer.str.extract(r"aggregation_type=(\w+)", 1)
    return pl.struct(
        scorerName=scorer_name,
        requestedOutput=requested_output,
        width=width,
        aggregationType=aggregation_type,
    ).alias("scorer")


def prepare_biosample(
    ontology_curie: pl.Expr,
    biosample_name: pl.Expr,
    biosample_type: pl.Expr,
    biosample_life_stage: pl.Expr,
) -> pl.Expr:
    """Prepare a biosample struct from its components.

    :param ontology_curie: Polars expression containing the ontology CURIE for the biosample.
    :param biosample_name: Polars expression containing the name of the biosample.
    :param biosample_type: Polars expression containing the type of the biosample.
    :param biosample_life_stage: Polars expression containing the life stage of the biosample.

    :return: Polars expression containing a struct with fields:
             id, name, type, lifeStage.
    """
    return pl.struct(
        id=ontology_curie,
        name=biosample_name,
        type=biosample_type,
        lifeStage=biosample_life_stage,
    ).alias("biosampleFromSource")


def transform_output(tidy_data: pl.DataFrame) -> pl.DataFrame:
    """Transform the tidy output data from the variant scorer into a Polars DataFrame.

    :param tidy_data: Polars DataFrame containing the tidy output data from the variant scorer.

    :return: Polars DataFrame containing the transformed output data.
    """
    parsed_tidy_data = tidy_data.select(
        parse_variant_id(pl.col("variant_id")).alias("variantId"),
        scored_interval_to_interval_struct(pl.col("scored_interval")).alias("interval"),
        parse_scorer(pl.col("variant_scorer")).alias("scorer"),
        pl.col("raw_score").alias("rawScore"),
        pl.col("quantile_score").alias("quantileScore"),
        pl.col("Assay title").alias("assay"),
        pl.col("data_source").alias("dataSource"),
        pl.col("gene_id").alias("geneFromSourceId"),
        pl.col("gene_name").alias("geneFromSourceSymbol"),
        prepare_biosample(
            pl.col("ontology_curie"),
            pl.col("biosample_name"),
            pl.col("biosample_type"),
            pl.col("biosample_life_stage"),
        ).alias("biosampleFromSource"),
        pl.struct(
            pl.col("junction_Start").alias("start"),
            pl.col("junction_End").alias("end"),
        ).alias("junction"),
        pl.col("transcription_factor").alias("transcriptionFactor"),
        pl.col("histone_mark").alias("histoneMark"),
        pl.col("gtex_tissue").alias("gtexTissue"),
    )

    return parsed_tidy_data.cast(TidyAnndataSchema.schema)
