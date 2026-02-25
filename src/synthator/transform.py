"""Transformation utilities for synthator."""

from typing import OrderedDict, Protocol

import polars as pl
from alphagenome.data.genome import Interval, Variant
from alphagenome.models.dna_output import OutputType
from alphagenome.models.variant_scorers import BaseVariantScorer
from anndata import AnnData
from loguru import logger


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


class Formatter(Protocol):
    def format(self, annotation_result: AnnData) -> pl.DataFrame:
        """Format the annotation result into a Polars DataFrame.

        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the formatted annotations for each variant.
        """
        _obs = self._transform_obs(annotation_result)
        _var = self._transform_var(annotation_result)
        _uns = self._transform_uns(annotation_result)
        _x = self._transform_X(annotation_result)

        if _x.is_empty():
            logger.error(f"Annotation failed for {_uns}.")
            return _x

        _x_obs = self._merge_x_and_obs(_x, _obs)
        _x_obs_var = self._merge_x_obs_and_var(_x_obs, _var, _obs)
        if not _obs.is_empty():
            _original_obs_schema = _obs.schema
            _x_obs_var = self._rescue_original_obs(
                fields=_original_obs_schema, _x_obs_var=_x_obs_var
            )
        _final = _x_obs_var.join(_uns, how="cross")

        return _final

    def _rescue_original_obs(
        self, fields: OrderedDict, _x_obs_var: pl.DataFrame
    ) -> pl.DataFrame:
        """Rescue the original obs columns if they were unpivoted during the merge.

        :param fields: List of original obs column names.
        :param _x_obs_var: Polars DataFrame containing the merged X, obs, and var data.

        :return: Polars DataFrame with the original obs columns rescued.
        """
        _split = pl.col("featureName").str.split("|").alias("featureName")
        _original_cols = [
            _split.list.get(i).alias(name) for i, name in enumerate(fields)
        ]
        _x_obs_var_unnested = _x_obs_var.with_columns(_original_cols).with_columns(
            featureName=pl.lit("gene")
        )
        return _x_obs_var_unnested

    def _merge_x_obs_and_var(
        self, _x_obs: pl.DataFrame, _var: pl.DataFrame, _obs: pl.DataFrame
    ) -> pl.DataFrame:
        """Add the var dataframe of the Anndata object to the final Polars DataFrame.

        :param _x_obs: Polars DataFrame containing the transformed X matrix and obs dataframe for each variant.
        :param _var: Polars DataFrame containing the transformed var dataframe for each variant.
        :param _obs: Polars DataFrame containing the transformed obs dataframe for each variant.
        :return: Polars DataFrame containing the var dataframe for each variant.
        """
        _x_obs_var = pl.concat([_var, _x_obs], how="horizontal")
        if not _obs.is_empty():
            _x_obs_var = _x_obs_var.unpivot(
                on=None,
                index=_var.columns,
                variable_name="featureName",
                value_name="featureValue",
            )
        return _x_obs_var

    def _merge_x_and_obs(self, _x: pl.DataFrame, _obs: pl.DataFrame) -> pl.DataFrame:
        """Add the obs dataframe of the Anndata object to the final Polars DataFrame.

        :param _x: Polars DataFrame containing the transformed X matrix for each variant.
        :param _obs: Polars DataFrame containing the transformed obs dataframe for each variant.
        :return: Polars DataFrame containing the obs dataframe for each variant.
        """
        if _obs.is_empty():
            # We have the variant based scores, but no gene observations.
            _x_obs = _x.transpose(column_names=["featureValue"]).with_columns(
                pl.lit("variant").cast(pl.Utf8).alias("featureName")
            )
        else:
            _header = (
                _obs.select(pl.concat_str(pl.col("*"), separator="|").alias("header"))
                .to_dict(as_series=False)
                .get("header", [])
            )
            _header_mapping = {
                f"column_{i}": gene_id for i, gene_id in enumerate(_header)
            }
            _x_obs = _x.transpose().rename(_header_mapping)

        return _x_obs

    def _transform_X(self, annotation_result: AnnData) -> pl.DataFrame:
        """Transform the X matrix of the Anndata object into a Polars DataFrame.

        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the transformed X matrix for each variant.
        """
        if annotation_result.X is None:
            return pl.DataFrame()
        return pl.DataFrame(annotation_result.X)

    def _transform_obs(self, annotation_result: AnnData) -> pl.DataFrame:
        """Transform the obs dataframe of the Anndata object into a Polars DataFrame.

        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the transformed obs dataframe for each variant.
        """
        return pl.DataFrame(annotation_result.obs)

    def _transform_var(self, annotation_result: AnnData) -> pl.DataFrame:
        """Transform the var dataframe of the Anndata object into a Polars DataFrame.

        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the transformed var dataframe for each variant.
        """
        return pl.DataFrame(annotation_result.var)

    def _transform_uns(self, annotation_result: AnnData) -> pl.DataFrame:
        """Transform the uns dictionary of the Anndata object into a Polars DataFrame.

        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the transformed uns dictionary for each variant.
        """
        i: Interval | None = annotation_result.uns.get("interval")
        v: Variant | None = annotation_result.uns.get("variant")
        vs: BaseVariantScorer | None = annotation_result.uns.get("variant_scorer")

        if not i or not v or not vs:
            raise ValueError(
                "Missing interval, variant, or variant scorer information in Anndata uns."
            )
        return pl.DataFrame(
            {
                "variantId": [variant_to_variant_id(v)],
                "interval": [
                    {"start": i.start, "end": i.end, "chromosome": i.chromosome}
                ],
                "variantScorer": [vs.name],
            }
        )


class ATACFormatter(Formatter):
    pass


class CAGEFormatter(Formatter):
    pass


class ChipHistoneFormatter(Formatter):
    pass


class ChipTFFormatter(Formatter):
    pass


class DNASEFormatter(Formatter):
    pass


class ContactMapsFormatter(Formatter):
    pass


class ProcapFormatter(Formatter):
    pass


class RNASeqFormatter(Formatter):
    pass


class SpliceJunctionsFormatter(Formatter):
    pass


class SpliceSiteUsageFormatter(Formatter):
    pass


class SpliceSitesFormatter(Formatter):
    pass


class OutputTypeMapper:
    _mapping_ = {
        OutputType.ATAC: ATACFormatter,
        OutputType.CAGE: CAGEFormatter,
        OutputType.CHIP_HISTONE: ChipHistoneFormatter,
        OutputType.CHIP_TF: ChipTFFormatter,
        OutputType.DNASE: DNASEFormatter,
        OutputType.CONTACT_MAPS: ContactMapsFormatter,
        OutputType.PROCAP: ProcapFormatter,
        OutputType.RNA_SEQ: RNASeqFormatter,
        OutputType.SPLICE_JUNCTIONS: SpliceJunctionsFormatter,
        OutputType.SPLICE_SITE_USAGE: SpliceSiteUsageFormatter,
        OutputType.SPLICE_SITES: SpliceSitesFormatter,
    }

    def format_anndata(
        self, output_type: OutputType, annotation_result: AnnData
    ) -> pl.DataFrame:
        """Format the annotation result into a Polars DataFrame based on the output type.

        :param output_type: The type of output to format.
        :param annotation_result: Anndata object containing the annotations for each variant.

        :return: Polars DataFrame containing the formatted annotations for each variant.
        """
        formatter_cls = self._mapping_.get(output_type)
        if formatter_cls is None:
            raise ValueError(f"Unsupported output type: {output_type}")
        formatter = formatter_cls()
        return formatter.format(annotation_result)
