"""Schema for tidy Anndata output."""

from enum import StrEnum

import polars as pl


class TidyAnndataField(StrEnum):
    """Tidy Anndata output field names."""

    VID = "variantId"
    INT = "interval"
    SCR = "scorer"
    RSC = "rawScore"
    QSC = "quantileScore"
    AST = "assay"
    DSC = "dataSource"
    GID = "geneFromSourceId"
    GSY = "geneFromSourceSymbol"
    BSS = "biosampleFromSource"
    JNC = "junction"
    TRF = "transcriptionFactor"
    HSM = "histoneMark"
    GTX = "gtexTissue"


class BiosampleField(StrEnum):
    """Biosample struct field names."""

    ID = "id"
    NME = "name"
    TYP = "type"
    LST = "lifeStage"


class JunctionField(StrEnum):
    """Junction struct field names."""

    STR = "start"
    END = "end"


class ScorerField(StrEnum):
    """Scorer struct field names."""

    NAM = "name"
    RQO = "requestedOutput"
    WDT = "width"
    AGT = "aggregationType"


class IntervalField(StrEnum):
    """Interval struct field names."""

    CHR = "chr"
    STR = "start"
    END = "end"


class TidyAnndataSchema:
    """Tidy Anndata output schema."""

    schema: pl.Schema = pl.Schema(
        {
            TidyAnndataField.VID.value: pl.Utf8,
            TidyAnndataField.INT.value: pl.Struct(
                [
                    pl.Field(IntervalField.CHR.value, pl.Utf8),
                    pl.Field(IntervalField.STR.value, pl.Int64),
                    pl.Field(IntervalField.END.value, pl.Int64),
                ]
            ),
            TidyAnndataField.SCR.value: pl.Struct(
                [
                    pl.Field(ScorerField.NAM.value, pl.Utf8),
                    pl.Field(ScorerField.RQO.value, pl.Utf8),
                    pl.Field(ScorerField.WDT.value, pl.Int32),
                    pl.Field(ScorerField.AGT.value, pl.Utf8),
                ]
            ),
            TidyAnndataField.RSC.value: pl.Float32,
            TidyAnndataField.QSC.value: pl.Float32,
            TidyAnndataField.AST.value: pl.Utf8,
            TidyAnndataField.DSC.value: pl.Utf8,
            TidyAnndataField.GID.value: pl.Utf8,
            TidyAnndataField.GSY.value: pl.Utf8,
            TidyAnndataField.BSS.value: pl.Struct(
                [
                    pl.Field(BiosampleField.ID.value, pl.Utf8),
                    pl.Field(BiosampleField.NME.value, pl.Utf8),
                    pl.Field(BiosampleField.TYP.value, pl.Utf8),
                    pl.Field(BiosampleField.LST.value, pl.Utf8),
                ]
            ),
            TidyAnndataField.JNC.value: pl.Struct(
                [
                    pl.Field(JunctionField.STR.value, pl.Int64),
                    pl.Field(JunctionField.END.value, pl.Int64),
                ]
            ),
            TidyAnndataField.TRF.value: pl.Utf8,
            TidyAnndataField.HSM.value: pl.Utf8,
            TidyAnndataField.GTX.value: pl.Utf8,
        }
    )
