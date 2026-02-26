"""Schema for VariantIndex input data."""

from enum import StrEnum

import polars as pl


class VariantField(StrEnum):
    """Variant field names."""

    VID = "variantId"
    CHR = "chromosome"
    POS = "position"
    REF = "referenceAllele"
    ALT = "alternateAllele"
    EFF = "variantEffect"
    SEV = "mostSevereConsequenceId"
    CSQ = "transcriptConsequences"
    RID = "rsIds"
    HID = "hgvsId"
    FRQ = "alleleFrequencies"
    XRF = "dbXrefs"
    DSC = "variantDescription"


class VariantEffectField(StrEnum):
    """Variant Effect field names."""

    MTD = "method"
    ASM = "assessment"
    SCR = "score"
    ASF = "assessmentFlag"
    TID = "targetId"
    NSC = "normalisedScore"


class TranscriptConsequenceField(StrEnum):
    """Transcript consequence field names."""

    CID = "variantFunctionalConsequenceIds"
    AAC = "aminoAcidChange"
    UID = "uniprotAccessions"
    IEC = "isEnsemblCanonical"
    CND = "codons"
    DFP = "distanceFromFootprint"
    TSS = "distanceFromTss"
    APR = "appris"
    MAN = "maneSelect"
    TID = "targetId"
    IMP = "impact"
    LOF = "lofteePrediction"
    SIF = "siftPrediction"
    PLP = "polyphenPrediction"
    CSQ = "consequenceScore"
    TIX = "transcriptIndex"
    APS = "approvedSymbol"
    BTP = "biotype"
    TRI = "transcriptId"


class AlleleFrequencyField(StrEnum):
    """Allele frequency field names."""

    POP = "populationName"
    FRQ = "alleleFrequency"


class DbXrefField(StrEnum):
    """Database cross reference field names."""

    XID = "id"
    SRC = "source"


class VariantSchema:
    schema: pl.Schema = pl.Schema(
        {
            VariantField.VID.value: pl.Utf8,
            VariantField.CHR.value: pl.Utf8,
            VariantField.POS.value: pl.Int32,
            VariantField.REF.value: pl.Utf8,
            VariantField.ALT.value: pl.Utf8,
            VariantField.EFF.value: pl.List(
                pl.Struct(
                    [
                        pl.Field(VariantEffectField.MTD.value, pl.Utf8),
                        pl.Field(VariantEffectField.ASM.value, pl.Utf8),
                        pl.Field(VariantEffectField.SCR.value, pl.Float32),
                        pl.Field(VariantEffectField.ASF.value, pl.Utf8),
                        pl.Field(VariantEffectField.TID.value, pl.Utf8),
                        pl.Field(VariantEffectField.NSC.value, pl.Float64),
                    ]
                )
            ),
            VariantField.SEV.value: pl.Utf8,
            VariantField.CSQ.value: pl.List(
                pl.Struct(
                    [
                        pl.Field(TranscriptConsequenceField.CID.value, pl.List(pl.Utf8)),
                        pl.Field(TranscriptConsequenceField.AAC.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.UID.value, pl.List(pl.Utf8)),
                        pl.Field(TranscriptConsequenceField.IEC.value, pl.Boolean),
                        pl.Field(TranscriptConsequenceField.CND.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.DFP.value, pl.Int64),
                        pl.Field(TranscriptConsequenceField.TSS.value, pl.Int64),
                        pl.Field(TranscriptConsequenceField.APR.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.MAN.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.TID.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.IMP.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.LOF.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.SIF.value, pl.Float32),
                        pl.Field(TranscriptConsequenceField.PLP.value, pl.Float32),
                        pl.Field(TranscriptConsequenceField.CSQ.value, pl.Float32),
                        pl.Field(TranscriptConsequenceField.TIX.value, pl.Int32),
                        pl.Field(TranscriptConsequenceField.APS.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.BTP.value, pl.Utf8),
                        pl.Field(TranscriptConsequenceField.TRI.value, pl.Utf8),
                    ]
                )
            ),
            VariantField.RID.value: pl.List(pl.Utf8),
            VariantField.HID.value: pl.Utf8,
            VariantField.FRQ.value: pl.List(
                pl.Struct(
                    [
                        pl.Field(AlleleFrequencyField.POP.value, pl.Utf8),
                        pl.Field(AlleleFrequencyField.FRQ.value, pl.Float64),
                    ]
                )
            ),
            VariantField.XRF.value: pl.List(
                pl.Struct(
                    [
                        pl.Field(DbXrefField.XID.value, pl.Utf8),
                        pl.Field(DbXrefField.SRC.value, pl.Utf8),
                    ]
                )
            ),
            VariantField.DSC.value: pl.Utf8,
        }
    )
