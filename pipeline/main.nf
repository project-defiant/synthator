#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

process SCORE_PARTITION {
    tag "${partition.name}"
    label 'synthator'

    publishDir "${params.output}/part_${partition_id}", mode: 'copy', pattern: '*.parquet'

    input:
    path partition

    output:
    path "part_${partition_id}/*.parquet"

    script:
    // Extract the part-0000X from file name
    partition_id = partition.name.replaceAll('^part-(\\d{5}).*parquet$', '$1')
    """
    synthator \\
        --variant-index-path "${partition}" \\
        --api-key "${params.api_key}" \\
        --output "part_${partition_id}" \\
        --batch-window ${params.batch_window} \\
        --test-mode
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------


workflow {
    // Parameter validation (must be inside workflow in DSL2)
    if (!params.variant_index) {
        error("Missing required parameter: --variant_index  (e.g. gs://bucket/dataset/)")
    }
    if (!params.api_key) {
        error("Missing required parameter: --api_key")
    }
    if (!params.output) {
        error("Missing required parameter: --output  (e.g. gs://bucket/results/)")
    }


    // Build channel — one item per Spark partition file
    def index_base = params.variant_index.replaceAll('/+$', '')

    def partitions_ch = channel.fromPath("${index_base}/*.parquet")
        .ifEmpty { error("No part.*.parquet files found at: ${index_base}") }


    result = SCORE_PARTITION(partitions_ch)

    result.subscribe { f -> log.info("Scored partition written: ${f}") }
}
