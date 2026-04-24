"""Shared post-processing tricks for Fitch OCR pipelines.

This package provides composable, pipeline-agnostic repair functions that
can be imported by both the TFL and modal OCR pipelines. Each trick handles
one narrow concern (symbol normalization, justification snapping, citation
cleanup) so that new repairs can be added and tested independently.

See ``docs/CROSS_PIPELINE_INVESTIGATION.md`` for the architectural
motivation.
"""