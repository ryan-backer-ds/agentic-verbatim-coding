"""Step 1: the survey dataset.

- source.py: fetch and load the pinned raw export
- cleaning.py: one tidy, PII-scrubbed row per answer (pure functions)
- splitting.py: train/test by respondent
- schema.py: column names, allowed values and the final validation
- pipeline.py: build_dataset() runs the steps above and saves the result
"""

from verbatim.data.pipeline import DatasetSummary, build_dataset

__all__ = ["DatasetSummary", "build_dataset"]
