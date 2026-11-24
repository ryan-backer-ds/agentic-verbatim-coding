"""Picking a segment of answers, the same way for statistics, search and agent tools."""

from collections.abc import Mapping
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

FftGroup = Literal["positive", "neutral", "negative", "unknown"]
Split = Literal["train", "dev", "test"]

# Text fields match case-insensitively on part of the value ("transport" finds
# "PTS - Patient Transport"); the others must match exactly.
_TEXT_FIELDS = ("trust", "service", "question")


class SegmentFilter(BaseModel):
    """A segment of answers. Fields that are set must all match; unset fields match anything.

    It is a pydantic model so agents can fill it in as structured output and
    tools can expose it as a typed parameter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trust: str | None = Field(default=None, description="NEAS (ambulance) or NHFT (healthcare)")
    service: str | None = Field(
        default=None, description="e.g. 111, PTS - Patient Transport, See and Convey, Forensic"
    )
    question: str | None = Field(default=None, description="e.g. What could we do better?")
    fft_group: FftGroup | None = Field(default=None, description="from the 1-5 FFT rating")
    split: Split | None = None

    def mask(self, responses: pd.DataFrame) -> pd.Series:
        """True for the rows of `responses` inside this segment."""
        keep = pd.Series(True, index=responses.index)
        for field, wanted in self.model_dump(exclude_none=True).items():
            column = responses[field].astype("string").fillna("")
            if field in _TEXT_FIELDS:
                keep &= column.str.contains(wanted, case=False, regex=False)
            else:
                keep &= column.eq(wanted)
        return keep

    def matches(self, record: Mapping[str, object]) -> bool:
        """The same rule as `mask`, for one record such as a document's metadata."""
        for field, wanted in self.model_dump(exclude_none=True).items():
            value = str(record.get(field) or "")
            if field in _TEXT_FIELDS:
                if wanted.casefold() not in value.casefold():
                    return False
            elif value != wanted:
                return False
        return True

    def describe(self) -> str:
        """A short label such as "trust=NEAS, fft_group=negative" (or "all answers")."""
        fields = self.model_dump(exclude_none=True)
        return ", ".join(f"{name}={value}" for name, value in fields.items()) or "all answers"
