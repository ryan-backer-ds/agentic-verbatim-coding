"""Train/dev/test split for evaluation: by respondent and by hash, so it never leaks or drifts."""

import zlib

import pandas as pd

_BUCKETS = 10_000


def split_by_respondent(
    df: pd.DataFrame, *, test_fraction: float, dev_fraction: float, seed: int
) -> pd.Series:
    """Assign every row to "train", "dev" or "test" by hashing its respondent ID.

    Splitting by respondent rather than by row keeps all answers from one person
    on the same side, so the test set measures performance on unseen people.
    Hashing instead of shuffling keeps assignments stable when new survey waves
    are appended later. Test takes the lowest buckets and dev the next ones, so
    resizing dev never moves anyone into or out of test.
    """
    buckets = (
        df["respondent_id"]
        .astype(str)
        .map(lambda respondent: zlib.crc32(f"{seed}:{respondent}".encode()) % _BUCKETS)
    )
    return (
        pd.Series("train", index=df.index)
        .mask(buckets < (test_fraction + dev_fraction) * _BUCKETS, "dev")
        .mask(buckets < test_fraction * _BUCKETS, "test")
    )
