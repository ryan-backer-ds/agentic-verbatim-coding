import pandas as pd

from verbatim.data.splitting import split_by_respondent


def _respondents(n, answers_each=1):
    return pd.DataFrame({"respondent_id": [f"r{i}" for i in range(n) for _ in range(answers_each)]})


def _split(df, *, test_fraction=0.2, dev_fraction=0.1, seed=42):
    return split_by_respondent(
        df, test_fraction=test_fraction, dev_fraction=dev_fraction, seed=seed
    )


def test_split_keeps_respondents_together_and_is_stable():
    df = _respondents(1000, answers_each=2)

    split = _split(df)

    assert split.equals(_split(df))
    assert df.assign(split=split).groupby("respondent_id")["split"].nunique().eq(1).all()
    shares = split.value_counts(normalize=True)
    assert 0.15 < shares["test"] < 0.25
    assert 0.05 < shares["dev"] < 0.15


def test_appending_respondents_never_moves_existing_ones():
    first_wave = _respondents(1000)
    both_waves = _respondents(2000)

    assert _split(both_waves).iloc[:1000].equals(_split(first_wave))


def test_resizing_dev_never_changes_test():
    df = _respondents(1000)

    without_dev = _split(df, dev_fraction=0).eq("test")
    with_dev = _split(df, dev_fraction=0.15).eq("test")

    assert without_dev.equals(with_dev)


def test_split_changes_with_the_seed():
    df = _respondents(1000)

    assert not _split(df, seed=1).equals(_split(df, seed=2))
