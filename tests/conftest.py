"""Fixtures shared by every test."""

import json
import os

import pandas as pd
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from verbatim import config
from verbatim.analysis import AnalystToolkit
from verbatim.config import DataSettings, Settings
from verbatim.data.schema import RESPONSE_COLUMNS
from verbatim.retrieval import VerbatimIndex, to_documents

# Every variable that can feed Settings: KEYS_FILE, plus each field name, either on its
# own (GROQ_API_KEY) or before "__" (LLM__PROVIDER). Built from the fields, so a secret
# added to Settings later is hidden from tests too.
_SETTINGS_VARIABLES = {"KEYS_FILE", *(name.upper() for name in Settings.model_fields)}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Tests never see the developer's real keys, .env file or GCP project."""
    for name in list(os.environ):
        if name.upper().split("__")[0] in _SETTINGS_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "LOCAL_ENV_FILE", tmp_path / "no.env")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def make_settings():
    """Build settings from explicit values only: `make_settings(groq_api_key="k")`."""

    def _make(**values) -> Settings:
        return Settings(_env_file=None, **values)

    return _make


# ---- A toy processed dataset, for the steps that come after step 1 -------------------

STAFF = "Staff manner & personal attributes"
TIMELINESS = "Timeliness of care"
GRATITUDE = "Positive experience & gratitude"
PARKING = "Parking"
TOY_CODEFRAME = ["Not assigned", STAFF, TIMELINESS, PARKING, GRATITUDE]

_BETTER = "What could we do better?"
_GOOD = "What was good?"
_TOY_ROWS = [
    # comment, respondent, trust, service, question, group, split, answer, themes, sentiment
    (
        "c1",
        "r1",
        "NEAS",
        "PTS - Patient Transport",
        _BETTER,
        "negative",
        "train",
        "The ambulance was two hours late",
        [TIMELINESS],
        4,
    ),
    (
        "c2",
        "r2",
        "NEAS",
        "PTS - Patient Transport",
        _BETTER,
        "positive",
        "train",
        "Nothing.",
        [GRATITUDE],
        1,
    ),
    ("c3", "r3", "NEAS", "111", _GOOD, "positive", "train", "Lovely friendly staff", [STAFF], 1),
    (
        "c4",
        "r4",
        "NHFT",
        "Forensic",
        _GOOD,
        "positive",
        "train",
        "Parking was easy and the staff were kind",
        [PARKING, STAFF],
        2,
    ),
    (
        "c5",
        "r5",
        "NEAS",
        "PTS - Patient Transport",
        _BETTER,
        "negative",
        "dev",
        "Waited two hours for transport home",
        [TIMELINESS],
        5,
    ),
    (
        "c6",
        "r6",
        "NEAS",
        "PTS - Patient Transport",
        _BETTER,
        "positive",
        "dev",
        "Nothing.",
        [GRATITUDE],
        1,
    ),
    (
        "c7",
        "r7",
        "NEAS",
        "PTS - Patient Transport",
        _BETTER,
        "positive",
        "dev",
        "Nothing.",
        [GRATITUDE],
        1,
    ),
    ("c8", "r8", "NHFT", "Forensic", _GOOD, "positive", "test", "Kind staff", [STAFF], 1),
]


@pytest.fixture
def toy_responses() -> pd.DataFrame:
    """Eight processed answers with every column of the real table."""
    columns = ["comment_id", "respondent_id", "trust", "service", "question", "fft_group"]
    columns += ["split", "answer", "human_themes", "human_sentiment"]
    df = pd.DataFrame(_TOY_ROWS, columns=columns)
    df["sub_service"] = pd.NA
    df["date"] = pd.Timestamp("2021-09-30")
    df["fft_score"] = pd.array([4, 1, 1, 2, 5, 1, 1, 1], dtype="Int64")
    df["fft_label"] = "Good"
    df["human_sentiment"] = df["human_sentiment"].astype("Int64")
    return df[RESPONSE_COLUMNS]


@pytest.fixture
def toy_settings(tmp_path, make_settings, toy_responses) -> Settings:
    """Settings whose data folder already holds the toy dataset, as build-data would leave it."""
    settings = make_settings(data=DataSettings(root=tmp_path / "data"))
    settings.data.processed_dir.mkdir(parents=True)
    toy_responses.to_parquet(settings.data.responses_path, index=False)
    codeframe = [{"theme": theme, "n_responses": 1} for theme in TOY_CODEFRAME]
    settings.data.codeframe_path.write_text(json.dumps(codeframe), encoding="utf-8")
    return settings


@pytest.fixture
def toy_index(toy_responses) -> VerbatimIndex:
    """The real index classes over the toy answers, with offline fake embeddings."""
    return VerbatimIndex(to_documents(toy_responses), DeterministicFakeEmbedding(size=32))


@pytest.fixture
def toy_toolkit(toy_responses, toy_index) -> AnalystToolkit:
    return AnalystToolkit(toy_responses, toy_index, min_group_size=1)
