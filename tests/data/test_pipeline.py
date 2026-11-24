import json

import pandas as pd

from verbatim.config import DataSettings
from verbatim.data.pipeline import build_dataset
from verbatim.data.schema import RESPONSE_COLUMNS
from verbatim.data.source import DatasetSource
from verbatim.utils import sha256_of_file


def test_build_dataset_end_to_end(raw_export, tmp_path, make_settings):
    upstream = tmp_path / "upstream.csv"
    raw_export.to_csv(upstream, index=False)
    source = DatasetSource(upstream.as_uri(), sha256_of_file(upstream), "raw.csv")
    settings = make_settings(data=DataSettings(root=tmp_path / "data"))

    summary = build_dataset(settings, source=source)

    assert (summary.raw_rows, summary.responses, summary.respondents) == (5, 2, 2)
    saved = pd.read_parquet(settings.data.responses_path)
    assert list(saved.columns) == RESPONSE_COLUMNS
    assert saved["answer"].tolist()[1] == "Call me on [PHONE]"

    codeframe = json.loads(settings.data.codeframe_path.read_text(encoding="utf-8"))
    assert codeframe == [
        {"theme": "Staff manner & personal attributes", "n_responses": 1},
        {"theme": "Timeliness of care", "n_responses": 2},
    ]
