"""Typed project settings, loaded once and shared by every module.

Values come from four layers; a higher layer wins:

  1. environment variables   how Cloud Run passes config and Secret Manager secrets
  2. the KEYS_FILE file      API keys, kept in a file outside the repository
  3. the local .env file     per-developer settings (git-ignored)
  4. the defaults below      safe values that work out of the box

Nested sections use a double underscore in variable names: LLM__PROVIDER=groq
sets `settings.llm.provider`. Secrets and Google Cloud variables keep their
usual flat names (GROQ_API_KEY, GOOGLE_CLOUD_PROJECT), so key files, Secret
Manager and Google's own libraries all agree on them.

Relative paths (`.env`, `data/`) are relative to the working directory, so run
commands from the project folder. The Docker image sets DATA__ROOT=/data.
"""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from verbatim.exceptions import ConfigurationError

LOCAL_ENV_FILE = Path(".env")


class Provider(StrEnum):
    """Where LLM calls go. `verbatim.llm.providers` builds a model for each."""

    GOOGLE = "google"  # Gemini through Google AI Studio, with an API key (local development)
    VERTEX = "vertex"  # Gemini through Vertex AI, with the runtime's service account (GCP)
    GROQ = "groq"  # open-weights models on Groq, with an API key


class DataSettings(BaseModel):
    """Where the survey data lives and how it is split."""

    root: Path = Path("data")
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    dev_fraction: float = Field(default=0.1, ge=0, lt=1)
    split_seed: int = 42

    @model_validator(mode="after")
    def _leave_room_for_train(self) -> Self:
        if self.test_fraction + self.dev_fraction >= 1:
            raise ValueError("test_fraction + dev_fraction must be below 1")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def responses_path(self) -> Path:
        return self.processed_dir / "responses.parquet"

    @property
    def codeframe_path(self) -> Path:
        return self.processed_dir / "codeframe.json"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.parquet"

    @property
    def coded_dir(self) -> Path:
        return self.root / "coded"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"


class LLMSettings(BaseModel):
    """Which provider to call, and which model on each."""

    provider: Provider = Provider.GOOGLE
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_model: str = "openai/gpt-oss-20b"


class CodingSettings(BaseModel):
    """How the coding agent runs over many answers (step 2)."""

    concurrency: int = Field(default=8, ge=1, le=64)
    few_shot: int = Field(default=3, ge=0, le=10)  # similar coded answers shown per call
    # Client-side budget for model calls; set it below a free-tier quota. None = no limit.
    requests_per_minute: int | None = Field(default=None, ge=1)


class RetrievalSettings(BaseModel):
    """How answers are embedded and searched (step 3)."""

    # hybrid: Gemini embeddings + BM25. keyword: BM25 only, no embedding calls at all
    # (for keys whose embedding quota can't cover the index yet).
    mode: Literal["hybrid", "keyword"] = "hybrid"
    embedding_model: str = "gemini-embedding-001"
    dimensions: int = Field(default=768, ge=128, le=3072)
    top_k: int = Field(default=8, ge=1, le=50)
    candidates: int = Field(default=40, ge=1, le=500)  # per retriever, before fusion
    embed_texts_per_minute: int | None = Field(default=None, ge=1)  # None = no limit


class AnalysisSettings(BaseModel):
    """Rules for the numbers and quotes the analyst agents report (steps 4-5)."""

    min_group_size: int = Field(default=10, ge=1)  # smaller groups are suppressed
    max_revisions: int = Field(default=2, ge=0, le=5)  # redrafts after a failed grounding check


class SafetySettings(BaseModel):
    """Guardrails and tracing (step 7)."""

    # projects/<project>/locations/<location>/templates/<id>; unset means local checks only.
    model_armor_template: str | None = None
    tracing: Literal["off", "console", "cloud_trace"] = "off"


class Settings(BaseSettings):
    """Every setting of the project, grouped by the part of the code that uses it."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",  # the keys file may hold keys for other projects
    )

    data: DataSettings = Field(default_factory=DataSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    coding: CodingSettings = Field(default_factory=CodingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)

    # Secrets. SecretStr keeps them out of logs, reprs and tracebacks.
    google_ai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Google Cloud, used by the "vertex" provider.
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"

    @field_validator("google_ai_api_key", "groq_api_key", "google_cloud_project", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """`GROQ_API_KEY=` in a file means "not set", not "set to an empty key"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


def env_files() -> tuple[Path, ...]:
    """The local .env, then the keys file it points to (later files win)."""
    keys_file = os.getenv("KEYS_FILE") or dotenv_values(LOCAL_ENV_FILE).get("KEYS_FILE")
    if not keys_file:
        return (LOCAL_ENV_FILE,)

    # pydantic-settings silently skips missing files, which would surface later as a
    # confusing "key not set" error. Fail here instead, with the real cause.
    keys_path = Path(keys_file).expanduser()
    if not keys_path.is_file():
        raise ConfigurationError(f"KEYS_FILE points to {keys_path}, which does not exist.")
    return (LOCAL_ENV_FILE, keys_path)


@lru_cache
def get_settings() -> Settings:
    """The settings for this process, read on first use and cached after that."""
    try:
        return Settings(_env_file=env_files())
    except ValidationError as exc:
        # Name each bad value by the variable that sets it, e.g. LLM__PROVIDER. Only the
        # messages are kept: pydantic's full text would echo the input values.
        problems = "; ".join(
            f"{'__'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"Invalid settings: {problems}") from exc
