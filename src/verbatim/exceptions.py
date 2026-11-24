"""Errors this package raises on purpose.

Catching `VerbatimError` catches all of them, while real bugs (a KeyError, a
TypeError) still surface as normal tracebacks. The command line relies on that
split: expected errors print one clear line, bugs print the full traceback.
"""


class VerbatimError(Exception):
    """Base class for every expected error in this package."""


class ConfigurationError(VerbatimError):
    """A setting or secret is missing or invalid."""


class DataIntegrityError(VerbatimError):
    """Data does not match what the pipeline expects."""


class DownloadError(VerbatimError):
    """A remote file could not be downloaded (network down, timeout, disk full)."""
