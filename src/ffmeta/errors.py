"""Exception hierarchy for :mod:`ffmeta`."""

from __future__ import annotations


class FfmetaError(Exception):
    """Base class for all errors raised by ffmeta."""


class FormatError(FfmetaError):
    """The file is not a 64-bit big-endian UM file, or its headers are inconsistent."""


class NoMetadataError(FfmetaError):
    """The file (or section) holds no ffmeta metadata."""


class MetadataExistsError(FfmetaError):
    """The file already holds ffmeta metadata."""


class SectionInUseError(FfmetaError):
    """The requested header section is already used by other data."""


class CorruptMetadataError(FfmetaError):
    """An ffmeta block was found but cannot be decoded."""


class FfmetaWarning(UserWarning):
    """Warnings issued by ffmeta (e.g. a remove that cannot restore the file byte for byte)."""
