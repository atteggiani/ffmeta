"""ffmeta: embed free-text metadata in UM fieldsfiles and ancillary files.

The metadata lives in the otherwise unused ``temp_historyfile`` header
section as a self-describing ASCII JSON block appended to the end of the file;
only that section's two pointer words change, so ``remove`` restores the
original file byte for byte.
"""

from ._verify import Check, Report, verify
from importlib.metadata import version
from .core import add, edit, from_mule, has_metadata, read, remove
from .errors import (
    CorruptMetadataError,
    FfmetaError,
    FfmetaWarning,
    FormatError,
    MetadataExistsError,
    NoMetadataError,
    SectionInUseError,
)

__version__ = version(__package__)

__all__ = [
    "__version__",
    "add",
    "read",
    "edit",
    "remove",
    "has_metadata",
    "from_mule",
    "verify",
    "Report",
    "Check",
    "FfmetaError",
    "FfmetaWarning",
    "FormatError",
    "NoMetadataError",
    "MetadataExistsError",
    "SectionInUseError",
    "CorruptMetadataError",
]
