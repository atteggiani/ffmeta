"""ffmeta: embed free-text metadata in UM fieldsfiles and ancillary files.

The metadata lives in an otherwise unused header section (``extra_constants``
by default) as a self-describing ASCII JSON block appended to the end of the
file; only the two pointer words of that section change, so the file stays
well-formed according to UMDP F03 and ``remove`` restores it byte for byte.
"""

from ._verify import Check, Report, verify
from ._version import __version__
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
