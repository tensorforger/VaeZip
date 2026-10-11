"""VaeZip package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vaezip")
except PackageNotFoundError:  # Source checkout that is not installed.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
