"""VaeZip: compression of VAE latents."""

from vaezip import codecs
from vaezip._version import __version__
from vaezip.api import load, save
from vaezip.encoded_tensor import EncodedTensor, deserialize, serialize

__all__ = [
    "EncodedTensor",
    "__version__",
    "codecs",
    "deserialize",
    "load",
    "save",
    "serialize",
]
