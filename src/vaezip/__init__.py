"""VaeZip: compression of VAE latents."""

from vaezip import codecs
from vaezip._version import __version__
from vaezip.api import decode, encode, load, load_tensor, save, save_tensor
from vaezip.encoded_tensor import EncodedTensor

__all__ = [
    "EncodedTensor",
    "__version__",
    "codecs",
    "decode",
    "encode",
    "load",
    "load_tensor",
    "save",
    "save_tensor",
]
