"""Codecs for compression of VAE latents."""

from vaezip.codecs.auto import AutoCodec
from vaezip.codecs.base import Codec
from vaezip.codecs.block_range_quant import BlockRangeQuantCodec
from vaezip.codecs.cast import CastCodec
from vaezip.codecs.lloyd_block_range_quant import LloydBlockRangeQuantCodec
from vaezip.codecs.range_quant import RangeQuantCodec
from vaezip.codecs.shortcuts import DEFAULT_SHORTCUT, list_shortcuts

__all__ = [
    "DEFAULT_SHORTCUT",
    "AutoCodec",
    "BlockRangeQuantCodec",
    "CastCodec",
    "Codec",
    "LloydBlockRangeQuantCodec",
    "RangeQuantCodec",
    "list_shortcuts",
]
