"""Construction of codecs from configs without knowing the codec class."""

from typing import Any

from vaezip.codecs.base import Codec
from vaezip.codecs.block_range_quant import BlockRangeQuantCodec
from vaezip.codecs.cast import CastCodec
from vaezip.codecs.config import ConfigSource, split_config
from vaezip.codecs.lloyd_block_range_quant import LloydBlockRangeQuantCodec
from vaezip.codecs.range_quant import RangeQuantCodec

CODEC_CLASSES: dict[str, type[Codec]] = {
    codec_class.__name__: codec_class
    for codec_class in (
        CastCodec,
        RangeQuantCodec,
        BlockRangeQuantCodec,
        LloydBlockRangeQuantCodec,
    )
}


class AutoCodec:
    """Builds the right codec class from a config."""

    def __init__(self) -> None:
        raise TypeError("AutoCodec is not instantiable, use AutoCodec.from_config()")

    @staticmethod
    def from_config(config: ConfigSource, **overrides: Any) -> Codec:
        """
        args:
            config: path to a JSON config, a shortcut name such as "cast_fp16",
            or a config dict
            overrides: constructor parameters that replace the ones from the config
        """
        class_name, parameters = split_config(config, overrides)
        if class_name is None:
            raise ValueError("Config has no '_class_name' entry")
        if class_name not in CODEC_CLASSES:
            raise ValueError(
                f"Unknown codec class {class_name!r}, available: {list(CODEC_CLASSES)}"
            )
        return CODEC_CLASSES[class_name](**parameters)
