"""Predefined codec configs addressed by short names."""

from typing import Any

CODEC_CLASS_KEY = "_class_name"
VERSION_KEY = "_vaezip_version"
DEFAULT_SHORTCUT = "cast_fp32"

BLOCK_QUANT_BITS = (8, 6, 5, 4, 3, 2)
RANGE_QUANT_BITS = (8, 6, 4)
BLOCK_QUANT_RANGE_BITS = 4
BLOCK_QUANT_BLOCK_EXTENT = 8


def _build_shortcut_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {
        DEFAULT_SHORTCUT: {CODEC_CLASS_KEY: "CastCodec", "cast_dtype": "float32"},
        "cast_fp16": {CODEC_CLASS_KEY: "CastCodec", "cast_dtype": "float16"},
        "cast_bf16": {CODEC_CLASS_KEY: "CastCodec", "cast_dtype": "bfloat16"},
        "cast_fp8_e4m3": {CODEC_CLASS_KEY: "CastCodec", "cast_dtype": "float8_e4m3fn"},
    }
    for bits in RANGE_QUANT_BITS:
        configs[f"rq_{bits}bit"] = {
            CODEC_CLASS_KEY: "RangeQuantCodec",
            "bits": bits,
            "channel_dim": 1,
        }
    block_size = [BLOCK_QUANT_BLOCK_EXTENT, BLOCK_QUANT_BLOCK_EXTENT]
    for prefix, class_name in (
        ("brq", "BlockRangeQuantCodec"),
        ("lbrq", "LloydBlockRangeQuantCodec"),
    ):
        for bits in BLOCK_QUANT_BITS:
            configs[f"{prefix}_{bits}bit"] = {
                CODEC_CLASS_KEY: class_name,
                "value_bits": bits,
                "range_bits": BLOCK_QUANT_RANGE_BITS,
                "channel_dim": 1,
                "block_size": list(block_size),
            }
    return configs


SHORTCUT_CONFIGS = _build_shortcut_configs()


def list_shortcuts() -> list[str]:
    """Names of all predefined codec configs."""
    return list(SHORTCUT_CONFIGS)
