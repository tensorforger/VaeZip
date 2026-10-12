import json
from pathlib import Path

import pytest
import torch

import vaezip
from vaezip.codecs import (
    AutoCodec,
    BlockRangeQuantCodec,
    CastCodec,
    Codec,
    LloydBlockRangeQuantCodec,
    RangeQuantCodec,
    list_shortcuts,
)

EXPECTED_SHORTCUTS = [
    "cast_fp32",
    "cast_fp16",
    "cast_bf16",
    "cast_fp8_e4m3",
    "rq_8bit",
    "rq_6bit",
    "rq_4bit",
    *[f"brq_{bits}bit" for bits in (8, 6, 5, 4, 3, 2)],
    *[f"lbrq_{bits}bit" for bits in (8, 6, 5, 4, 3, 2)],
]


def test_all_shortcuts_exist() -> None:
    assert set(EXPECTED_SHORTCUTS) <= set(list_shortcuts())


@pytest.mark.parametrize("name", EXPECTED_SHORTCUTS)
def test_shortcut_roundtrip(name: str) -> None:
    codec = AutoCodec.from_config(name)
    latents = torch.randn(2, 8, 20, 20)
    decoded = codec.decode(codec.encode(latents))
    assert decoded.shape == latents.shape


def test_shortcut_classes_and_parameters() -> None:
    cast = AutoCodec.from_config("cast_fp16")
    assert isinstance(cast, CastCodec)
    assert cast.cast_dtype == torch.float16
    assert AutoCodec.from_config("cast_bf16").cast_dtype == torch.bfloat16  # type: ignore[attr-defined]
    assert AutoCodec.from_config("cast_fp8_e4m3").cast_dtype == torch.float8_e4m3fn  # type: ignore[attr-defined]
    for bits in (8, 6, 4):
        codec = AutoCodec.from_config(f"rq_{bits}bit")
        assert isinstance(codec, RangeQuantCodec)
        assert codec.bits == bits
        assert codec.channel_dim == 1
    for bits in (8, 6, 5, 4, 3, 2):
        for prefix, expected_class in (
            ("brq", BlockRangeQuantCodec),
            ("lbrq", LloydBlockRangeQuantCodec),
        ):
            codec = AutoCodec.from_config(f"{prefix}_{bits}bit")
            assert type(codec) is expected_class
            assert codec.value_bits == bits  # type: ignore[attr-defined]
            assert codec.channel_dim == 1  # type: ignore[attr-defined]
            assert len(codec.block_size) == 2  # type: ignore[attr-defined]


def test_shortcut_overrides() -> None:
    codec = AutoCodec.from_config("brq_8bit", block_size=(4, 4, 4))
    assert isinstance(codec, BlockRangeQuantCodec)
    assert codec.block_size == (4, 4, 4)
    assert codec.value_bits == 8
    assert AutoCodec.from_config("lbrq_4bit", cache_table=True).get_config()[
        "cache_table"
    ]


def test_shortcut_overrides_do_not_leak_between_calls() -> None:
    AutoCodec.from_config("brq_8bit", block_size=(2, 2))
    assert AutoCodec.from_config("brq_8bit").block_size == (8, 8)  # type: ignore[attr-defined]


def test_unknown_shortcut_and_class() -> None:
    with pytest.raises(ValueError, match="shortcut"):
        AutoCodec.from_config("no_such_codec")
    with pytest.raises(ValueError, match="Unknown codec class"):
        AutoCodec.from_config({"_class_name": "NoSuchCodec"})
    with pytest.raises(ValueError, match="_class_name"):
        AutoCodec.from_config({"bits": 4})
    with pytest.raises(TypeError):
        AutoCodec()


@pytest.mark.parametrize(
    "codec",
    [
        CastCodec(torch.bfloat16),
        CastCodec(torch.float8_e4m3fnuz),
        RangeQuantCodec(bits=5, channel_dim=-1),
        BlockRangeQuantCodec(
            value_bits=3, range_bits=5, channel_dim=2, block_size=(2, 3, 4)
        ),
        LloydBlockRangeQuantCodec(
            value_bits=3, block_size=(2,), lloyd_fast=False, cache_table=True
        ),
    ],
    ids=lambda codec: type(codec).__name__,
)
def test_config_file_roundtrip(codec: Codec, tmp_path: Path) -> None:
    path = tmp_path / "codec.json"
    codec.save_config(path)
    config = json.loads(path.read_text())
    assert config["_class_name"] == type(codec).__name__
    assert config["_vaezip_version"] == vaezip.__version__

    for rebuilt in (type(codec).from_config(path), AutoCodec.from_config(str(path))):
        assert type(rebuilt) is type(codec)
        assert rebuilt.get_config() == codec.get_config()


def test_class_from_config_checks_class_name(tmp_path: Path) -> None:
    path = tmp_path / "codec.json"
    CastCodec().save_config(path)
    with pytest.raises(ValueError):
        RangeQuantCodec.from_config(path)
    assert isinstance(CastCodec.from_config("cast_bf16"), CastCodec)
    with pytest.raises(ValueError):
        CastCodec.from_config("rq_8bit")


def test_class_from_config_without_class_name_and_with_overrides(
    tmp_path: Path,
) -> None:
    path = tmp_path / "codec.json"
    path.write_text(json.dumps({"bits": 3}))
    codec = RangeQuantCodec.from_config(path, channel_dim=-1)
    assert codec.bits == 3
    assert codec.channel_dim == -1


def test_config_dict_is_not_modified() -> None:
    config = {"_class_name": "RangeQuantCodec", "bits": 4}
    AutoCodec.from_config(config, bits=6)
    assert config == {"_class_name": "RangeQuantCodec", "bits": 4}


def test_invalid_config_file(tmp_path: Path) -> None:
    path = tmp_path / "codec.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError):
        AutoCodec.from_config(path)


def test_repr_shows_parameters() -> None:
    assert "bits=4" in repr(RangeQuantCodec(bits=4))
