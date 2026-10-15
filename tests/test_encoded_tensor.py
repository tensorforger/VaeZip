import pytest
import torch

import vaezip
from vaezip import EncodedTensor
from vaezip.codecs import (
    AutoCodec,
    BlockRangeQuantCodec,
    CastCodec,
    LloydBlockRangeQuantCodec,
    RangeQuantCodec,
)


def test_encode_attaches_codec_config() -> None:
    codec = LloydBlockRangeQuantCodec(value_bits=3, cache_table=True)
    encoded = codec.encode(torch.randn(1, 2, 8, 8))
    assert encoded.codec_config == codec.get_config()


def test_helpers(device: torch.device) -> None:
    encoded = BlockRangeQuantCodec().encode(torch.randn(2, 4, 8, 8, device=device))
    assert encoded.nbytes > 0
    moved = encoded.to("cpu")
    assert all(tensor.device.type == "cpu" for tensor in moved.tensors.values())
    assert moved.metadata == encoded.metadata


@pytest.mark.parametrize("name", ["cast_fp16", "rq_4bit", "brq_6bit", "lbrq_4bit"])
def test_decode_uses_codec_from_metadata(name: str) -> None:
    latents = torch.randn(2, 8, 24, 24)
    codec = AutoCodec.from_config(name)
    encoded = codec.encode(latents)
    assert torch.equal(encoded.decode(), codec.decode(encoded))


def test_decode_with_explicit_codec() -> None:
    encoded = RangeQuantCodec(bits=8).encode(torch.randn(2, 4, 8, 8))
    expected = encoded.decode()
    assert torch.equal(encoded.decode(codec="rq_8bit"), expected)
    assert torch.equal(encoded.decode(codec=RangeQuantCodec(bits=8)), expected)


def test_decode_with_mismatching_codec_raises() -> None:
    encoded = vaezip.encode(torch.randn(2, 4, 8, 8), codec="rq_8bit")
    with pytest.raises(ValueError, match="does not match"):
        encoded.decode(codec="rq_4bit")
    with pytest.raises(ValueError, match="does not match"):
        encoded.decode(codec="cast_fp16")
    with pytest.raises(ValueError, match="does not match"):
        encoded.decode(codec=RangeQuantCodec(bits=8, channel_dim=2))


def test_decode_comparison_ignores_version_but_not_options() -> None:
    encoded = vaezip.encode(torch.randn(2, 4, 8, 8), codec="lbrq_4bit")
    assert encoded.codec_config is not None
    encoded.metadata["codec_config"] = encoded.metadata["codec_config"].replace(
        encoded.codec_config["_vaezip_version"], "0.0.1"
    )
    encoded.decode(codec="lbrq_4bit")
    with pytest.raises(ValueError, match="does not match"):
        encoded.decode(codec=AutoCodec.from_config("lbrq_4bit", cache_table=True))


def test_decode_without_metadata_requires_codec() -> None:
    latents = torch.randn(2, 4, 8, 8)
    encoded = vaezip.encode(latents, codec="rq_8bit")
    bare = EncodedTensor(tensors=encoded.tensors)
    with pytest.raises(ValueError, match="metadata"):
        bare.decode()
    assert torch.equal(bare.decode(codec="rq_8bit"), encoded.decode())


def test_decode_dtype_and_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16)
    encoded = vaezip.encode(latents, codec="brq_8bit")
    decoded = encoded.decode(dtype=torch.bfloat16, device=device)
    assert decoded.dtype == torch.bfloat16
    assert decoded.device.type == device.type
    assert decoded.shape == latents.shape


def test_decode_device_of_encoded_tensors_is_default(device: torch.device) -> None:
    encoded = vaezip.encode(torch.randn(2, 4, 8, 8, device=device), codec="rq_8bit")
    assert encoded.decode().device.type == device.type


def test_decode_with_codec_instance_state() -> None:
    codec = CastCodec(torch.float16)
    latents = torch.randn(2, 3, 4)
    assert torch.equal(codec.encode(latents).decode(), latents.half().float())


def test_decode_unknown_shortcut_raises() -> None:
    encoded = vaezip.encode(torch.randn(2, 3))
    with pytest.raises(ValueError):
        encoded.decode(codec="nope")


def test_serialization_is_compact() -> None:
    latents = torch.randn(2, 32, 64, 64)
    raw_bytes = latents.numel() * 4
    sizes = {
        name: len(vaezip.save_tensor(latents, codec=name))
        for name in ("cast_fp32", "cast_fp16", "rq_8bit", "brq_4bit", "lbrq_2bit")
    }
    assert sizes["cast_fp32"] == pytest.approx(raw_bytes, rel=0.01)
    assert sizes["cast_fp16"] == pytest.approx(raw_bytes / 2, rel=0.01)
    assert sizes["rq_8bit"] == pytest.approx(raw_bytes / 4, rel=0.01)
    assert sizes["brq_4bit"] == pytest.approx(raw_bytes / 8 * 1.03, rel=0.02)
    assert sizes["lbrq_2bit"] == pytest.approx(raw_bytes / 16 * 1.06, rel=0.03)
