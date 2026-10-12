from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError

import vaezip
from vaezip.codecs import AutoCodec, BlockRangeQuantCodec, CastCodec, RangeQuantCodec


def test_save_returns_bytes_and_load_roundtrips() -> None:
    latents = torch.randn(4, 32, 16, 16)
    data = vaezip.save(latents, codec="cast_fp16")
    assert isinstance(data, bytes)
    assert torch.allclose(vaezip.load(data), latents, atol=1e-2)


def test_default_codec_is_lossless_fp32() -> None:
    latents = torch.randn(2, 3, 4, 4)
    assert torch.equal(vaezip.load(vaezip.save(latents)), latents)


@pytest.mark.parametrize("suffix", [".vz.safetensors", ".pth", ".pt"])
def test_save_and_load_file(tmp_path: Path, suffix: str) -> None:
    latents = torch.randn(2, 4, 16, 16)
    path = tmp_path / f"latents{suffix}"
    assert vaezip.save(latents, path, codec="rq_8bit") is None
    assert path.exists()
    decoded = vaezip.load(str(path))
    assert decoded.shape == latents.shape
    assert torch.allclose(decoded, latents, atol=0.05)
    assert torch.equal(vaezip.load(path), decoded)


def test_file_content_matches_requested_format(tmp_path: Path) -> None:
    latents = torch.randn(2, 4, 8, 8)
    path = tmp_path / "latents.bin"
    vaezip.save(latents, path, format="pth")
    assert torch.equal(vaezip.load(path, format="pth"), latents)
    with pytest.raises(SafetensorError):
        vaezip.load(path)


def test_bytes_format_argument() -> None:
    latents = torch.randn(2, 4, 8, 8)
    data = vaezip.save(latents, format="pth")
    assert torch.equal(vaezip.load(data, format="pth"), latents)


@pytest.mark.parametrize("name", ["cast_fp16", "rq_4bit", "brq_6bit", "lbrq_4bit"])
def test_load_uses_codec_from_metadata(name: str) -> None:
    latents = torch.randn(2, 8, 24, 24)
    codec = AutoCodec.from_config(name)
    data = vaezip.save(latents, codec=codec)
    assert torch.equal(vaezip.load(data), codec.decode(codec.encode(latents)))


def test_save_with_codec_object_and_config_path(tmp_path: Path) -> None:
    latents = torch.randn(2, 4, 12, 12)
    codec = BlockRangeQuantCodec(value_bits=6, block_size=(3, 3))
    config_path = tmp_path / "codec.json"
    codec.save_config(config_path)
    from_object = vaezip.save(latents, codec=codec)
    from_path = vaezip.save(latents, codec=str(config_path))
    assert torch.equal(vaezip.load(from_object), vaezip.load(from_path))


def test_load_with_explicit_codec() -> None:
    latents = torch.randn(2, 4, 8, 8)
    data = vaezip.save(latents, codec="rq_8bit")
    assert torch.equal(vaezip.load(data, codec="rq_8bit"), vaezip.load(data))
    assert torch.equal(
        vaezip.load(data, codec=RangeQuantCodec(bits=8)), vaezip.load(data)
    )


def test_load_with_mismatching_codec_raises() -> None:
    data = vaezip.save(torch.randn(2, 4, 8, 8), codec="rq_8bit")
    with pytest.raises(ValueError, match="does not match"):
        vaezip.load(data, codec="rq_4bit")
    with pytest.raises(ValueError, match="does not match"):
        vaezip.load(data, codec="cast_fp16")
    with pytest.raises(ValueError, match="does not match"):
        vaezip.load(data, codec=RangeQuantCodec(bits=8, channel_dim=2))


def test_load_comparison_is_strict_about_encoding_options() -> None:
    data = vaezip.save(torch.randn(2, 4, 8, 8), codec="lbrq_4bit")
    with pytest.raises(ValueError, match="does not match"):
        vaezip.load(data, codec=AutoCodec.from_config("lbrq_4bit", cache_table=True))


def test_load_without_metadata_requires_codec() -> None:
    latents = torch.randn(2, 4, 8, 8)
    data = vaezip.save(latents, codec="rq_8bit", save_metadata=False)
    with pytest.raises(ValueError, match="metadata"):
        vaezip.load(data)
    decoded = vaezip.load(data, codec="rq_8bit")
    assert torch.equal(decoded, vaezip.load(vaezip.save(latents, codec="rq_8bit")))


def test_load_without_metadata_with_wrong_codec_does_not_check() -> None:
    data = vaezip.save(torch.randn(2, 4, 8, 8), codec="rq_8bit", save_metadata=False)
    with pytest.raises((KeyError, ValueError)):
        vaezip.load(data, codec="brq_4bit")


def test_load_dtype_and_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16)
    data = vaezip.save(latents, codec="brq_8bit")
    decoded = vaezip.load(data, dtype=torch.bfloat16, device=device)
    assert decoded.dtype == torch.bfloat16
    assert decoded.device.type == device.type
    assert decoded.shape == latents.shape


def test_save_from_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16, device=device)
    data = vaezip.save(latents, codec="lbrq_6bit")
    decoded = vaezip.load(data, device=device)
    assert torch.allclose(decoded.cpu(), latents.cpu(), atol=0.2)


def test_save_accepts_codec_instance_state() -> None:
    codec = CastCodec(torch.float16)
    latents = torch.randn(2, 3, 4)
    assert torch.equal(
        vaezip.load(vaezip.save(latents, codec=codec)), latents.half().float()
    )


def test_unknown_shortcut_in_save_and_load() -> None:
    with pytest.raises(ValueError):
        vaezip.save(torch.randn(2, 3), codec="nope")
    data = vaezip.save(torch.randn(2, 3))
    with pytest.raises(ValueError):
        vaezip.load(data, codec="nope")


def test_package_exports() -> None:
    assert vaezip.codecs.CastCodec is CastCodec
    assert isinstance(vaezip.__version__, str)
