from pathlib import Path

import pytest
import torch

import vaezip
from vaezip.codecs import AutoCodec, BlockRangeQuantCodec, CastCodec


def test_encode_default_codec_is_lossless_fp32() -> None:
    latents = torch.randn(2, 3, 4, 4)
    assert torch.equal(vaezip.decode(vaezip.encode(latents)), latents)


@pytest.mark.parametrize("name", ["cast_fp16", "rq_4bit", "brq_6bit", "lbrq_4bit"])
def test_encode_with_shortcut_matches_codec(name: str) -> None:
    latents = torch.randn(2, 8, 24, 24)
    codec = AutoCodec.from_config(name)
    assert torch.equal(
        vaezip.encode(latents, codec=name).decode(),
        codec.decode(codec.encode(latents)),
    )


def test_encode_with_codec_object_and_config_path(tmp_path: Path) -> None:
    latents = torch.randn(2, 4, 12, 12)
    codec = BlockRangeQuantCodec(value_bits=6, block_size=(3, 3))
    config_path = tmp_path / "codec.json"
    codec.save_config(config_path)
    from_object = vaezip.encode(latents, codec=codec)
    from_path = vaezip.encode(latents, codec=str(config_path))
    assert from_object.metadata == from_path.metadata
    assert torch.equal(from_object.decode(), from_path.decode())


def test_encode_unknown_shortcut_raises() -> None:
    with pytest.raises(ValueError):
        vaezip.encode(torch.randn(2, 3), codec="nope")


def test_decode_function_matches_method() -> None:
    encoded = vaezip.encode(torch.randn(2, 4, 8, 8), codec="rq_8bit")
    assert torch.equal(
        vaezip.decode(encoded, dtype=torch.float16), encoded.decode(dtype=torch.float16)
    )


def test_save_returns_bytes_and_load_returns_dict() -> None:
    latents = torch.randn(4, 32, 16, 16)
    data = vaezip.save({"latents": vaezip.encode(latents, codec="cast_fp16")})
    assert isinstance(data, bytes)
    loaded = vaezip.load(data)
    assert list(loaded) == ["latents"]
    assert torch.allclose(loaded["latents"].decode(), latents, atol=1e-2)


def test_save_and_load_file_with_different_codecs(tmp_path: Path) -> None:
    tensor_a = torch.randn(2, 8, 16, 16)
    tensor_b = torch.randn(2, 16, 32)
    codec_b = BlockRangeQuantCodec(value_bits=8, block_size=(8,))
    path = tmp_path / "batch.vz.safetensors"
    assert (
        vaezip.save(
            {
                "tensor_b": vaezip.encode(tensor_b, codec=codec_b),
                "tensor_a": vaezip.encode(tensor_a, codec="lbrq_4bit"),
            },
            path,
        )
        is None
    )
    loaded = vaezip.load(path)
    assert list(loaded) == ["tensor_a", "tensor_b"]
    assert loaded["tensor_a"].codec_config is not None
    assert loaded["tensor_a"].codec_config["_class_name"] == "LloydBlockRangeQuantCodec"
    assert loaded["tensor_b"].codec_config is not None
    assert loaded["tensor_b"].codec_config["_class_name"] == "BlockRangeQuantCodec"
    assert loaded["tensor_a"].decode().shape == tensor_a.shape
    assert torch.allclose(loaded["tensor_b"].decode(), tensor_b, atol=0.05)
    assert torch.equal(
        vaezip.load(str(path))["tensor_b"].decode(), loaded["tensor_b"].decode()
    )


def test_roundtrip_is_equal_to_direct_decode() -> None:
    latents = torch.randn(2, 8, 24, 24)
    codec = AutoCodec.from_config("lbrq_4bit")
    expected = codec.decode(codec.encode(latents))
    loaded = vaezip.load(vaezip.save({"latents": codec.encode(latents)}))
    assert torch.equal(loaded["latents"].decode(), expected)


def test_names_with_delimiter_roundtrip() -> None:
    latents = torch.randn(2, 4, 8, 8)
    names = ["a.b", "a", "model.layer.0.latents"]
    data = vaezip.save({name: vaezip.encode(latents) for name in names})
    loaded = vaezip.load(data)
    assert list(loaded) == sorted(names)
    for encoded in loaded.values():
        assert torch.equal(encoded.decode(), latents)


@pytest.mark.parametrize("suffix", [".pth", ".pt", ".bin"])
def test_pth_format_is_explicit(tmp_path: Path, suffix: str) -> None:
    latents = torch.randn(2, 4, 8, 8)
    path = tmp_path / f"latents{suffix}"
    vaezip.save({"latents": vaezip.encode(latents)}, path, format="pth")
    loaded = vaezip.load(path, format="pth")
    assert torch.equal(loaded["latents"].decode(), latents)
    with pytest.raises(ValueError, match="safetensors"):
        vaezip.load(path)


def test_safetensors_is_default_regardless_of_suffix(tmp_path: Path) -> None:
    latents = torch.randn(2, 4, 8, 8)
    path = tmp_path / "latents.pth"
    vaezip.save({"latents": vaezip.encode(latents)}, path)
    assert torch.equal(vaezip.load(path)["latents"].decode(), latents)
    with pytest.raises(ValueError, match="pth"):
        vaezip.load(path, format="pth")


def test_pth_bytes_roundtrip() -> None:
    latents = torch.randn(2, 4, 8, 8)
    data = vaezip.save({"latents": vaezip.encode(latents)}, format="pth")
    assert torch.equal(vaezip.load(data, format="pth")["latents"].decode(), latents)


def test_load_device_and_decode_options(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16)
    data = vaezip.save({"latents": vaezip.encode(latents, codec="brq_8bit")})
    encoded = vaezip.load(data, device=device)["latents"]
    assert all(tensor.device.type == device.type for tensor in encoded.tensors.values())
    decoded = encoded.decode(dtype=torch.bfloat16)
    assert decoded.dtype == torch.bfloat16
    assert decoded.device.type == device.type
    assert decoded.shape == latents.shape


def test_save_from_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16, device=device)
    data = vaezip.save({"latents": vaezip.encode(latents, codec="lbrq_6bit")})
    decoded = vaezip.load(data, device=device)["latents"].decode()
    assert torch.allclose(decoded.cpu(), latents.cpu(), atol=0.2)


def test_save_rejects_tensor_and_single_encoded_tensor() -> None:
    latents = torch.randn(2, 3, 4, 4)
    with pytest.raises(TypeError):
        vaezip.save(latents)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        vaezip.save(vaezip.encode(latents))  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        vaezip.save({"latents": latents})  # type: ignore[dict-item]


def test_save_rejects_invalid_names_and_empty_dict() -> None:
    encoded = vaezip.encode(torch.randn(2, 3, 4, 4))
    with pytest.raises(ValueError):
        vaezip.save({})
    with pytest.raises(ValueError):
        vaezip.save({"": encoded})
    with pytest.raises(ValueError):
        vaezip.save({1: encoded})  # type: ignore[dict-item]


def test_save_rejects_unsupported_format() -> None:
    encoded = vaezip.encode(torch.randn(2, 3, 4, 4))
    with pytest.raises(ValueError, match="Unsupported format"):
        vaezip.save({"latents": encoded}, format="npz")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="Unsupported format"):
        vaezip.load(b"", format="npz")  # type: ignore[arg-type]


def test_save_tensor_and_load_tensor(tmp_path: Path) -> None:
    latents = torch.randn(2, 8, 16, 16)
    path = tmp_path / "latents.vz.safetensors"
    assert vaezip.save_tensor(latents, path, codec="lbrq_4bit") is None
    decoded = vaezip.load_tensor(path)
    codec = AutoCodec.from_config("lbrq_4bit")
    assert torch.equal(decoded, codec.decode(codec.encode(latents)))


def test_save_tensor_default_codec_and_bytes() -> None:
    latents = torch.randn(2, 3, 4, 4)
    data = vaezip.save_tensor(latents)
    assert isinstance(data, bytes)
    assert torch.equal(vaezip.load_tensor(data), latents)


def test_save_tensor_is_loadable_with_load() -> None:
    latents = torch.randn(2, 4, 8, 8)
    loaded = vaezip.load(vaezip.save_tensor(latents, codec="rq_8bit"))
    assert list(loaded) == ["tensor"]
    assert torch.allclose(loaded["tensor"].decode(), latents, atol=0.05)


def test_load_tensor_options(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16)
    data = vaezip.save_tensor(latents, codec="brq_8bit", format="pth")
    decoded = vaezip.load_tensor(data, dtype=torch.float16, device=device, format="pth")
    assert decoded.dtype == torch.float16
    assert decoded.device.type == device.type


def test_load_tensor_with_explicit_codec() -> None:
    latents = torch.randn(2, 4, 8, 8)
    data = vaezip.save_tensor(latents, codec="rq_8bit")
    assert torch.equal(
        vaezip.load_tensor(data, codec="rq_8bit"), vaezip.load_tensor(data)
    )
    with pytest.raises(ValueError, match="does not match"):
        vaezip.load_tensor(data, codec="rq_4bit")


def test_load_tensor_rejects_multiple_entries() -> None:
    encoded = vaezip.encode(torch.randn(2, 4, 8, 8))
    data = vaezip.save({"tensor": encoded, "other": encoded})
    with pytest.raises(ValueError, match="single entry"):
        vaezip.load_tensor(data)


def test_load_tensor_rejects_other_key_name() -> None:
    data = vaezip.save({"latents": vaezip.encode(torch.randn(2, 4, 8, 8))})
    with pytest.raises(ValueError, match="latents"):
        vaezip.load_tensor(data)


def test_package_exports() -> None:
    assert vaezip.codecs.CastCodec is CastCodec
    assert isinstance(vaezip.__version__, str)
    for name in ("encode", "decode", "save", "load", "save_tensor", "load_tensor"):
        assert callable(getattr(vaezip, name))
    for name in ("serialize", "deserialize"):
        assert not hasattr(vaezip, name)
    assert not hasattr(vaezip.EncodedTensor, "serialize")
    assert not hasattr(vaezip.EncodedTensor, "deserialize")
