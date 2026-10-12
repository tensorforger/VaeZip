import struct

import pytest
import torch

import vaezip
from vaezip import EncodedTensor
from vaezip.codecs import BlockRangeQuantCodec, CastCodec, LloydBlockRangeQuantCodec


@pytest.mark.parametrize("format", ["safetensors", "pth"])
def test_serialize_roundtrip(format: str) -> None:
    encoded = EncodedTensor(
        tensors={
            "a": torch.randn(2, 3),
            "b": torch.randint(0, 255, (4,), dtype=torch.uint8),
            "c": torch.randn(3).to(torch.float8_e4m3fn),
            "d": torch.randn(2).to(torch.bfloat16),
        },
        metadata={"key": "value"},
    )
    restored = vaezip.deserialize(
        vaezip.serialize(encoded, format=format), format=format
    )
    assert restored.metadata == {"key": "value"}
    assert restored.tensors.keys() == encoded.tensors.keys()
    for name, tensor in encoded.tensors.items():
        assert restored.tensors[name].dtype == tensor.dtype
        assert torch.equal(restored.tensors[name].float(), tensor.float())


@pytest.mark.parametrize("format", ["safetensors", "pth"])
def test_save_metadata_false_drops_metadata(format: str) -> None:
    encoded = CastCodec().encode(torch.randn(2, 3))
    assert encoded.codec_config is not None
    with_metadata = encoded.serialize(format=format)
    without_metadata = encoded.serialize(save_metadata=False, format=format)
    assert (
        EncodedTensor.deserialize(with_metadata, format=format).codec_config is not None
    )
    restored = EncodedTensor.deserialize(without_metadata, format=format)
    assert restored.metadata == {}
    assert restored.codec_config is None
    assert len(without_metadata) < len(with_metadata)


def test_default_format_is_safetensors() -> None:
    data = CastCodec().encode(torch.randn(2, 3)).serialize()
    (header_size,) = struct.unpack_from("<Q", data, 0)
    assert data[8:9] == b"{"
    assert header_size < len(data)
    assert EncodedTensor.deserialize(data).tensors["values"].shape == (2, 3)


def test_unsupported_format() -> None:
    encoded = CastCodec().encode(torch.randn(2, 3))
    with pytest.raises(ValueError):
        encoded.serialize(format="npz")
    with pytest.raises(ValueError):
        EncodedTensor.deserialize(b"", format="npz")


def test_encode_attaches_codec_config() -> None:
    codec = LloydBlockRangeQuantCodec(value_bits=3, cache_table=True)
    encoded = codec.encode(torch.randn(1, 2, 8, 8))
    assert encoded.codec_config == codec.get_config()


def test_encoded_tensor_helpers(device: torch.device) -> None:
    encoded = BlockRangeQuantCodec().encode(torch.randn(2, 4, 8, 8, device=device))
    assert encoded.nbytes > 0
    moved = encoded.to("cpu")
    assert all(tensor.device.type == "cpu" for tensor in moved.tensors.values())
    assert moved.metadata == encoded.metadata


def test_serialize_from_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 8, 8, device=device)
    codec = BlockRangeQuantCodec(value_bits=6)
    restored = EncodedTensor.deserialize(codec.encode(latents).serialize())
    expected = codec.decode(codec.encode(latents)).cpu()
    assert torch.allclose(codec.decode(restored), expected, atol=1e-5)


def test_serialization_is_compact() -> None:
    latents = torch.randn(2, 32, 64, 64)
    raw_bytes = latents.numel() * 4
    sizes = {
        name: len(vaezip.save(latents, codec=name))
        for name in ("cast_fp32", "cast_fp16", "rq_8bit", "brq_4bit", "lbrq_2bit")
    }
    assert sizes["cast_fp32"] == pytest.approx(raw_bytes, rel=0.01)
    assert sizes["cast_fp16"] == pytest.approx(raw_bytes / 2, rel=0.01)
    assert sizes["rq_8bit"] == pytest.approx(raw_bytes / 4, rel=0.01)
    assert sizes["brq_4bit"] == pytest.approx(raw_bytes / 8 * 1.03, rel=0.02)
    assert sizes["lbrq_2bit"] == pytest.approx(raw_bytes / 16 * 1.06, rel=0.03)
