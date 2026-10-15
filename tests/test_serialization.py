import io
import json
import struct
from typing import Any

import pytest
import torch
from safetensors.torch import save as save_safetensors

import vaezip
from vaezip import EncodedTensor
from vaezip.serialization import (
    FORMAT_VERSION,
    FORMAT_VERSION_KEY,
    PACKAGE_VERSION_KEY,
    deserialize,
    flatten,
    serialize,
    unflatten,
)


def make_encoded() -> EncodedTensor:
    return EncodedTensor(
        tensors={
            "a": torch.randn(2, 3),
            "b": torch.randint(0, 255, (4,), dtype=torch.uint8),
            "c": torch.randn(3).to(torch.float8_e4m3fn),
            "d": torch.randn(2).to(torch.bfloat16),
        },
        metadata={"key": "value"},
    )


def read_safetensors_header(data: bytes) -> dict[str, Any]:
    (header_size,) = struct.unpack_from("<Q", data, 0)
    header: dict[str, Any] = json.loads(data[8 : 8 + header_size])
    return header


@pytest.mark.parametrize("format", ["safetensors", "pth"])
def test_roundtrip_preserves_tensors_and_metadata(format: str) -> None:
    encoded_tensors = {"first": make_encoded(), "second": make_encoded()}
    restored = deserialize(serialize(encoded_tensors, format=format), format=format)
    assert list(restored) == ["first", "second"]
    for name, encoded in encoded_tensors.items():
        assert restored[name].metadata == {"key": "value"}
        assert restored[name].tensors.keys() == encoded.tensors.keys()
        for inner_name, tensor in encoded.tensors.items():
            restored_tensor = restored[name].tensors[inner_name]
            assert restored_tensor.dtype == tensor.dtype
            assert torch.equal(restored_tensor.float(), tensor.float())


def test_entry_without_metadata_roundtrips() -> None:
    encoded = EncodedTensor(tensors={"a": torch.randn(2)})
    restored = deserialize(serialize({"x": encoded}))
    assert restored["x"].metadata == {}


def test_safetensors_layout() -> None:
    latents = torch.randn(2, 8, 16, 16)
    encoded = vaezip.encode(latents, codec="lbrq_3bit")
    header = read_safetensors_header(vaezip.save({"model.latents": encoded}))
    metadata = header.pop("__metadata__")
    assert set(header) == {f"model.latents.{name}" for name in encoded.tensors}
    assert metadata[FORMAT_VERSION_KEY] == FORMAT_VERSION
    assert metadata[PACKAGE_VERSION_KEY] == vaezip.__version__
    assert metadata["model.latents.codec_config"] == encoded.metadata["codec_config"]
    assert len(metadata) == 3


def test_default_format_is_safetensors() -> None:
    data = serialize({"x": make_encoded()})
    (header_size,) = struct.unpack_from("<Q", data, 0)
    assert data[8:9] == b"{"
    assert header_size < len(data)


def test_flatten_unflatten_inverse() -> None:
    encoded_tensors = {"b.c": make_encoded(), "a": make_encoded()}
    tensors, metadata = flatten(encoded_tensors)
    assert "b.c.a" in tensors
    assert metadata["b.c.key"] == "value"
    restored = unflatten(tensors, metadata)
    assert list(restored) == ["a", "b.c"]
    assert restored["b.c"].metadata == {"key": "value"}


def test_flatten_makes_cpu_contiguous_copies() -> None:
    source = torch.randn(4, 4).t()
    tensors, _ = flatten({"x": EncodedTensor(tensors={"a": source})})
    assert tensors["x.a"].is_contiguous()
    assert tensors["x.a"].data_ptr() != source.data_ptr()


@pytest.mark.parametrize(
    "encoded_tensors",
    [
        {"x": EncodedTensor(tensors={"a.b": torch.randn(2)})},
        {"x": EncodedTensor(tensors={"": torch.randn(2)})},
        {"x": EncodedTensor(tensors={"a": torch.randn(2)}, metadata={"k.v": "1"})},
        {"x": EncodedTensor(tensors={})},
    ],
)
def test_flatten_rejects_invalid_inner_names(
    encoded_tensors: dict[str, EncodedTensor],
) -> None:
    with pytest.raises(ValueError):
        flatten(encoded_tensors)


def test_flatten_rejects_non_string_metadata_value() -> None:
    encoded = EncodedTensor(tensors={"a": torch.randn(2)}, metadata={"k": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="string"):
        flatten({"x": encoded})


def test_flatten_rejects_non_dict_input() -> None:
    with pytest.raises(TypeError):
        flatten(make_encoded())  # type: ignore[arg-type]


def test_unsupported_format() -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        serialize({"x": make_encoded()}, format="npz")
    with pytest.raises(ValueError, match="Unsupported format"):
        deserialize(b"", format="npz")


def test_load_rejects_file_without_format_version() -> None:
    data = save_safetensors({"x.a": torch.randn(2)}, metadata={"x.key": "value"})
    with pytest.raises(ValueError, match="Not a valid VaeZip file"):
        deserialize(data)


def test_load_rejects_unknown_format_version() -> None:
    data = save_safetensors(
        {"x.a": torch.randn(2)}, metadata={FORMAT_VERSION_KEY: "999"}
    )
    with pytest.raises(ValueError, match="Not a valid VaeZip file"):
        deserialize(data)


def test_load_rejects_key_without_delimiter() -> None:
    data = save_safetensors(
        {"a": torch.randn(2)}, metadata={FORMAT_VERSION_KEY: FORMAT_VERSION}
    )
    with pytest.raises(ValueError, match="unexpected key"):
        deserialize(data)


def test_load_rejects_metadata_without_tensors() -> None:
    data = save_safetensors(
        {"x.a": torch.randn(2)},
        metadata={FORMAT_VERSION_KEY: FORMAT_VERSION, "y.codec_config": "{}"},
    )
    with pytest.raises(ValueError, match="no tensors"):
        deserialize(data)


def test_load_rejects_unexpected_metadata_key() -> None:
    data = save_safetensors(
        {"x.a": torch.randn(2)},
        metadata={FORMAT_VERSION_KEY: FORMAT_VERSION, "stray": "1"},
    )
    with pytest.raises(ValueError, match="unexpected key"):
        deserialize(data)


@pytest.mark.parametrize("format", ["safetensors", "pth"])
def test_load_rejects_garbage(format: str) -> None:
    with pytest.raises(ValueError, match="Not a valid VaeZip"):
        deserialize(b"definitely not a vaezip file", format=format)


def test_load_rejects_pth_with_wrong_structure() -> None:
    buffer = io.BytesIO()
    torch.save({"unexpected": 1}, buffer)
    with pytest.raises(ValueError, match="Not a valid VaeZip"):
        deserialize(buffer.getvalue(), format="pth")
