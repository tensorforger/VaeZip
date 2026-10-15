"""High level encoding, decoding, saving and loading of tensors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import overload

import torch

from vaezip.codecs.auto import AutoCodec
from vaezip.codecs.base import Codec
from vaezip.codecs.shortcuts import DEFAULT_SHORTCUT
from vaezip.encoded_tensor import EncodedTensor
from vaezip.serialization import (
    SAFETENSORS_FORMAT,
    SerializationFormat,
    deserialize,
    serialize,
)

CodecSpecification = Codec | str | os.PathLike[str]
SINGLE_TENSOR_NAME = "tensor"


def encode(
    tensor: torch.Tensor, codec: CodecSpecification | None = None
) -> EncodedTensor:
    """
    args:
        tensor: [B, ...], floating point
        codec: codec object, shortcut name or config path, default is "cast_fp32"
    returns:
        encoded: tensors on the device of the input, metadata with the codec config
    """
    if codec is None:
        codec = DEFAULT_SHORTCUT
    resolved_codec = codec if isinstance(codec, Codec) else AutoCodec.from_config(codec)
    return resolved_codec.encode(tensor)


def decode(
    encoded: EncodedTensor,
    codec: CodecSpecification | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Same as `EncodedTensor.decode()`.

    returns:
        tensor: [B, ...], original shape, dtype: dtype
    """
    return encoded.decode(codec=codec, dtype=dtype, device=device)


@overload
def save(
    encoded_tensors: dict[str, EncodedTensor],
    path: None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> bytes: ...


@overload
def save(
    encoded_tensors: dict[str, EncodedTensor],
    path: str | os.PathLike[str],
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> None: ...


def save(
    encoded_tensors: dict[str, EncodedTensor],
    path: str | os.PathLike[str] | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> bytes | None:
    """
    Writes named encoded tensors to a file, or returns bytes if there is no path.

    args:
        encoded_tensors: name -> encoded tensor, tensors may use different codecs
        path: output file, `None` to return bytes
        format: "safetensors" or "pth", never inferred from the path
    """
    return _write_or_return(serialize(encoded_tensors, format=format), path)


def load(
    source: str | os.PathLike[str] | bytes | bytearray,
    device: torch.device | str | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> dict[str, EncodedTensor]:
    """
    Reads a file or bytes written by `save()`.

    args:
        source: file path or bytes
        device: device of the loaded tensors, default is CPU
        format: "safetensors" or "pth", never inferred from the path
    returns:
        encoded_tensors: name -> encoded tensor, sorted by name
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = Path(source).read_bytes()
    encoded_tensors = deserialize(data, format=format)
    if device is not None:
        encoded_tensors = {
            name: encoded.to(device) for name, encoded in encoded_tensors.items()
        }
    return encoded_tensors


@overload
def save_tensor(
    tensor: torch.Tensor,
    path: None = None,
    codec: CodecSpecification | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> bytes: ...


@overload
def save_tensor(
    tensor: torch.Tensor,
    path: str | os.PathLike[str],
    codec: CodecSpecification | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> None: ...


def save_tensor(
    tensor: torch.Tensor,
    path: str | os.PathLike[str] | None = None,
    codec: CodecSpecification | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> bytes | None:
    """
    Encodes a single tensor and saves it as `{"tensor": encoded}`.

    args:
        tensor: [B, ...], floating point
        path: output file, `None` to return bytes
        codec: codec object, shortcut name or config path, default is "cast_fp32"
        format: "safetensors" or "pth", never inferred from the path
    """
    encoded_tensors = {SINGLE_TENSOR_NAME: encode(tensor, codec=codec)}
    return _write_or_return(serialize(encoded_tensors, format=format), path)


def load_tensor(
    source: str | os.PathLike[str] | bytes | bytearray,
    codec: CodecSpecification | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    format: SerializationFormat = SAFETENSORS_FORMAT,
) -> torch.Tensor:
    """
    Loads and decodes a file that holds exactly one entry named "tensor".

    args:
        source: file path or bytes
        codec: codec object, shortcut name or config path, by default the codec
        from the metadata
        dtype: floating point dtype of the result
        device: device to decode on, default is CPU
        format: "safetensors" or "pth", never inferred from the path
    returns:
        tensor: [B, ...], original shape, dtype: dtype
    """
    encoded_tensors = load(source, format=format)
    if list(encoded_tensors) != [SINGLE_TENSOR_NAME]:
        raise ValueError(
            f"Expected a single entry named {SINGLE_TENSOR_NAME!r}, "
            f"found: {list(encoded_tensors)}"
        )
    return encoded_tensors[SINGLE_TENSOR_NAME].decode(
        codec=codec, dtype=dtype, device=device
    )


def _write_or_return(data: bytes, path: str | os.PathLike[str] | None) -> bytes | None:
    if path is None:
        return data
    Path(path).write_bytes(data)
    return None
