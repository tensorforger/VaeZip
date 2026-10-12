"""High level save and load of tensors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import overload

import torch

from vaezip.codecs.auto import AutoCodec
from vaezip.codecs.base import Codec
from vaezip.codecs.config import without_version
from vaezip.codecs.shortcuts import DEFAULT_SHORTCUT
from vaezip.encoded_tensor import (
    PTH_FORMAT,
    SAFETENSORS_FORMAT,
    EncodedTensor,
)

CodecSpecification = Codec | str | os.PathLike[str]
PTH_SUFFIXES = (".pth", ".pt")


def _resolve_codec(codec: CodecSpecification | None) -> Codec:
    if codec is None:
        return AutoCodec.from_config(DEFAULT_SHORTCUT)
    if isinstance(codec, Codec):
        return codec
    return AutoCodec.from_config(codec)


def _infer_format(path: str | os.PathLike[str], format: str | None) -> str:
    if format is not None:
        return format
    return PTH_FORMAT if Path(path).suffix in PTH_SUFFIXES else SAFETENSORS_FORMAT


@overload
def save(
    tensor: torch.Tensor,
    path: None = None,
    codec: CodecSpecification | None = None,
    save_metadata: bool = True,
    format: str | None = None,
) -> bytes: ...


@overload
def save(
    tensor: torch.Tensor,
    path: str | os.PathLike[str],
    codec: CodecSpecification | None = None,
    save_metadata: bool = True,
    format: str | None = None,
) -> None: ...


def save(
    tensor: torch.Tensor,
    path: str | os.PathLike[str] | None = None,
    codec: CodecSpecification | None = None,
    save_metadata: bool = True,
    format: str | None = None,
) -> bytes | None:
    """
    Encodes the tensor and writes it to a file, or returns bytes if there is no path.

    args:
        tensor: [B, ...], floating point
        path: output file, `None` to return bytes
        codec: codec object, shortcut name or config path, default is "cast_fp32"
        save_metadata: store the codec config, required by `load()` without a codec
        format: "safetensors" or "pth", by default inferred from the path suffix
        (safetensors if unknown)
    """
    resolved_codec = _resolve_codec(codec)
    if path is None:
        serialization_format = format or SAFETENSORS_FORMAT
    else:
        serialization_format = _infer_format(path, format)
    data = resolved_codec.encode(tensor).serialize(
        save_metadata=save_metadata, format=serialization_format
    )
    if path is None:
        return data
    Path(path).write_bytes(data)
    return None


def load(
    source: str | os.PathLike[str] | bytes | bytearray,
    codec: CodecSpecification | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    format: str | None = None,
) -> torch.Tensor:
    """
    Reads a file or bytes written by `save()` and decodes the tensor.

    args:
        source: file path or bytes
        codec: codec object, shortcut name or config path, by default the codec
        from the metadata. If both are available they must have the same config.
        dtype: floating point dtype of the result
        device: device to decode on, default is CPU
        format: "safetensors" or "pth", by default inferred from the path suffix
        (safetensors for bytes)
    returns:
        tensor: [B, ...], original shape, dtype: dtype
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        serialization_format = format or SAFETENSORS_FORMAT
    else:
        data = Path(source).read_bytes()
        serialization_format = _infer_format(source, format)
    encoded = EncodedTensor.deserialize(data, format=serialization_format)
    stored_config = encoded.codec_config

    if codec is None:
        if stored_config is None:
            raise ValueError(
                "The data has no codec config in its metadata, "
                "pass the codec explicitly"
            )
        resolved_codec = AutoCodec.from_config(stored_config)
    else:
        resolved_codec = _resolve_codec(codec)
        if stored_config is not None and without_version(
            stored_config
        ) != without_version(resolved_codec.get_config()):
            raise ValueError(
                f"The codec {resolved_codec!r} does not match the config stored in the "
                f"data: {stored_config}"
            )
    if device is not None:
        encoded = encoded.to(device)
    return resolved_codec.decode(encoded, dtype=dtype)
