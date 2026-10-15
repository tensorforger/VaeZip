"""Storage of named encoded tensors in safetensors and pth containers."""

from __future__ import annotations

import io
import json
import pickle
import struct
from typing import Any, Literal, get_args

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from vaezip._version import __version__
from vaezip.encoded_tensor import EncodedTensor

SerializationFormat = Literal["safetensors", "pth"]
SAFETENSORS_FORMAT: SerializationFormat = "safetensors"
PTH_FORMAT: SerializationFormat = "pth"
SUPPORTED_FORMATS: tuple[str, ...] = get_args(SerializationFormat)

FORMAT_VERSION = "1"
FORMAT_VERSION_KEY = "vaezip_format_version"
PACKAGE_VERSION_KEY = "vaezip_version"
KEY_DELIMITER = "."

_PTH_TENSORS_KEY = "tensors"
_PTH_METADATA_KEY = "metadata"
_SAFETENSORS_HEADER_SIZE_BYTES = 8
_SAFETENSORS_METADATA_KEY = "__metadata__"


def check_format(format: str) -> None:
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {format!r}, expected one of {SUPPORTED_FORMATS}"
        )


def flatten(
    encoded_tensors: dict[str, EncodedTensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Joins entry names with inner names: tensor "codes" of entry "latents" becomes
    "latents.codes", metadata is flattened the same way.

    args:
        encoded_tensors: entry name -> encoded tensor, names are non-empty
    returns:
        tensors: flat name -> tensor, on CPU, contiguous
        metadata: flat key -> string, includes the format and package versions
    """
    if not isinstance(encoded_tensors, dict):
        raise TypeError(
            f"Expected a dict of EncodedTensor, got {type(encoded_tensors).__name__}"
        )
    if not encoded_tensors:
        raise ValueError("Can not save an empty dict")
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {
        FORMAT_VERSION_KEY: FORMAT_VERSION,
        PACKAGE_VERSION_KEY: __version__,
    }
    for name, encoded in encoded_tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Names must be non-empty strings, got {name!r}")
        if not isinstance(encoded, EncodedTensor):
            raise TypeError(
                f"Expected an EncodedTensor for {name!r}, got {type(encoded).__name__}"
            )
        if not encoded.tensors:
            raise ValueError(f"EncodedTensor {name!r} has no tensors")
        for inner_name, tensor in encoded.tensors.items():
            _check_inner_key(name, inner_name)
            tensors[f"{name}{KEY_DELIMITER}{inner_name}"] = (
                tensor.detach().to("cpu", copy=True).contiguous()
            )
        for key, value in encoded.metadata.items():
            _check_inner_key(name, key)
            if not isinstance(value, str):
                raise ValueError(
                    f"Metadata {key!r} of {name!r} must be a string, "
                    f"got {type(value).__name__}"
                )
            metadata[f"{name}{KEY_DELIMITER}{key}"] = value
    return tensors, metadata


def unflatten(
    tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> dict[str, EncodedTensor]:
    """
    Inverse of `flatten()`, splits keys on the last delimiter.

    returns:
        entry name -> encoded tensor, sorted by name
    """
    format_version = metadata.get(FORMAT_VERSION_KEY)
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Not a valid VaeZip file, expected {FORMAT_VERSION_KEY}="
            f"{FORMAT_VERSION!r} in the metadata, got {format_version!r}"
        )
    grouped_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in tensors.items():
        name, inner_name = _split_key(flat_key)
        grouped_tensors.setdefault(name, {})[inner_name] = tensor
    grouped_metadata: dict[str, dict[str, str]] = {}
    for flat_key, value in metadata.items():
        if flat_key in (FORMAT_VERSION_KEY, PACKAGE_VERSION_KEY):
            continue
        name, key = _split_key(flat_key)
        if name not in grouped_tensors:
            raise ValueError(
                f"Not a valid VaeZip file, metadata {flat_key!r} has no tensors"
            )
        grouped_metadata.setdefault(name, {})[key] = value
    return {
        name: EncodedTensor(
            tensors=grouped_tensors[name], metadata=grouped_metadata.get(name, {})
        )
        for name in sorted(grouped_tensors)
    }


def serialize(
    encoded_tensors: dict[str, EncodedTensor], format: str = SAFETENSORS_FORMAT
) -> bytes:
    check_format(format)
    tensors, metadata = flatten(encoded_tensors)
    if format == SAFETENSORS_FORMAT:
        return save_safetensors(tensors, metadata=metadata)
    buffer = io.BytesIO()
    torch.save({_PTH_TENSORS_KEY: tensors, _PTH_METADATA_KEY: metadata}, buffer)
    return buffer.getvalue()


def deserialize(
    data: bytes, format: str = SAFETENSORS_FORMAT
) -> dict[str, EncodedTensor]:
    """returns: entry name -> encoded tensor with tensors on CPU, sorted by name"""
    check_format(format)
    try:
        if format == SAFETENSORS_FORMAT:
            tensors = load_safetensors(data)
            metadata = _read_safetensors_metadata(data)
        else:
            content = torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=True
            )
            tensors = dict(content[_PTH_TENSORS_KEY])
            metadata = dict(content[_PTH_METADATA_KEY])
    except (
        SafetensorError,
        struct.error,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        RuntimeError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(f"Not a valid VaeZip {format} file: {error}") from error
    return unflatten(tensors, metadata)


def _check_inner_key(name: str, key: Any) -> None:
    if not isinstance(key, str) or not key or KEY_DELIMITER in key:
        raise ValueError(
            f"Tensor and metadata names inside {name!r} must be non-empty strings "
            f"without {KEY_DELIMITER!r}, got {key!r}"
        )


def _split_key(flat_key: str) -> tuple[str, str]:
    name, delimiter, inner_name = flat_key.rpartition(KEY_DELIMITER)
    if not delimiter or not name or not inner_name:
        raise ValueError(f"Not a valid VaeZip file, unexpected key {flat_key!r}")
    return name, inner_name


def _read_safetensors_metadata(data: bytes) -> dict[str, str]:
    (header_size,) = struct.unpack_from("<Q", data, 0)
    header_end = _SAFETENSORS_HEADER_SIZE_BYTES + header_size
    header = json.loads(bytes(data[_SAFETENSORS_HEADER_SIZE_BYTES:header_end]))
    metadata: dict[str, str] = header.get(_SAFETENSORS_METADATA_KEY, {})
    return metadata
