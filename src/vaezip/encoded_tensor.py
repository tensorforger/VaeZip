"""Container for compressed latents and its (de)serialization."""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

CODEC_CONFIG_KEY = "codec_config"
SAFETENSORS_FORMAT = "safetensors"
PTH_FORMAT = "pth"
SUPPORTED_FORMATS = (SAFETENSORS_FORMAT, PTH_FORMAT)

_PTH_TENSORS_KEY = "tensors"
_PTH_METADATA_KEY = "metadata"
_SAFETENSORS_HEADER_SIZE_BYTES = 8
_SAFETENSORS_METADATA_KEY = "__metadata__"


def check_format(format: str) -> None:
    if format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {format!r}, expected one of {SUPPORTED_FORMATS}"
        )


@dataclass
class EncodedTensor:
    """
    Output of `Codec.encode()` and input of `Codec.decode()`.

    tensors: named tensors with the compressed representation.
    metadata: optional string metadata, normally holds the JSON config of the codec.
    """

    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def codec_config(self) -> dict[str, Any] | None:
        """Parsed codec config from the metadata, `None` if there is none."""
        raw_config = self.metadata.get(CODEC_CONFIG_KEY)
        if raw_config is None:
            return None
        config = json.loads(raw_config)
        if not isinstance(config, dict):
            raise ValueError("Codec config in the metadata must be a JSON object")
        return config

    @property
    def nbytes(self) -> int:
        """Total size of the stored tensors in bytes."""
        return sum(
            tensor.numel() * tensor.element_size() for tensor in self.tensors.values()
        )

    def to(self, device: torch.device | str) -> EncodedTensor:
        return EncodedTensor(
            tensors={name: tensor.to(device) for name, tensor in self.tensors.items()},
            metadata=dict(self.metadata),
        )

    def serialize(
        self, save_metadata: bool = True, format: str = SAFETENSORS_FORMAT
    ) -> bytes:
        """
        args:
            save_metadata: store the metadata (including the codec config)
            format: "safetensors" or "pth"
        returns:
            serialized bytes
        """
        check_format(format)
        metadata = dict(self.metadata) if save_metadata else {}
        cpu_tensors = {
            name: tensor.detach().to("cpu", copy=True).contiguous()
            for name, tensor in self.tensors.items()
        }
        if format == SAFETENSORS_FORMAT:
            return save_safetensors(cpu_tensors, metadata=metadata or None)
        buffer = io.BytesIO()
        torch.save({_PTH_TENSORS_KEY: cpu_tensors, _PTH_METADATA_KEY: metadata}, buffer)
        return buffer.getvalue()

    @classmethod
    def deserialize(
        cls, data: bytes, format: str = SAFETENSORS_FORMAT
    ) -> EncodedTensor:
        """
        args:
            data: bytes produced by `serialize()`
            format: "safetensors" or "pth"
        returns:
            encoded tensor with tensors on CPU
        """
        check_format(format)
        if format == SAFETENSORS_FORMAT:
            return cls(
                tensors=load_safetensors(data),
                metadata=_read_safetensors_metadata(data),
            )
        content = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        return cls(
            tensors=dict(content[_PTH_TENSORS_KEY]),
            metadata=dict(content[_PTH_METADATA_KEY]),
        )


def serialize(
    encoded: EncodedTensor,
    save_metadata: bool = True,
    format: str = SAFETENSORS_FORMAT,
) -> bytes:
    return encoded.serialize(save_metadata=save_metadata, format=format)


def deserialize(data: bytes, format: str = SAFETENSORS_FORMAT) -> EncodedTensor:
    return EncodedTensor.deserialize(data, format=format)


def _read_safetensors_metadata(data: bytes) -> dict[str, str]:
    (header_size,) = struct.unpack_from("<Q", data, 0)
    header_end = _SAFETENSORS_HEADER_SIZE_BYTES + header_size
    header = json.loads(bytes(data[_SAFETENSORS_HEADER_SIZE_BYTES:header_end]))
    metadata: dict[str, str] = header.get(_SAFETENSORS_METADATA_KEY, {})
    return metadata
