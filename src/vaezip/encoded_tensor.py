"""Container for compressed latents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vaezip.codecs.base import Codec

CODEC_CONFIG_KEY = "codec_config"


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

    def decode(
        self,
        codec: Codec | str | os.PathLike[str] | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        args:
            codec: codec object, shortcut name or config path, by default the codec
            from the metadata. If both are available they must have the same config.
            dtype: floating point dtype of the result
            device: device to decode on, by default the device of the stored tensors
        returns:
            tensor: [B, ...], original shape, dtype: dtype
        """
        from vaezip.codecs.auto import AutoCodec
        from vaezip.codecs.base import Codec
        from vaezip.codecs.config import without_version

        stored_config = self.codec_config
        if codec is None:
            if stored_config is None:
                raise ValueError(
                    "The encoded tensor has no codec config in its metadata, "
                    "pass the codec explicitly"
                )
            resolved_codec = AutoCodec.from_config(stored_config)
        else:
            resolved_codec = (
                codec if isinstance(codec, Codec) else AutoCodec.from_config(codec)
            )
            if stored_config is not None and without_version(
                stored_config
            ) != without_version(resolved_codec.get_config()):
                raise ValueError(
                    f"The codec {resolved_codec!r} does not match the config stored "
                    f"in the encoded tensor: {stored_config}"
                )
        encoded = self if device is None else self.to(device)
        return resolved_codec.decode(encoded, dtype=dtype)
