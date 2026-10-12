"""Abstract codec API."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch

from vaezip._version import __version__
from vaezip.codecs.config import ConfigSource, split_config
from vaezip.codecs.shortcuts import CODEC_CLASS_KEY, VERSION_KEY
from vaezip.encoded_tensor import CODEC_CONFIG_KEY, EncodedTensor

CodecType = TypeVar("CodecType", bound="Codec")


class Codec(ABC):
    """
    Compression algorithm for tensors.

    The configuration is fixed at construction. Encoding takes a tensor and returns an
    `EncodedTensor` on the same device, decoding restores a float tensor of the
    original shape.
    """

    def encode(self, tensor: torch.Tensor) -> EncodedTensor:
        """
        args:
            tensor: [B, ...], floating point, finite values
        returns:
            encoded: tensors on the device of the input, metadata with the codec config
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise TypeError(f"Expected a floating point tensor, got {tensor.dtype}")
        if tensor.numel() == 0:
            raise ValueError("Can not encode an empty tensor")
        tensors = self._encode_tensors(tensor.detach())
        metadata = {CODEC_CONFIG_KEY: json.dumps(self.get_config())}
        return EncodedTensor(tensors=tensors, metadata=metadata)

    def decode(
        self, encoded: EncodedTensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        args:
            encoded: output of `encode()` of the same codec configuration
            dtype: floating point dtype of the result
        returns:
            tensor: [B, ...], original shape, dtype: dtype, device of encoded tensors
        """
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating point dtype, got {dtype}")
        return self._decode_tensors(encoded.tensors, dtype)

    @abstractmethod
    def _encode_tensors(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        args:
            tensor: [B, ...], floating point
        returns:
            tensors: compressed representation, per-sample tensors have the batch first
        """

    @abstractmethod
    def _decode_tensors(
        self, tensors: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        args:
            tensors: output of `_encode_tensors()`
            dtype: floating point dtype of the result
        returns:
            tensor: [B, ...], original shape, dtype: dtype
        """

    @abstractmethod
    def get_config_parameters(self) -> dict[str, Any]:
        """JSON-serializable constructor parameters, `from_config` accepts them back."""

    def get_config(self) -> dict[str, Any]:
        """Diffusers-style config: class name, package version and the parameters."""
        return {
            CODEC_CLASS_KEY: type(self).__name__,
            VERSION_KEY: __version__,
            **self.get_config_parameters(),
        }

    def save_config(self, path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as config_file:
            json.dump(self.get_config(), config_file, indent=2)
            config_file.write("\n")

    @classmethod
    def from_config(
        cls: type[CodecType], config: ConfigSource, **overrides: Any
    ) -> CodecType:
        """
        args:
            config: path to a JSON config, a shortcut name or a config dict
            overrides: constructor parameters that replace the ones from the config
        """
        class_name, parameters = split_config(config, overrides)
        if class_name is not None and class_name != cls.__name__:
            raise ValueError(
                f"Config describes {class_name}, "
                f"can not construct {cls.__name__} from it"
            )
        return cls(**parameters)

    def __repr__(self) -> str:
        parameters = ", ".join(
            f"{name}={value!r}" for name, value in self.get_config_parameters().items()
        )
        return f"{type(self).__name__}({parameters})"
