"""Codec that casts values to a lower precision floating point dtype."""

from typing import Any

import torch

from vaezip.codecs.base import Codec

VALUES_KEY = "values"


def parse_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Accepts a torch dtype or its name such as "float16"."""
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown torch dtype {dtype!r}")
        return resolved
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(
        f"Expected a torch.dtype or a dtype name, got {type(dtype).__name__}"
    )


class CastCodec(Codec):
    """Stores the tensor in a floating point dtype, out-of-range values are clamped."""

    def __init__(self, cast_dtype: torch.dtype | str = torch.float16) -> None:
        super().__init__()
        self.cast_dtype = parse_dtype(cast_dtype)
        if not self.cast_dtype.is_floating_point:
            raise ValueError(
                f"cast_dtype must be floating point, got {self.cast_dtype}"
            )

    def _encode_tensors(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        args:
            tensor: [B, ...], floating point
        returns:
            tensors: values: [B, ...], dtype: cast_dtype
        """
        cast_max = torch.finfo(self.cast_dtype).max
        if cast_max < torch.finfo(tensor.dtype).max:
            tensor = tensor.clamp(-cast_max, cast_max)
        return {VALUES_KEY: tensor.to(self.cast_dtype)}

    def _decode_tensors(
        self, tensors: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        args:
            tensors: values: [B, ...], dtype: cast_dtype
        returns:
            tensor: [B, ...], dtype: dtype
        """
        return tensors[VALUES_KEY].to(dtype)

    def get_config_parameters(self) -> dict[str, Any]:
        return {"cast_dtype": str(self.cast_dtype).removeprefix("torch.")}
