"""Per-channel range quantization to low-bit integers."""

import math
from typing import Any

import torch

from vaezip.codecs.base import Codec
from vaezip.utils.bit_packing import check_bits, max_code, pack_bits, unpack_bits
from vaezip.utils.layout import (
    channel_from_second,
    channel_to_second,
    check_channel_dim,
    shape_with_channel_second,
)

CODES_KEY = "codes"
CHANNEL_MIN_KEY = "channel_min"
CHANNEL_MAX_KEY = "channel_max"
SHAPE_KEY = "shape"
MIN_RANGE = torch.finfo(torch.float32).tiny


class RangeQuantCodec(Codec):
    """
    Normalizes each channel of each sample by its min and max, then quantizes to
    `bits`-bit integers packed into bytes. The first dimension is the batch, any number
    of spatial dimensions is supported.
    """

    def __init__(self, bits: int = 8, channel_dim: int = 1) -> None:
        super().__init__()
        check_bits(bits, "bits")
        check_channel_dim(channel_dim)
        self.bits = bits
        self.channel_dim = channel_dim

    def _encode_tensors(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        args:
            tensor: [B, ...], floating point, one channel dimension at channel_dim
        returns:
            tensors: codes: [B, ceil(N * bits / 8)], channel_min, channel_max: [B, C],
            shape: [ndim], dtype: torch.int64, N: number of values per sample
        """
        channel_first = channel_to_second(tensor.float(), self.channel_dim)
        batch_size, channels = channel_first.shape[:2]
        values = channel_first.reshape(batch_size, channels, -1)  # [B, C, S]
        channel_min = values.amin(dim=2, keepdim=True)
        channel_max = values.amax(dim=2, keepdim=True)
        step = (channel_max - channel_min).clamp(min=MIN_RANGE) / max_code(self.bits)
        codes = torch.round((values - channel_min) / step)
        codes = codes.clamp(0, max_code(self.bits)).to(torch.uint8)
        return {
            CODES_KEY: pack_bits(codes.reshape(batch_size, -1), self.bits),
            CHANNEL_MIN_KEY: channel_min.reshape(batch_size, channels),
            CHANNEL_MAX_KEY: channel_max.reshape(batch_size, channels),
            SHAPE_KEY: torch.tensor(
                tensor.shape, dtype=torch.int64, device=tensor.device
            ),
        }

    def _decode_tensors(
        self, tensors: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        args:
            tensors: output of `_encode_tensors()`
        returns:
            tensor: [B, ...], dtype: dtype
        """
        shape = [int(size) for size in tensors[SHAPE_KEY].tolist()]
        batch_size, channels, *spatial_shape = shape_with_channel_second(
            shape, self.channel_dim
        )
        count = channels * math.prod(spatial_shape)
        codes = unpack_bits(tensors[CODES_KEY], count, self.bits)
        codes = codes.reshape(batch_size, channels, -1).float()  # [B, C, S]
        channel_min = tensors[CHANNEL_MIN_KEY].reshape(batch_size, channels, 1)
        channel_max = tensors[CHANNEL_MAX_KEY].reshape(batch_size, channels, 1)
        step = (channel_max - channel_min).clamp(min=MIN_RANGE) / max_code(self.bits)
        values = (codes * step + channel_min).reshape(
            batch_size, channels, *spatial_shape
        )
        return channel_from_second(values, self.channel_dim).to(dtype).contiguous()

    def get_config_parameters(self) -> dict[str, Any]:
        return {"bits": self.bits, "channel_dim": self.channel_dim}
