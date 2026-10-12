"""Blockwise range quantization to low-bit integers."""

import math
from collections.abc import Sequence
from typing import Any

import torch

from vaezip.codecs.base import Codec
from vaezip.codecs.range_quant import (
    CHANNEL_MAX_KEY,
    CHANNEL_MIN_KEY,
    CODES_KEY,
    MIN_RANGE,
    SHAPE_KEY,
)
from vaezip.utils.bit_packing import check_bits, max_code, pack_bits, unpack_bits
from vaezip.utils.blocks import merge_blocks, padded_size, split_into_blocks
from vaezip.utils.layout import (
    channel_from_second,
    channel_to_second,
    check_channel_dim,
    shape_with_channel_second,
)

RANGE_LOWER_KEY = "range_lower"
RANGE_UPPER_KEY = "range_upper"


def parse_block_size(block_size: Sequence[int]) -> tuple[int, ...]:
    if isinstance(block_size, (str, bytes)) or not isinstance(block_size, Sequence):
        raise TypeError("block_size must be a sequence of ints")
    parsed = tuple(block_size)
    if not parsed:
        raise ValueError("block_size must have at least one dimension")
    for extent in parsed:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
            raise ValueError(f"block_size must contain positive ints, got {block_size}")
    return parsed


class BlockRangeQuantCodec(Codec):
    """
    Quantizes each block of each channel between its own min and max.
    The block ranges are quantized to `range_bits` relative to the min and max of the
    whole channel, the values to `value_bits` inside the block range. Everything is
    packed into bytes.

    The first dimension is the batch, one other dimension is the channel, the remaining
    dimensions are spatial and `block_size` has one entry for each of them. Spatial
    sizes that are not divisible by the block size are padded internally, the decoded
    tensor has the original shape.
    """

    def __init__(
        self,
        value_bits: int = 4,
        range_bits: int = 4,
        channel_dim: int = 1,
        block_size: Sequence[int] = (4, 4),
    ) -> None:
        super().__init__()
        check_bits(value_bits, "value_bits")
        check_bits(range_bits, "range_bits")
        check_channel_dim(channel_dim)
        self.value_bits = value_bits
        self.range_bits = range_bits
        self.channel_dim = channel_dim
        self.block_size = parse_block_size(block_size)

    def _encode_tensors(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        args:
            tensor: [B, ...] with ndim == len(block_size) + 2, floating point
        returns:
            tensors: codes: [B, ceil(C * G * V * value_bits / 8)],
            range_lower, range_upper: [B, ceil(C * G * range_bits / 8)],
            channel_min, channel_max: [B, C], shape: [ndim], dtype: torch.int64,
            G: number of blocks per channel, V: number of values per block
        """
        if tensor.ndim != len(self.block_size) + 2:
            raise ValueError(
                f"Expected a tensor with {len(self.block_size) + 2} dimensions "
                f"(batch, channel and {len(self.block_size)} spatial), got shape "
                f"{tuple(tensor.shape)}"
            )
        channel_first = channel_to_second(tensor.float(), self.channel_dim)
        blocks = split_into_blocks(channel_first, self.block_size)  # [B, C, G, V]
        batch_size = blocks.shape[0]
        range_levels = max_code(self.range_bits)

        channel_min = blocks.amin(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        channel_max = blocks.amax(dim=(2, 3), keepdim=True)
        range_step = _range_step(channel_min, channel_max, range_levels)
        block_min = blocks.amin(dim=3, keepdim=True)  # [B, C, G, 1]
        block_max = blocks.amax(dim=3, keepdim=True)
        lower = torch.floor((block_min - channel_min) / range_step)
        lower = lower.clamp(0, range_levels - 1)
        upper = torch.ceil((block_max - channel_min) / range_step).clamp(
            max=range_levels
        )
        upper = torch.maximum(upper, lower + 1)
        start, span = _block_grid(lower, upper, channel_min, range_step)

        normalized = ((blocks - start) / span).clamp(0, 1)
        codes, quantizer_tensors = self._quantize_normalized(normalized, span)
        return {
            CODES_KEY: pack_bits(codes.reshape(batch_size, -1), self.value_bits),
            RANGE_LOWER_KEY: pack_bits(
                lower.reshape(batch_size, -1).to(torch.uint8), self.range_bits
            ),
            RANGE_UPPER_KEY: pack_bits(
                upper.reshape(batch_size, -1).to(torch.uint8), self.range_bits
            ),
            CHANNEL_MIN_KEY: channel_min.reshape(batch_size, -1),
            CHANNEL_MAX_KEY: channel_max.reshape(batch_size, -1),
            SHAPE_KEY: torch.tensor(
                tensor.shape, dtype=torch.int64, device=tensor.device
            ),
            **quantizer_tensors,
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
        if len(shape) != len(self.block_size) + 2:
            raise ValueError(
                f"Encoded tensor has {len(shape)} dimensions, "
                f"this codec expects {len(self.block_size) + 2}"
            )
        batch_size, channels, *spatial_shape = shape_with_channel_second(
            shape, self.channel_dim
        )
        blocks_per_channel = math.prod(
            padded_size(size, extent) // extent
            for size, extent in zip(spatial_shape, self.block_size, strict=True)
        )
        values_per_block = math.prod(self.block_size)
        range_levels = max_code(self.range_bits)

        ranges_count = channels * blocks_per_channel
        lower = unpack_bits(tensors[RANGE_LOWER_KEY], ranges_count, self.range_bits)
        upper = unpack_bits(tensors[RANGE_UPPER_KEY], ranges_count, self.range_bits)
        lower = lower.reshape(batch_size, channels, blocks_per_channel, 1).float()
        upper = upper.reshape(batch_size, channels, blocks_per_channel, 1).float()
        channel_min = tensors[CHANNEL_MIN_KEY].reshape(batch_size, channels, 1, 1)
        channel_max = tensors[CHANNEL_MAX_KEY].reshape(batch_size, channels, 1, 1)
        range_step = _range_step(channel_min, channel_max, range_levels)
        start, span = _block_grid(lower, upper, channel_min, range_step)

        codes = unpack_bits(
            tensors[CODES_KEY], ranges_count * values_per_block, self.value_bits
        )
        codes = codes.reshape(
            batch_size, channels, blocks_per_channel, values_per_block
        )
        normalized = self._dequantize_normalized(codes, tensors)
        blocks = normalized * span + start  # [B, C, G, V]
        channel_first = merge_blocks(blocks, spatial_shape, self.block_size)
        return (
            channel_from_second(channel_first, self.channel_dim).to(dtype).contiguous()
        )

    def _quantize_normalized(
        self, normalized: torch.Tensor, span: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        args:
            normalized: [B, C, G, V], values: [0...1], position inside the block range
            span: [B, C, G, 1], width of the block range
        returns:
            codes: [B, C, G, V], dtype: torch.uint8, values: [0...2^value_bits - 1]
            tensors: additional tensors needed for dequantization
        """
        codes = torch.round(normalized * max_code(self.value_bits))
        return codes.to(torch.uint8), {}

    def _dequantize_normalized(
        self, codes: torch.Tensor, tensors: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        args:
            codes: [B, C, G, V], dtype: torch.uint8
            tensors: output of `_encode_tensors()`
        returns:
            normalized: [B, C, G, V], dtype: torch.float32, values: [0...1]
        """
        return codes.float() / max_code(self.value_bits)

    def get_config_parameters(self) -> dict[str, Any]:
        return {
            "value_bits": self.value_bits,
            "range_bits": self.range_bits,
            "channel_dim": self.channel_dim,
            "block_size": list(self.block_size),
        }


def _range_step(
    channel_min: torch.Tensor, channel_max: torch.Tensor, range_levels: int
) -> torch.Tensor:
    """
    args:
        channel_min, channel_max: [B, C, 1, 1]
    returns:
        range_step: [B, C, 1, 1], distance between neighbouring block range positions
    """
    return ((channel_max - channel_min) / range_levels).clamp(min=MIN_RANGE)


def _block_grid(
    lower: torch.Tensor,
    upper: torch.Tensor,
    channel_min: torch.Tensor,
    range_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    args:
        lower, upper: [B, C, G, 1], dtype: torch.float32, block range positions
        channel_min, range_step: [B, C, 1, 1]
    returns:
        start: [B, C, G, 1], lower end of the block range
        span: [B, C, G, 1], width of the block range
    """
    return channel_min + lower * range_step, (upper - lower) * range_step
