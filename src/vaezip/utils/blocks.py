"""Splitting spatial dimensions into blocks and merging them back."""

import math

import torch


def padded_size(size: int, block_extent: int) -> int:
    return math.ceil(size / block_extent) * block_extent


def pad_to_block_multiple(
    tensor: torch.Tensor, block_size: tuple[int, ...]
) -> torch.Tensor:
    """
    args:
        tensor: [B, C, *spatial]
        block_size: block extent per spatial dimension
    returns:
        tensor: [B, C, *padded_spatial], spatial sizes divisible by the block size,
        padding repeats edge values so block ranges are not affected
    """
    for offset, block_extent in enumerate(block_size):
        dim = 2 + offset
        size = tensor.shape[dim]
        target_size = padded_size(size, block_extent)
        if target_size != size:
            source_index = torch.arange(target_size, device=tensor.device)
            tensor = tensor.index_select(dim, source_index.clamp(max=size - 1))
    return tensor


def split_into_blocks(
    tensor: torch.Tensor, block_size: tuple[int, ...]
) -> torch.Tensor:
    """
    args:
        tensor: [B, C, *spatial]
        block_size: block extent per spatial dimension
    returns:
        blocks: [B, C, G, V], G: number of blocks per channel, V: values per block
    """
    padded = pad_to_block_multiple(tensor, block_size)
    batch_size, channels = padded.shape[:2]
    block_counts = [
        size // extent
        for size, extent in zip(padded.shape[2:], block_size, strict=True)
    ]
    split_shape = [
        part
        for count, extent in zip(block_counts, block_size, strict=True)
        for part in (count, extent)
    ]
    ndim = len(block_size)
    count_dims = range(2, 2 + 2 * ndim, 2)
    extent_dims = range(3, 3 + 2 * ndim, 2)
    split = padded.reshape(batch_size, channels, *split_shape)  # [B, C, n1, b1, n2, b2]
    grouped = split.permute(0, 1, *count_dims, *extent_dims)  # [B, C, n1, n2, b1, b2]
    return grouped.reshape(
        batch_size, channels, math.prod(block_counts), math.prod(block_size)
    )


def merge_blocks(
    blocks: torch.Tensor, spatial_shape: list[int], block_size: tuple[int, ...]
) -> torch.Tensor:
    """
    args:
        blocks: [B, C, G, V]
        spatial_shape: original spatial sizes before padding
        block_size: block extent per spatial dimension
    returns:
        tensor: [B, C, *spatial_shape]
    """
    batch_size, channels = blocks.shape[:2]
    block_counts = [
        padded_size(size, extent) // extent
        for size, extent in zip(spatial_shape, block_size, strict=True)
    ]
    ndim = len(block_size)
    grouped = blocks.reshape(batch_size, channels, *block_counts, *block_size)
    split_order = [
        dim for index in range(ndim) for dim in (2 + index, 2 + ndim + index)
    ]
    split = grouped.permute(0, 1, *split_order)  # [B, C, n1, b1, n2, b2]
    padded_shape = [
        count * extent for count, extent in zip(block_counts, block_size, strict=True)
    ]
    padded = split.reshape(batch_size, channels, *padded_shape)
    crop = tuple(slice(0, size) for size in spatial_shape)
    return padded[(slice(None), slice(None), *crop)]
