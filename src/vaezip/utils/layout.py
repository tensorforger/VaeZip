"""Tensor layout handling: batch, channel and spatial dimensions."""

import torch

BATCH_DIM = 0


def check_channel_dim(channel_dim: int) -> None:
    if isinstance(channel_dim, bool) or not isinstance(channel_dim, int):
        raise TypeError(f"channel_dim must be an int, got {type(channel_dim).__name__}")
    if channel_dim == BATCH_DIM:
        raise ValueError("channel_dim can not be 0, the first dimension is the batch")


def resolve_channel_dim(channel_dim: int, ndim: int) -> int:
    """Non-negative channel index, negative values count from the end."""
    if not -ndim < channel_dim < ndim or channel_dim % ndim == BATCH_DIM:
        raise ValueError(
            f"channel_dim={channel_dim} is invalid for a tensor with {ndim} dimensions"
        )
    return channel_dim % ndim


def channel_to_second(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """
    args:
        tensor: [B, ...] with the channel at `channel_dim`
    returns:
        tensor: [B, C, ...] view with spatial dimensions in the original order
    """
    return tensor.movedim(resolve_channel_dim(channel_dim, tensor.ndim), 1)


def channel_from_second(tensor: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """
    args:
        tensor: [B, C, ...]
    returns:
        tensor: [B, ...] view with the channel moved to `channel_dim`
    """
    return tensor.movedim(1, resolve_channel_dim(channel_dim, tensor.ndim))


def shape_with_channel_second(shape: list[int], channel_dim: int) -> list[int]:
    """Shape [B, C, ...] of a tensor with the given original shape."""
    channel_index = resolve_channel_dim(channel_dim, len(shape))
    spatial_sizes = [
        size for index, size in enumerate(shape) if index not in (0, channel_index)
    ]
    return [shape[0], shape[channel_index], *spatial_sizes]
