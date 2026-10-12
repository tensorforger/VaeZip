import pytest
import torch

from vaezip.utils.bit_packing import pack_bits, unpack_bits
from vaezip.utils.blocks import merge_blocks, split_into_blocks
from vaezip.utils.layout import (
    channel_from_second,
    channel_to_second,
    check_channel_dim,
    resolve_channel_dim,
)


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("count", [1, 7, 8, 9, 100])
def test_pack_bits_roundtrip(bits: int, count: int) -> None:
    codes = torch.randint(0, 2**bits, (3, count), dtype=torch.uint8)
    packed = pack_bits(codes, bits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (3, -(-count * bits // 8))
    assert torch.equal(unpack_bits(packed, count, bits), codes)


@pytest.mark.parametrize(
    ("spatial_shape", "block_size"),
    [
        ((8, 8), (4, 4)),
        ((9, 5), (4, 2)),
        ((3,), (4,)),
        ((5, 6, 7), (2, 3, 4)),
        ((4, 4, 4, 4), (2, 2, 2, 2)),
        ((10, 10), (1, 10)),
    ],
)
def test_blocks_roundtrip(
    spatial_shape: tuple[int, ...], block_size: tuple[int, ...]
) -> None:
    tensor = torch.randn(2, 3, *spatial_shape)
    blocks = split_into_blocks(tensor, block_size)
    assert blocks.shape[:2] == (2, 3)
    assert blocks.shape[3] == torch.Size(block_size).numel()
    restored = merge_blocks(blocks, list(spatial_shape), block_size)
    assert torch.equal(restored, tensor)


def test_blocks_contain_neighbouring_values() -> None:
    tensor = torch.arange(16.0).reshape(1, 1, 4, 4)
    blocks = split_into_blocks(tensor, (2, 2))
    assert blocks[0, 0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]
    assert blocks[0, 0, 3].tolist() == [10.0, 11.0, 14.0, 15.0]


def test_padding_repeats_edge_values() -> None:
    tensor = torch.tensor([[[1.0, 2.0, 3.0]]])
    blocks = split_into_blocks(tensor, (4,))
    assert blocks[0, 0, 0].tolist() == [1.0, 2.0, 3.0, 3.0]


def test_channel_dim_validation() -> None:
    with pytest.raises(ValueError):
        check_channel_dim(0)
    with pytest.raises(TypeError):
        check_channel_dim(1.0)  # type: ignore[arg-type]
    check_channel_dim(-1)
    assert resolve_channel_dim(-1, 4) == 3
    assert resolve_channel_dim(2, 5) == 2
    for invalid in (-4, 4, 5):
        with pytest.raises(ValueError):
            resolve_channel_dim(invalid, 4)


@pytest.mark.parametrize("channel_dim", [1, 2, 3, -1, -3])
def test_channel_movement_roundtrip(channel_dim: int) -> None:
    tensor = torch.randn(2, 3, 4, 5)
    moved = channel_to_second(tensor, channel_dim)
    assert moved.shape[1] == tensor.shape[resolve_channel_dim(channel_dim, 4)]
    assert torch.equal(channel_from_second(moved, channel_dim), tensor)
