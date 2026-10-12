import pytest
import torch

from vaezip.codecs import BlockRangeQuantCodec, LloydBlockRangeQuantCodec

CODEC_CLASSES = [BlockRangeQuantCodec, LloydBlockRangeQuantCodec]


def error_bound(latents: torch.Tensor, value_bits: int) -> float:
    global_range = float(latents.max() - latents.min())
    return global_range / (2 * (2**value_bits - 1)) * 1.001 + 1e-6


@pytest.mark.parametrize(
    ("input_shape", "channel_dim", "block_size"),
    [
        ((2, 3, 16, 16), 1, (4, 4)),
        ((2, 3, 17, 13), 1, (4, 4)),
        ((2, 3, 5, 7), 1, (8, 8)),
        ((2, 3, 9, 6), 1, (1, 4)),
        ((2, 16, 3, 3), -3, (2, 2)),
        ((2, 4, 8, 16, 16), 1, (4, 4, 4)),
        ((2, 5, 8, 15, 17), 1, (1, 4, 4)),
        ((2, 8, 3, 16, 16), 2, (4, 4, 4)),
        ((2, 8, 16, 16, 3), -1, (4, 4, 2)),
        ((2, 3, 10), 1, (4,)),
        ((2, 10, 3), 2, (4,)),
        ((1, 2, 5, 6, 7, 3), 1, (2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_layouts_and_padding_keep_shape(
    codec_class: type[BlockRangeQuantCodec],
    input_shape: tuple[int, ...],
    channel_dim: int,
    block_size: tuple[int, ...],
) -> None:
    latents = torch.randn(*input_shape)
    codec = codec_class(
        value_bits=6, range_bits=6, channel_dim=channel_dim, block_size=block_size
    )
    decoded = codec.decode(codec.encode(latents))
    assert decoded.shape == latents.shape
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    assert (decoded - latents).pow(2).mean() <= error_bound(latents, 6) ** 2
    if codec_class is BlockRangeQuantCodec:
        assert (decoded - latents).abs().max() <= error_bound(latents, 6)


@pytest.mark.parametrize("value_bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_error_within_half_step(value_bits: int, device: torch.device) -> None:
    latents = torch.randn(2, 4, 24, 24, device=device)
    codec = BlockRangeQuantCodec(value_bits=value_bits, range_bits=8)
    decoded = codec.decode(codec.encode(latents))
    assert decoded.device.type == device.type
    assert (decoded - latents).abs().max() <= error_bound(latents, value_bits)


@pytest.mark.parametrize("range_bits", [1, 2, 3, 4, 5, 6, 7, 8])
def test_range_bits_do_not_break_roundtrip(range_bits: int) -> None:
    latents = torch.randn(2, 3, 16, 16)
    codec = BlockRangeQuantCodec(value_bits=8, range_bits=range_bits)
    decoded = codec.decode(codec.encode(latents))
    assert (decoded - latents).abs().max() <= error_bound(latents, 8)


def test_block_ranges_improve_over_global_range() -> None:
    latents = torch.randn(2, 4, 32, 32) * torch.linspace(0.01, 3.0, 32).reshape(
        1, 1, 32, 1
    )
    blockwise = BlockRangeQuantCodec(value_bits=4, range_bits=8, block_size=(4, 4))
    one_block = BlockRangeQuantCodec(value_bits=4, range_bits=8, block_size=(32, 32))
    blockwise_error = (
        (blockwise.decode(blockwise.encode(latents)) - latents).pow(2).mean()
    )
    one_block_error = (
        (one_block.decode(one_block.encode(latents)) - latents).pow(2).mean()
    )
    assert blockwise_error < one_block_error


@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_wrong_number_of_dimensions_raises(
    codec_class: type[BlockRangeQuantCodec],
) -> None:
    codec = codec_class(block_size=(4, 4))
    for shape in [(2, 3, 8), (2, 3, 8, 8, 8), (4,)]:
        with pytest.raises(ValueError):
            codec.encode(torch.randn(*shape))


@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_invalid_arguments(codec_class: type[BlockRangeQuantCodec]) -> None:
    with pytest.raises(ValueError):
        codec_class(channel_dim=0)
    with pytest.raises(ValueError):
        codec_class(value_bits=9)
    with pytest.raises(ValueError):
        codec_class(range_bits=0)
    with pytest.raises(ValueError):
        codec_class(block_size=())
    with pytest.raises(ValueError):
        codec_class(block_size=(4, 0))
    with pytest.raises(TypeError):
        codec_class(block_size=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        codec_class(channel_dim=-4, block_size=(4, 4)).encode(torch.randn(2, 3, 8, 8))


@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_channel_layouts_are_equivalent(
    codec_class: type[BlockRangeQuantCodec],
) -> None:
    latents = torch.randn(2, 4, 3, 10, 10)
    canonical = codec_class(channel_dim=1, block_size=(2, 4, 4))
    channel_third = codec_class(channel_dim=2, block_size=(2, 4, 4))
    expected = canonical.decode(canonical.encode(latents))
    actual = channel_third.decode(channel_third.encode(latents.transpose(1, 2)))
    assert torch.equal(actual, expected.transpose(1, 2))


@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_samples_are_independent(codec_class: type[BlockRangeQuantCodec]) -> None:
    scales = torch.tensor([1.0, 30.0, 0.05]).reshape(3, 1, 1, 1)
    latents = torch.randn(3, 4, 12, 12) * scales
    codec = codec_class(value_bits=4, block_size=(4, 4))
    batched = codec.decode(codec.encode(latents))
    for index in range(3):
        single = codec.decode(codec.encode(latents[index : index + 1]))
        assert torch.equal(single, batched[index : index + 1])


@pytest.mark.parametrize("value_bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("range_bits", [3, 4, 6])
def test_packed_sizes(value_bits: int, range_bits: int) -> None:
    latents = torch.randn(2, 4, 16, 16)
    encoded = BlockRangeQuantCodec(
        value_bits=value_bits, range_bits=range_bits, block_size=(4, 4)
    ).encode(latents)
    values_per_sample = 4 * 16 * 16
    blocks_per_sample = 4 * 4 * 4
    assert encoded.tensors["codes"].shape == (
        2,
        -(-values_per_sample * value_bits // 8),
    )
    assert encoded.tensors["range_lower"].shape == (
        2,
        -(-blocks_per_sample * range_bits // 8),
    )
    assert encoded.tensors["range_upper"].shape == (
        2,
        -(-blocks_per_sample * range_bits // 8),
    )
    assert encoded.tensors["channel_min"].shape == (2, 4)


def test_constant_input_and_constant_blocks() -> None:
    latents = torch.zeros(2, 3, 8, 8)
    latents[:, :, :4, :4] = 2.0
    for codec_class in CODEC_CLASSES:
        codec = codec_class(value_bits=4, range_bits=4, block_size=(4, 4))
        decoded = codec.decode(codec.encode(latents))
        assert torch.isfinite(decoded).all()
        assert torch.allclose(decoded, latents, atol=1e-5)
        constant = torch.full((2, 3, 8, 8), -3.0)
        assert torch.allclose(codec.decode(codec.encode(constant)), constant, atol=1e-5)


@pytest.mark.parametrize("codec_class", CODEC_CLASSES)
def test_decode_dtype(codec_class: type[BlockRangeQuantCodec]) -> None:
    latents = torch.randn(2, 4, 8, 8)
    codec = codec_class(value_bits=8)
    encoded = codec.encode(latents)
    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        decoded = codec.decode(encoded, dtype=dtype)
        assert decoded.dtype == dtype
        assert torch.allclose(decoded.float(), latents, atol=0.1)


def test_decoding_with_incompatible_codec_raises() -> None:
    encoded = BlockRangeQuantCodec(block_size=(4, 4, 4)).encode(
        torch.randn(2, 3, 8, 8, 8)
    )
    with pytest.raises(ValueError):
        BlockRangeQuantCodec(block_size=(4, 4)).decode(encoded)


def test_large_outlier_does_not_leak_into_other_channels() -> None:
    latents = torch.randn(1, 3, 16, 16)
    latents[0, 0, 0, 0] = 1000.0
    codec = BlockRangeQuantCodec(value_bits=8, range_bits=8, block_size=(4, 4))
    decoded = codec.decode(codec.encode(latents))
    assert (decoded[:, 1:] - latents[:, 1:]).abs().max() < 0.05


def test_dtype_inputs() -> None:
    latents = torch.randn(2, 3, 8, 8).to(torch.float16)
    codec = BlockRangeQuantCodec(value_bits=8)
    decoded = codec.decode(codec.encode(latents))
    assert torch.allclose(decoded, latents.float(), atol=0.05)
