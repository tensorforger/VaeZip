import pytest
import torch

from vaezip.codecs import RangeQuantCodec


def max_range_error_bound(latents: torch.Tensor, bits: int) -> float:
    return float(latents.max() - latents.min()) / (2 * (2**bits - 1)) * 1.001 + 1e-6


@pytest.mark.parametrize("bits", range(1, 9))
def test_error_within_half_step(bits: int, device: torch.device) -> None:
    latents = torch.randn(3, 5, 9, 11, device=device)
    codec = RangeQuantCodec(bits=bits)
    decoded = codec.decode(codec.encode(latents))
    assert decoded.shape == latents.shape
    assert decoded.dtype == torch.float32
    assert decoded.device.type == device.type
    assert (decoded - latents).abs().max() <= max_range_error_bound(latents, bits)


@pytest.mark.parametrize(
    ("input_shape", "channel_dim"),
    [
        ((2, 6), 1),
        ((2, 6, 7), 1),
        ((2, 7, 6), -1),
        ((2, 3, 6, 7), 1),
        ((2, 3, 6, 7), 2),
        ((2, 3, 6, 7), 3),
        ((2, 3, 6, 7), -2),
        ((2, 3, 4, 5, 6), 1),
        ((2, 3, 4, 5, 6), 2),
        ((2, 3, 4, 5, 6), -1),
        ((1, 3, 4, 5, 6, 7), 4),
    ],
)
def test_layouts(input_shape: tuple[int, ...], channel_dim: int) -> None:
    latents = torch.randn(*input_shape)
    codec = RangeQuantCodec(bits=6, channel_dim=channel_dim)
    decoded = codec.decode(codec.encode(latents))
    assert decoded.shape == latents.shape
    assert (decoded - latents).abs().max() <= max_range_error_bound(latents, 6)


def test_channels_are_normalized_independently() -> None:
    latents = torch.randn(2, 3, 16, 16)
    latents[:, 1] *= 1000.0
    codec = RangeQuantCodec(bits=8)
    decoded = codec.decode(codec.encode(latents))
    for channel in range(3):
        channel_range = latents[:, channel].max() - latents[:, channel].min()
        error = (decoded[:, channel] - latents[:, channel]).abs().max()
        assert error <= channel_range / 255 / 2 * 1.001 + 1e-6


def test_samples_are_independent() -> None:
    latents = torch.randn(3, 4, 8, 8) * torch.tensor([1.0, 50.0, 0.01]).reshape(
        3, 1, 1, 1
    )
    codec = RangeQuantCodec(bits=5)
    batched = codec.decode(codec.encode(latents))
    for index in range(3):
        single = codec.decode(codec.encode(latents[index : index + 1]))
        assert torch.equal(single, batched[index : index + 1])


def test_channel_dim_layouts_are_equivalent() -> None:
    latents = torch.randn(2, 4, 5, 6)
    canonical = RangeQuantCodec(bits=4, channel_dim=1)
    channel_last = RangeQuantCodec(bits=4, channel_dim=-1)
    expected = canonical.decode(canonical.encode(latents))
    moved = latents.permute(0, 2, 3, 1)
    actual = channel_last.decode(channel_last.encode(moved))
    assert torch.equal(actual, expected.permute(0, 2, 3, 1))


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
def test_size_matches_bits(bits: int) -> None:
    latents = torch.randn(2, 8, 16, 16)
    encoded = RangeQuantCodec(bits=bits).encode(latents)
    assert encoded.tensors["codes"].shape == (2, 8 * 16 * 16 * bits // 8)
    assert encoded.tensors["codes"].dtype == torch.uint8


def test_constant_input() -> None:
    latents = torch.full((2, 3, 4, 4), 1.5)
    codec = RangeQuantCodec(bits=4)
    assert torch.allclose(codec.decode(codec.encode(latents)), latents)


def test_accepts_low_precision_inputs_and_decode_dtype() -> None:
    latents = torch.randn(2, 3, 8, 8).to(torch.bfloat16)
    codec = RangeQuantCodec(bits=8)
    decoded = codec.decode(codec.encode(latents), dtype=torch.bfloat16)
    assert decoded.dtype == torch.bfloat16
    assert torch.allclose(decoded.float(), latents.float(), atol=0.05)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        RangeQuantCodec(bits=0)
    with pytest.raises(ValueError):
        RangeQuantCodec(bits=9)
    with pytest.raises(ValueError):
        RangeQuantCodec(channel_dim=0)
    with pytest.raises(ValueError):
        RangeQuantCodec(channel_dim=3).encode(torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        RangeQuantCodec(channel_dim=-3).encode(torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        RangeQuantCodec().encode(torch.randn(4))
