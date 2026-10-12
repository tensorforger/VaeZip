import pytest
import torch

from vaezip.codecs import BlockRangeQuantCodec, LloydBlockRangeQuantCodec


def mean_squared_error(codec: BlockRangeQuantCodec, latents: torch.Tensor) -> float:
    return float((codec.decode(codec.encode(latents)) - latents).pow(2).mean())


@pytest.mark.parametrize("lloyd_fast", [True, False])
@pytest.mark.parametrize("value_bits", [2, 3, 4])
def test_lloyd_levels_beat_uniform_levels_on_gaussian_data(
    lloyd_fast: bool, value_bits: int
) -> None:
    latents = torch.randn(2, 8, 32, 32)
    uniform = BlockRangeQuantCodec(value_bits=value_bits, block_size=(8, 8))
    lloyd = LloydBlockRangeQuantCodec(
        value_bits=value_bits, block_size=(8, 8), lloyd_fast=lloyd_fast
    )
    assert mean_squared_error(lloyd, latents) < mean_squared_error(uniform, latents)


@pytest.mark.parametrize("lloyd_fast", [True, False])
def test_table_is_stored_and_sorted(lloyd_fast: bool) -> None:
    latents = torch.randn(3, 4, 16, 16)
    codec = LloydBlockRangeQuantCodec(value_bits=3, lloyd_fast=lloyd_fast)
    table = codec.encode(latents).tensors["table"]
    assert table.shape == (3, 8)
    assert table.dtype == torch.float16
    assert (table.diff(dim=1) >= 0).all()
    assert table.min() >= 0 and table.max() <= 1


@pytest.mark.parametrize("cache_table", [True, False])
def test_decoding_is_independent_of_the_encoding_codec(cache_table: bool) -> None:
    latents = torch.randn(2, 4, 16, 16)
    encoder = LloydBlockRangeQuantCodec(value_bits=4, cache_table=cache_table)
    decoder = LloydBlockRangeQuantCodec(value_bits=4)
    encoded = encoder.encode(latents)
    assert torch.equal(decoder.decode(encoded), encoder.decode(encoded))
    assert "table" in encoded.tensors


def test_cached_table_is_reused_until_flushed() -> None:
    codec = LloydBlockRangeQuantCodec(value_bits=3, cache_table=True)
    first = codec.encode(torch.randn(2, 4, 16, 16))
    second = codec.encode(torch.rand(3, 4, 16, 16) * 5)
    assert second.tensors["table"].shape == (3, 8)
    for sample_table in second.tensors["table"]:
        assert torch.equal(sample_table, first.tensors["table"][0])
    assert torch.equal(first.tensors["table"][0], first.tensors["table"][1])

    codec.flush_cache()
    third = codec.encode(torch.rand(2, 4, 16, 16))
    assert not torch.equal(third.tensors["table"][0], first.tensors["table"][0])


def test_without_cache_tables_follow_each_sample() -> None:
    codec = LloydBlockRangeQuantCodec(value_bits=3, cache_table=False)
    latents = torch.cat([torch.randn(1, 4, 16, 16), torch.rand(1, 4, 16, 16) * 5])
    table = codec.encode(latents).tensors["table"]
    assert not torch.equal(table[0], table[1])


def test_flush_cache_without_cache_is_safe() -> None:
    codec = LloydBlockRangeQuantCodec()
    codec.flush_cache()
    codec.encode(torch.randn(1, 2, 8, 8))


def test_cache_with_different_batch_sizes_and_shapes() -> None:
    codec = LloydBlockRangeQuantCodec(value_bits=4, cache_table=True)
    for shape in [(1, 4, 16, 16), (3, 4, 17, 9), (2, 2, 8, 8)]:
        latents = torch.randn(*shape)
        decoded = codec.decode(codec.encode(latents))
        assert decoded.shape == latents.shape


def test_lloyd_on_device(device: torch.device) -> None:
    latents = torch.randn(2, 4, 16, 16, device=device)
    codec = LloydBlockRangeQuantCodec(value_bits=5, cache_table=True)
    encoded = codec.encode(latents)
    assert all(tensor.device.type == device.type for tensor in encoded.tensors.values())
    assert codec.decode(encoded).device.type == device.type


def test_config_contains_lloyd_options() -> None:
    parameters = LloydBlockRangeQuantCodec(
        lloyd_fast=False, cache_table=True
    ).get_config_parameters()
    assert parameters["lloyd_fast"] is False
    assert parameters["cache_table"] is True
