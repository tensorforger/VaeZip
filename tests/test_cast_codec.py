import pytest
import torch

from vaezip.codecs import CastCodec


@pytest.mark.parametrize(
    "cast_dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ],
)
def test_cast_roundtrip(cast_dtype: torch.dtype, device: torch.device) -> None:
    latents = torch.randn(2, 4, 8, 8, device=device)
    codec = CastCodec(cast_dtype=cast_dtype)
    encoded = codec.encode(latents)
    assert encoded.tensors["values"].dtype == cast_dtype
    assert encoded.tensors["values"].device.type == device.type
    decoded = codec.decode(encoded)
    assert decoded.dtype == torch.float32
    assert decoded.shape == latents.shape
    assert torch.equal(decoded, latents.to(cast_dtype).float())


def test_cast_clamps_out_of_range_values() -> None:
    latents = torch.tensor([[1.0e6, -1.0e6, 1.0]])
    decoded = CastCodec(torch.float8_e4m3fn).decode(
        CastCodec(torch.float8_e4m3fn).encode(latents)
    )
    assert torch.isfinite(decoded).all()
    assert decoded[0, 0] == 448.0
    assert decoded[0, 1] == -448.0


@pytest.mark.parametrize("input_shape", [(3, 5), (2, 3, 4), (1, 2, 3, 4, 5, 6)])
def test_cast_any_shape(input_shape: tuple[int, ...]) -> None:
    latents = torch.randn(*input_shape)
    codec = CastCodec(torch.bfloat16)
    assert codec.decode(codec.encode(latents)).shape == latents.shape


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_cast_accepts_input_dtypes(input_dtype: torch.dtype) -> None:
    latents = torch.randn(2, 3, 4).to(input_dtype)
    codec = CastCodec(torch.float16)
    decoded = codec.decode(codec.encode(latents))
    assert torch.allclose(decoded, latents.float(), atol=1e-2)


def test_decode_dtype() -> None:
    latents = torch.randn(2, 3, 4)
    codec = CastCodec(torch.float16)
    encoded = codec.encode(latents)
    assert codec.decode(encoded, dtype=torch.bfloat16).dtype == torch.bfloat16
    assert codec.decode(encoded, dtype=torch.float64).dtype == torch.float64
    with pytest.raises(TypeError):
        codec.decode(encoded, dtype=torch.int32)


def test_cast_accepts_dtype_name() -> None:
    assert CastCodec("float16").cast_dtype == torch.float16
    assert CastCodec("torch.bfloat16").cast_dtype == torch.bfloat16
    with pytest.raises(ValueError):
        CastCodec("not_a_dtype")
    with pytest.raises(ValueError):
        CastCodec(torch.int8)


def test_encode_rejects_invalid_input() -> None:
    codec = CastCodec()
    with pytest.raises(TypeError):
        codec.encode(torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError):
        codec.encode([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        codec.encode(torch.empty(0, 3))


def test_config_is_diffusers_style() -> None:
    config = CastCodec(torch.float8_e4m3fn).get_config()
    assert config["_class_name"] == "CastCodec"
    assert config["cast_dtype"] == "float8_e4m3fn"
    assert isinstance(config["_vaezip_version"], str)
