# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0]

First release.

### Added

- `Codec` abstract API and `EncodedTensor` container with `.safetensors` (default) and `.pth` serialization, optional metadata with the codec config.
- Codecs:
  - `CastCodec`: cast to a lower precision floating dtype.
  - `RangeQuantCodec`: per-channel min/max quantization to 1-8 bits.
  - `BlockRangeQuantCodec`: blockwise min/max quantization for any number of spatial dimensions, any channel position, padding for sizes not divisible by the block size.
  - `LloydBlockRangeQuantCodec`: the same with Lloyd-Max level tables, fast approximation and table caching for encoding.
- Bit-packed storage for all quantizing codecs.
- Diffusers-style JSON configs, `Codec.from_config()` and `AutoCodec.from_config()` with keyword overrides.
- Shortcut configs: `cast_fp32`, `cast_fp16`, `cast_bf16`, `cast_fp8_e4m3`, `rq_{8,6,4}bit`, `brq_{8,6,5,4,3,2}bit`, `lbrq_{8,6,5,4,3,2}bit`. Block shortcut parameters are taken from the LatentStorageBenchmark run 1.2 (MSE Pareto front on FLUX.2 latents).
- High level `vaezip.save()` and `vaezip.load()` for files and bytes, with a strict check that a user codec matches the stored config.
- `decode(dtype=...)` and `load(dtype=..., device=...)` to decode directly to a target dtype and device.
- Test suite, type annotations (`py.typed`), CI for tests, linting and build, and a PyPI publishing workflow.
