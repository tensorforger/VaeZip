"""Packing of low-bit integer codes into bytes."""

import torch
import torch.nn.functional as functional

BITS_PER_BYTE = 8


def check_bits(bits: int, name: str = "bits") -> None:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"{name} must be an int, got {type(bits).__name__}")
    if not 1 <= bits <= BITS_PER_BYTE:
        raise ValueError(f"{name} must be in [1, {BITS_PER_BYTE}], got {bits}")


def max_code(bits: int) -> int:
    return (1 << bits) - 1


def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """
    args:
        codes: [B, N], dtype: torch.uint8, values: [0...2^bits - 1]
        bits: number of bits per code, in [1...8]
    returns:
        packed: [B, ceil(N * bits / 8)], dtype: torch.uint8, codes stored LSB first
    """
    if bits == BITS_PER_BYTE:
        return codes.contiguous()
    batch_size = codes.shape[0]
    bit_shifts = torch.arange(bits, device=codes.device, dtype=torch.uint8)
    bit_planes = ((codes.unsqueeze(-1) >> bit_shifts) & 1).reshape(batch_size, -1)
    bit_planes = functional.pad(bit_planes, (0, (-bit_planes.shape[1]) % BITS_PER_BYTE))
    byte_weights = 1 << torch.arange(
        BITS_PER_BYTE, device=codes.device, dtype=torch.uint8
    )
    byte_groups = bit_planes.reshape(batch_size, -1, BITS_PER_BYTE) * byte_weights
    return byte_groups.sum(dim=-1, dtype=torch.uint8)


def unpack_bits(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """
    args:
        packed: [B, ceil(count * bits / 8)], dtype: torch.uint8
        count: number of codes per sample
        bits: number of bits per code, in [1...8]
    returns:
        codes: [B, count], dtype: torch.uint8, values: [0...2^bits - 1]
    """
    if bits == BITS_PER_BYTE:
        return packed[:, :count]
    batch_size = packed.shape[0]
    bit_shifts = torch.arange(BITS_PER_BYTE, device=packed.device, dtype=torch.uint8)
    bit_planes = ((packed.unsqueeze(-1) >> bit_shifts) & 1).reshape(batch_size, -1)
    bit_planes = bit_planes[:, : count * bits]
    code_weights = 1 << torch.arange(bits, device=packed.device, dtype=torch.uint8)
    code_groups = bit_planes.reshape(batch_size, count, bits) * code_weights
    return code_groups.sum(dim=-1, dtype=torch.uint8)
