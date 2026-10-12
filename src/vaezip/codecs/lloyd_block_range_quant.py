"""Blockwise range quantization with Lloyd-Max level tables."""

from collections.abc import Sequence
from typing import Any

import torch

from vaezip.codecs.block_range_quant import BlockRangeQuantCodec
from vaezip.utils.lloyd import fit_lloyd_table, fit_lloyd_tables_fast, nearest_level

TABLE_KEY = "table"


class LloydBlockRangeQuantCodec(BlockRangeQuantCodec):
    """
    `BlockRangeQuantCodec` where the levels inside each block range are not uniform but
    follow a Lloyd-Max table fitted to the values of each sample and stored in the
    encoded tensor.

    lloyd_fast: fit the tables with a fast histogram-based approximation.
    cache_table: fit one table on the first encoded batch and reuse it for all next
    calls. This speeds up encoding only, decoding always uses the table stored in the
    encoded tensor.
    """

    def __init__(
        self,
        value_bits: int = 4,
        range_bits: int = 4,
        channel_dim: int = 1,
        block_size: Sequence[int] = (4, 4),
        lloyd_fast: bool = True,
        cache_table: bool = False,
    ) -> None:
        super().__init__(
            value_bits=value_bits,
            range_bits=range_bits,
            channel_dim=channel_dim,
            block_size=block_size,
        )
        self.lloyd_fast = lloyd_fast
        self.cache_table = cache_table
        self._cached_table: torch.Tensor | None = None

    def flush_cache(self) -> None:
        """Drops the cached table, the next encoded batch fits a new one."""
        self._cached_table = None

    def _quantize_normalized(
        self, normalized: torch.Tensor, span: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        args:
            normalized: [B, C, G, V], values: [0...1], position inside the block range
            span: [B, C, G, 1], width of the block range
        returns:
            codes: [B, C, G, V], dtype: torch.uint8, values: [0...2^value_bits - 1]
            tensors: table: [B, 2^value_bits], dtype: torch.float16, sorted, [0...1]
        """
        batch_size = normalized.shape[0]
        values = normalized.reshape(batch_size, -1)
        if self._cached_table is not None:
            tables = self._cached_table.to(values.device).unsqueeze(0)
        else:
            weights = (span.expand_as(normalized) ** 2).reshape(batch_size, -1)
            if self.cache_table:
                tables = self._fit_tables(values.reshape(1, -1), weights.reshape(1, -1))
                self._cached_table = tables[0].clone()
            else:
                tables = self._fit_tables(values, weights)
        tables = tables.expand(batch_size, -1).contiguous()
        codes = nearest_level(values, tables).reshape(normalized.shape)
        return codes, {TABLE_KEY: tables.to(torch.float16)}

    def _dequantize_normalized(
        self, codes: torch.Tensor, tensors: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        args:
            codes: [B, C, G, V], dtype: torch.uint8
            tensors: output of `_encode_tensors()`, table: [B, 2^value_bits]
        returns:
            normalized: [B, C, G, V], dtype: torch.float32, values: [0...1]
        """
        batch_size = codes.shape[0]
        table = tensors[TABLE_KEY].float()
        levels = table.gather(1, codes.reshape(batch_size, -1).long())
        return levels.reshape(codes.shape)

    def _fit_tables(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        args:
            values, weights: [R, N]
        returns:
            tables: [R, 2^value_bits], dtype: torch.float32
        """
        levels = 2**self.value_bits
        if self.lloyd_fast:
            return fit_lloyd_tables_fast(values, weights, levels)
        return torch.stack(
            [
                fit_lloyd_table(row_values, row_weights, levels)
                for row_values, row_weights in zip(values, weights, strict=True)
            ]
        )

    def get_config_parameters(self) -> dict[str, Any]:
        return {
            **super().get_config_parameters(),
            "lloyd_fast": self.lloyd_fast,
            "cache_table": self.cache_table,
        }
