"""Lloyd-Max quantization tables for values normalized to [0, 1]."""

import torch

MAX_FIT_POINTS = 2**17
TINY = 1e-30


def nearest_level(values: torch.Tensor, tables: torch.Tensor) -> torch.Tensor:
    """
    args:
        values: [B, N], values: [0...1]
        tables: [B, K], sorted ascending
    returns:
        codes: [B, N], dtype: torch.uint8, index of the nearest table entry
    """
    midpoints = (tables[:, 1:] + tables[:, :-1]) / 2
    return torch.searchsorted(midpoints.contiguous(), values.contiguous()).to(
        torch.uint8
    )


def fit_lloyd_table(
    values: torch.Tensor,
    weights: torch.Tensor,
    levels: int,
    iterations: int = 20,
) -> torch.Tensor:
    """
    args:
        values: [N], values: [0...1]
        weights: [N], positive importance of each value
        levels: number of table entries
    returns:
        table: [levels], dtype: torch.float32, sorted, fp16-representable
    """
    stride = max(1, values.numel() // MAX_FIT_POINTS)
    values, weights = values[::stride], weights[::stride]
    weights = weights / weights.mean().clamp(min=TINY)
    table = torch.linspace(0, 1, levels, device=values.device)
    for _ in range(iterations):
        assignment = nearest_level(values.unsqueeze(0), table.unsqueeze(0))[0].long()
        weighted_sum = torch.zeros_like(table).index_add_(
            0, assignment, values * weights
        )
        weight_total = torch.zeros_like(table).index_add_(0, assignment, weights)
        table = torch.where(
            weight_total > 0, weighted_sum / weight_total.clamp(min=TINY), table
        )
    return table.to(torch.float16).float().sort().values


def fit_lloyd_tables_fast(
    values: torch.Tensor,
    weights: torch.Tensor,
    levels: int,
    iterations: int = 2,
    bins: int = 4096,
) -> torch.Tensor:
    """
    Batched approximation of `fit_lloyd_table` on a weighted histogram of the values.
    The start is companded quantiles, followed by a few Lloyd steps on bin centroids.

    args:
        values: [R, N], values: [0...1]
        weights: [R, N], positive importance of each value
        levels: number of table entries
    returns:
        tables: [R, levels], dtype: torch.float32, sorted, fp16-representable
    """
    rows = values.shape[0]
    weights = weights / weights.mean(dim=1, keepdim=True).clamp(min=TINY)
    bin_index = (values * bins).long().clamp_(0, bins - 1)
    mass = torch.zeros(rows, bins, device=values.device).scatter_add_(
        1, bin_index, weights
    )
    moment = torch.zeros_like(mass).scatter_add_(1, bin_index, values * weights)
    bin_centers = (torch.arange(bins, device=values.device) + 0.5) / bins
    centroid = torch.where(mass > 0, moment / mass.clamp(min=TINY), bin_centers)

    cumulative_mass = mass.pow(1 / 3).cumsum(dim=1)
    targets = (torch.arange(levels, device=values.device) + 0.5) / levels
    positions = torch.searchsorted(
        cumulative_mass / cumulative_mass[:, -1:],
        targets.expand(rows, -1).contiguous(),
    )
    tables = centroid.gather(1, positions.clamp(max=bins - 1))
    for _ in range(iterations):
        assignment = torch.searchsorted(
            ((tables[:, 1:] + tables[:, :-1]) / 2).contiguous(), centroid
        )
        weighted_sum = torch.zeros_like(tables).scatter_add_(1, assignment, moment)
        weight_total = torch.zeros_like(tables).scatter_add_(1, assignment, mass)
        tables = torch.where(
            weight_total > 0, weighted_sum / weight_total.clamp(min=TINY), tables
        )
    return tables.to(torch.float16).float().sort(dim=1).values
