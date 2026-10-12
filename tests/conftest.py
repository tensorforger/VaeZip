import pytest
import torch


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)
