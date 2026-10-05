import numpy as np
import pytest

from psg import PROJECT_ROOT


@pytest.fixture
def outdir(request):
    """figures/tests/<test module>/<test name>/ : where a test writes its outcome plots."""
    d = PROJECT_ROOT / "figures" / "tests" / request.module.__name__ / request.node.name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def rng():
    return np.random.default_rng(0)
