import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.evsim.generator import generate_dataset
from src.evsim.validation import validate_dataset


def test_invalid_soc_is_rejected():
    dataset = generate_dataset(num_vehicles=1, num_trips=1, interval_s=60)
    dataset.telemetry[0]["soc_pct"] = 120
    with pytest.raises(ValueError):
        validate_dataset(dataset)
