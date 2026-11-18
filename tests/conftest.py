from pathlib import Path

import pytest

from ict_pipeline.environment.model import Graphics, Host, Resources
from ict_pipeline.reporting import metadata


@pytest.fixture(autouse=True)
def isolate_reporting_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run-tracking fixtures independent of OS tools and accelerators."""
    def hardware(host: Host, workspace: Path) -> tuple[Resources, Graphics]:
        return Resources("Fixture CPU", 2, None, None), Graphics("unknown", (), "unavailable", None)

    monkeypatch.setattr(metadata, "probe_hardware", hardware)
    monkeypatch.setattr(metadata, "run_probe", lambda _command: ("unknown", ""))
