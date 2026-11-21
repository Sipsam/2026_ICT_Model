import os
import subprocess
import sys
from pathlib import Path

from multimodal_fixtures import make_sample


def test_sample_inspector_reads_only_the_selected_fixture(tmp_path: Path) -> None:
    row = make_sample(tmp_path, 1, values=(12345, 23456, 34567))
    script = Path(__file__).resolve().parents[1] / "scripts" / "inspect_sample.py"
    inputs = [
        tmp_path / row[f"{modality}_path"]
        for modality in ("image", "annotation", "sensor")
    ]
    original_bytes = [path.read_bytes() for path in inputs]
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--image",
            str(inputs[0]),
            "--annotation",
            str(inputs[1]),
            "--sensor",
            str(inputs[2]),
        ],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1"},
        check=True,
    )
    assert "'image': True, 'json': True, 'sensor': True" in result.stdout
    assert "Sensor XLSX" in result.stdout
    assert "34567" in result.stdout.split("Sensor XLSX", 1)[1]
    assert [path.read_bytes() for path in inputs] == original_bytes
