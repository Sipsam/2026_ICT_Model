import hashlib
from pathlib import Path

from ict_pipeline.data.preprocessing.audit import sha256
from ict_pipeline.data.preprocessing.leakage import file_sha
from ict_pipeline.pipeline.store import fingerprint_path


def test_audit_and_leakage_hash_raw_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    payload = bytes(range(256)) * 4097
    path.write_bytes(payload)
    assert sha256(path) == file_sha(path) == hashlib.sha256(payload).hexdigest()
    assert fingerprint_path(path) != file_sha(path)
