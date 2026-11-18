# Inspection database

Set `ICT_API_CONFIG` to a JSON settings file and use the same file for the API and `ict history`. Paths are relative to the working directory. A minimal SQLite settings file is:

```json
{
  "storage_backend": "sqlite",
  "storage_root": "outputs/inspection-api",
  "database_name": "inspections.sqlite3",
  "retention_seconds": null
}
```

The file backend is the default for existing configurations. Both backends support history and statistics. The file backend scans JSON records, while SQLite uses indexes for time, status, mode, detector family, model ID, and defect class. Keep SQLite on a local disk in a private writable directory separate from source data and the model root. `database_name` is a single `.sqlite3` filename. Allow SQLite to manage its `-wal` and `-shm` sidecars. Use filesystem permissions to protect the database, its sidecars, and exports.

## Records and schema upgrades

Each inspection has an insert-only UUID key and a UTC creation timestamp with microsecond precision. The JSON payload carries both version 1 record envelopes, checkpoint and run fingerprints, model provenance, optional source and sensor identifiers, and the result. Detection rows preserve class IDs and names, source category IDs, confidence, and original-image bounds. Writes of the record and all its detections commit together. The storage layer works independently of detector initialization.

SQLite uses a separate application ID and `PRAGMA user_version`. Database schema version 1 creates the record and detection tables. Version 2 adds query indexes. Read/write open checks the identity and layout, then applies supported upgrades in one transaction. Unsupported versions and inconsistent schemas fail to open. Read-only history commands can read version 1 or 2 without applying upgrades. Back up the database before upgrading the application.

Each operation opens its own connection and enables foreign keys. The writable store uses WAL and waits up to five seconds by default for a lock. An operation that cannot acquire its lock fails and can be retried. List counts and rows come from the same SQLite read transaction. List, detail, and export validate the stored record they return. Statistics use the indexed columns and detection rows written by the same transactions. Keep database changes under the repository's control and use SQLite integrity and foreign-key checks when investigating storage problems.

The API stores sanitized source identifiers and checkpoint fingerprints. Use non-sensitive values for identifiers and optional metadata. Source images, checkpoint files, unprocessed sensor readings, and machine paths are unnecessary for the database record.

Input streams use stable UUIDs in the existing insert-only key. Restarting an unchanged replay can reuse a retained record even while retention hides it from history. An explicitly purged record no longer provides evidence for deduplication. [Replay recovery](input-streaming.md#queue-recovery-and-status)

Product inspection records store the version 1 `multimodal` block in the existing
record JSON. SQLite needs no migration. The modality filter reads submitted
evidence from JSON. A legacy record with an image descriptor matches image mode.
The detection table continues to store only spatial detections. Production
statistics exclude simulated, untrained, and unavailable product records.

## Browse and export

See the [API filter and response contracts](rest-api.md#history-filters-and-statistics) for the full filter list. The CLI takes shared filters before the subcommand.

```text
ict history --mode real --model-id line-3 list --limit 50 --offset 0
ict history --modality paired list --limit 50
ict history --mode mock --defect-class cold_solder --order oldest export outputs/mock-history --format csv
ict history --created-from 2026-01-01T00:00:00Z export outputs/january-history --format json
```

`list` emits a JSON page on stdout. `export` writes every matching visible record into a fresh directory and prints the export path. JSON exports contain `schema_version: 1` and `items`. CSV contains one inspection per row with the full versioned record in a JSON cell. A full SQLite export holds one read snapshot, so concurrent writers can continue and the WAL can grow until the export finishes. With the file backend, files published during a scan may appear in that operation.

## Import existing JSON records

Point `ICT_API_CONFIG` at the SQLite destination. If replacing an active file-backed service, complete a final import after stopping file writes, then start the service with the SQLite settings. Use the existing file repository directory as the source. Its published files have lowercase UUID hex names followed by `.json`.

```text
ict history import-files outputs/legacy-inspections --dry-run
ict history import-files outputs/legacy-inspections --apply
```

The default is a dry run. It validates both version fields, record identity, and contents, checks duplicate IDs against any existing destination, and reports prospective inserts. A preview with an absent database leaves it absent. SQLite may create transient WAL sidecars when reading an existing database.

`--apply` commits one valid new record at a time. The command can restart after interruption. An identical UUID and normalized record counts as a duplicate. A UUID with different content is reported as an ID conflict and the destination record takes precedence. Invalid files and conflicts are reported by UUID and result in exit status 1, while valid records still import. Source files remain in place, including records hidden by the source retention policy. The destination applies its own retention policy based on each original creation timestamp. Standalone `ict inspect` reports use a different envelope and remain local report artifacts.

## Retention

Set `retention_seconds` to a positive number of seconds to hide records once their age reaches the configured duration. The policy applies to both backends, including retrieval by ID, history, exports, and statistics. A null value makes records visible indefinitely. Retention is evaluated at read time, so increasing the duration can reveal records that still exist.

```text
ict history purge
ict history purge --approve-purge
```

Use `purge` to preview the affected SQLite database and its expired row count. `--approve-purge` authorizes deletion of expired inspection rows and their detection rows under the current policy. The count can change as new rows expire between preview and deletion. There is no automatic purge. Source records, images, and checkpoints stay in place. Deleting rows frees pages for SQLite to reuse. It does not guarantee file shrinkage or erase historical copies from backups.

## Backup and restore

Use SQLite's online backup API for a consistent snapshot of a live database. Copying only the live main file can miss committed WAL entries. Store the backup in a separate private location. This example reserves a fresh destination, copies the snapshot, and checks it. Keep completed backups only after both checks succeed.

```python
import sqlite3
from contextlib import closing
from pathlib import Path

source = Path("outputs/inspection-api/inspections.sqlite3").resolve(strict=True)
backup = Path("outputs/inspection-backups/history-001.sqlite3")
backup.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
backup.touch(mode=0o600, exist_ok=False)

with (
    closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as live,
    closing(sqlite3.connect(backup)) as snapshot,
):
    live.backup(snapshot)
    if snapshot.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("Backup failed SQLite integrity_check.")
    if snapshot.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("Backup failed SQLite foreign_key_check.")
```

To restore, stop writers, copy a verified backup into a fresh private storage root, and set `storage_root` and `database_name` to the restored location. Start the service to verify database compatibility. Check known inspection IDs and counts using the intended retention settings before resuming submissions. Keep the original database intact until restoration is verified. Back up file-backed stores by copying their published JSON records, preferably after pausing writes for a consistent set.
