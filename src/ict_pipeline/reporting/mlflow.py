"""Optional adapter for publishing a sanitized report snapshot."""

import os
from contextlib import suppress
from pathlib import Path

from ..config.settings import SetupError
from ..config.storage import SetupFiles
from .catalog import Run


def export_to_mlflow(workspace: Path, experiment_id: str, document: Run) -> None:
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    uri = credentials.values.get("MLFLOW_TRACKING_URI")
    if uri is None:
        raise SetupError("Set MLFLOW_TRACKING_URI in the environment or .env before exporting.")
    credentials.protect_path(experiment_id, "MLflow experiment ID")
    try:
        from mlflow import MlflowClient
    except ImportError:
        raise SetupError("MLflow export requires the optional ict-pipeline[mlflow] dependency.") from None

    try:
        client = MlflowClient(tracking_uri=uri.get_secret_value())
        run = client.create_run(experiment_id, run_name="ICT comparison report")
        run_id = run.info.run_id
        try:
            client.log_dict(run_id, credentials.redact(document), "comparison.json")
        except (Exception, KeyboardInterrupt) as exc:
            with suppress(Exception):
                client.set_terminated(run_id, status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
            raise
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        raise SetupError("MLflow export failed. Check the tracking service, experiment ID and permissions.") from None
