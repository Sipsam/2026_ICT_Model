"""Non-interactive JSON protocol for the installed target runtime."""

import json
import os
import platform
import sys
from contextlib import redirect_stderr, redirect_stdout

from pydantic import ValidationError

from ..config.settings import Credentials
from .jobs import _SAFE_ERRORS, dispatch
from .model import ExecutorError, WorkerRequest


def main() -> int:
    credentials = Credentials.load({}, os.environ)
    try:
        if len(sys.argv) == 4:
            if platform.system() != "Linux":
                raise ExecutorError("Detached workers require Linux.")
            descriptor = int(sys.argv[1])
            with os.fdopen(descriptor, "rb"):
                os.set_inheritable(descriptor, False)
                dispatch(WorkerRequest(action="work", workspace=sys.argv[2], job=sys.argv[3]), credentials)
            return 0
        ssh = sys.argv[1:] == ["--ssh"]
        if len(sys.argv) != 1 and not ssh:
            raise ExecutorError("Invalid worker invocation.")
        raw = sys.stdin.buffer.read(16385)
        if len(raw) > 16384:
            raise ExecutorError("Worker request exceeds the size limit.")
        request = WorkerRequest.model_validate_json(raw)
        if request.action == "work":
            raise ExecutorError("Invalid worker request.")
        if (ssh or request.action in {"submit", "resume"}) and platform.system() != "Linux":
            raise ExecutorError("SSH jobs require a Linux target.")
        # Reserve stdout for the protocol. Persist only structured stage events.
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            result = dispatch(request, credentials)
        print(json.dumps({"schema_version": 1, "result": credentials.redact(result)}, ensure_ascii=True))
    except _SAFE_ERRORS as exc:
        print(json.dumps({"schema_version": 1, "error": credentials.redact_text(str(exc))}))
    except (OSError, ValidationError, ValueError, RuntimeError):
        print(json.dumps({"schema_version": 1, "error": "Invalid worker request or inaccessible target files."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
