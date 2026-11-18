"""Public contracts for local pipeline stages."""

from .model import PipelineError, Stage, StageContext, StageInput, StageResult
from .runner import StageRunner

__all__ = ["PipelineError", "Stage", "StageContext", "StageInput", "StageResult", "StageRunner"]
