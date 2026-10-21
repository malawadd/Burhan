"""Shared profile contract. Findings use the analysis input frame's coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class FrameContext:
    sequence: int
    captured_monotonic: float
    source_id: str = "unknown"
    session_id: str = "unknown"
    source_time_ms: float | None = None
    unit_id: str | None = None
    source_dimensions: tuple[int, int] | None = None  # width,height before bounded video resize
    analysis_dimensions: tuple[int, int] | None = None
    source_frame_index: int | None = None  # original decoded video index, before source skips
    source_timestamp_semantics: str | None = None


@dataclass
class Finding:
    kind: str
    label: str
    bbox: tuple[int, int, int, int] | None = None  # x, y, width, height
    score: float | None = None  # method specific, never universal confidence
    method: str = "opencv"
    severity: str = "info"
    task_id: str | None = None
    evidence_timestamp_ms: float | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    assessment_status: str | None = None


@dataclass
class InspectionResult:
    status: str
    title: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    guidance: str = ""
    action: str = "observe"
    annotated_frame: np.ndarray | None = None
    event_key: str | None = None
    emit_event: bool = False
    domain_state: dict[str, Any] = field(default_factory=dict)


class Profile(Protocol):
    def process(self, frame: np.ndarray, context: FrameContext) -> InspectionResult: ...
    def reset(self) -> None: ...
    def describe(self) -> dict[str, Any]: ...
