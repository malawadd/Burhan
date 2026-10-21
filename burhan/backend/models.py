from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal['crack', 'surface', 'ppe', 'manual']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Scene(Strict):
    environment: Literal['indoor', 'outdoor', 'transitional', 'unknown']
    description: str = Field(min_length=1, max_length=600)
    confidence: float = Field(ge=0, le=1)


class Suggestion(Strict):
    taskId: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=200)
    mode: Mode
    reason: str = Field(min_length=1, max_length=600)
    automatic: bool


class Action(Strict):
    id: str = Field(min_length=1, max_length=100)
    type: Literal['select_detector', 'request_view', 'request_manual_check', 'ask_context', 'prepare_finding']
    mode: Mode | None
    taskId: str | None
    instruction: str = Field(min_length=1, max_length=600)


class Plan(Strict):
    scene: Scene
    guidance: str = Field(min_length=1, max_length=800)
    # Optional for older clients/providers; the orchestrator binds missing values.
    guidanceTaskId: str | None = Field(default=None, max_length=160)
    reason: str = Field(min_length=1, max_length=800)
    suggestions: list[Suggestion] = Field(max_length=3)
    actions: list[Action] = Field(max_length=4)
    needsHumanReview: Literal[True]


class Pair(Strict):
    code: str = Field(min_length=6, max_length=100)
    deviceName: str = Field(min_length=1, max_length=100)


class NewSession(Strict):
    inspectionId: str = Field(min_length=1, max_length=200)
    language: Literal['en', 'ar'] = 'en'
    mode: Mode = 'manual'
    stage: str = Field(default='', max_length=160)
    taskId: str = Field(default='', max_length=160)
    location: str = Field(default='', max_length=500)


class Context(BaseModel):
    model_config = ConfigDict(extra='ignore')
    mode: Mode = 'manual'
    locked: bool = False
    activeTaskId: str = Field(default='', max_length=160)
    selectionMode: Literal['legacy', 'confirm'] = 'legacy'
    selectionConfirmed: bool = False
    dismissedTaskIds: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(default_factory=list, max_length=100)
    completedTaskIds: list[Annotated[str, Field(max_length=160)]] = Field(default_factory=list, max_length=100)
    correction: str = Field(default='', max_length=2000)
    evidenceCount: int = Field(default=0, ge=0, le=100000)
    observations: list[dict] = Field(default_factory=list, max_length=100)
    quality: dict = Field(default_factory=dict)


class Turn(Strict):
    image: str | None = Field(default=None, max_length=6_000_000)
    sceneVersion: int = Field(ge=0, le=1_000_000_000)
    context: Context = Field(default_factory=Context)
    # The UI may retry exactly this request without spending another model turn.
    requestId: str | None = Field(default=None, max_length=100)


class ToolResult(Strict):
    actionId: str = Field(min_length=1, max_length=100)
    status: Literal['completed', 'skipped', 'failed', 'accepted', 'rejected']
    detail: str = Field(default='', max_length=2000)


class ToolResults(Strict):
    turnId: str
    sceneVersion: int = Field(ge=0)
    results: list[ToolResult] = Field(max_length=10)


class CheckEvent(Strict):
    eventId: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_.:-]+$')
    type: Literal['proposed', 'selected', 'dismissed', 'stopped', 'completed', 'skipped']
    taskId: str = Field(min_length=1, max_length=160)
    sceneVersion: int = Field(ge=0, le=1_000_000_000)
    sourceTurnId: str | None = Field(default=None, min_length=1, max_length=100)
