"""Pydantic models for EpsiLint v3 API."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class DeploymentContext(BaseModel):
    n: int = 10000
    trust: str = "unknown"        # central | local | distributed | unknown
    sensitivity: str = "medium"   # high | medium | low
    unit: str = "event"           # user | event | unknown
    training_mode: str = "none"   # dpsgd | federated | none
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None


class AnalysisRequest(BaseModel):
    code: str
    context: DeploymentContext = DeploymentContext()


class RuleFinding(BaseModel):
    rule_id: str
    severity: str        # critical | warning | info | pass
    title: str
    detail: str
    source: str          # framework reference
    framework: str       # nist | w3c | vanderveen | diffmu | custom
    line: Optional[int] = None


class PyramidLayer(BaseModel):
    layer: str
    status: str          # pass | warn | fail | na
    detail: str


class CompositionEntry(BaseModel):
    query: str
    library: str
    epsilon: float
    cumulative: float
    line: int


class AnalysisResult(BaseModel):
    score: int
    score_label: str
    libraries: list[str]
    epsilons: list[dict]
    deltas: list[dict]
    mechanisms: list[dict]
    total_epsilon: float
    advanced_epsilon: float
    findings: list[RuleFinding]
    pyramid: list[PyramidLayer]
    composition: list[CompositionEntry]
    recommendations: list[dict]
    rule_summary: dict          # {framework: {pass: n, fail: n, ...}}


class RuleDefinition(BaseModel):
    id: str
    framework: str
    title: str
    description: str
    severity_on_match: str      # critical | warning | info
    severity_on_pass: str       # pass | info
    pattern: Optional[str] = None
    condition: str              # python expression evaluated against analysis state
    detail_template: str
    source: str
    enabled: bool = True
    tags: list[str] = []
