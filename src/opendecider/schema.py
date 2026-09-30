"""Request schema for /v1/decide (mirrors the public Jev interface *shape*; not TypeSafe's code)."""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 256
MAX_OPTIONS_TOTAL = 4096   # >255 goes through the two-stage path


class OptionSpec(BaseModel):
    """An option / level with an optional description (criteria for picking it). Answers are keyed by `name`."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str | None = None


OptionLike = Union[str, OptionSpec]


def opt_name(o) -> str:
    return o if isinstance(o, str) else o.name


def opt_text(o) -> str:
    """What the model reads for this option: `name: description` when a description is given."""
    if isinstance(o, str) or not o.description:
        return opt_name(o)
    return f"{o.name}: {o.description}"


class _Q(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    instructions: str | None = None          # exact conditions / boundary cases, read with the question
    calibration: str | None = None           # id of a fitted calibration profile (POST /v1/calibrate)
    none: bool | None = None                 # report P(none) (models with the `none` sink); False: closed world


class ChoiceQ(_Q):
    type: Literal["choice"]
    options: list[OptionLike] = Field(min_length=2, max_length=MAX_OPTIONS_TOTAL)

    @field_validator("options")
    @classmethod
    def _opts(cls, v):
        names = [opt_name(o) for o in v]
        if any(not n.strip() for n in names):
            raise ValueError("options must be non-empty")
        return v


class NoulQ(_Q):
    type: Literal["noul"]
    criteria: str | None = None              # what counts as "yes"


class ScoreQ(_Q):
    type: Literal["score"]
    levels: list[OptionLike] = Field(min_length=2, max_length=255)


AtomicSpec = Annotated[Union[ChoiceQ, NoulQ, ScoreQ], Field(discriminator="type")]

REF = re.compile(r"\{([A-Za-z0-9_]+)\}")


class CombineSpec(BaseModel):
    """p_yes = sigmoid(bias + sum_f weights[f] * feature_f). Features: "<part>" = P(yes) of a noul part;
    "<part>=<option>" = probability of that option of a choice/score part; "<part>.expected" = score expected index."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["logistic"] = "logistic"
    weights: dict[str, float] = Field(min_length=1)
    bias: float = 0.0


class CompositeQ(_Q):
    """Caller-side decomposition: several atomic sub-questions (answered in the same batch against the same state)
    combined into one calibrated yes/no probability. `prompt` describes the composite (not sent to the model)."""
    type: Literal["composite"]
    parts: dict[str, AtomicSpec] = Field(min_length=1, max_length=32)
    combine: CombineSpec

    @model_validator(mode="after")
    def _features(self):
        for f in self.combine.weights:
            base = f.split("=", 1)[0].split(".", 1)[0]
            if base not in self.parts:
                raise ValueError(f"combine feature {f!r} refers to unknown part {base!r}")
        return self


QuestionSpec = Annotated[Union[ChoiceQ, NoulQ, ScoreQ, CompositeQ], Field(discriminator="type")]


class SamplerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: int = Field(1, ge=1, le=16)
    mode: Literal["mc_dropout", "gaussian_noise"] = "mc_dropout"


class ContextExample(BaseModel):
    """An in-context example: another state, with answers to (some of) the same questions. Answers present =
    labelled in-context learning; answers absent/None = unlabelled context (e.g. other items for comparison)."""
    model_config = ConfigDict(extra="forbid")
    state: Any
    answers: dict[str, Union[str, bool, int, None]] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Any
    questions: dict[str, QuestionSpec] = Field(min_length=1, max_length=MAX_QUESTIONS)
    sampler: SamplerSpec = SamplerSpec()
    examples: list[ContextExample] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _state(self):
        if self.state is None or (isinstance(self.state, str) and not self.state.strip()):
            raise ValueError("state must be a non-empty string or JSON value")
        waves(self.questions)            # validates references: known names, no cycles
        return self


def refs_of(spec) -> set[str]:
    """Names of top-level questions whose chosen value is substituted into this question's prompt(s)."""
    out = set(REF.findall(spec.prompt)) if spec.type != "composite" else set()
    if spec.type == "composite":
        for part in spec.parts.values():
            out |= set(REF.findall(part.prompt))
    return out


def waves(questions: dict) -> list[list[str]]:
    """Dependency levels for sequential chaining: wave k only references answers from waves < k."""
    deps = {n: refs_of(q) for n, q in questions.items()}
    for n, d in deps.items():
        bad = d - set(questions)
        if bad:
            raise ValueError(f"question {n!r} references unknown question(s) {sorted(bad)}")
        if n in d:
            raise ValueError(f"question {n!r} references itself")
    done, out = set(), []
    while len(done) < len(questions):
        w = [n for n in questions if n not in done and deps[n] <= done]
        if not w:
            raise ValueError("circular question references")
        out.append(w)
        done |= set(w)
    return out


class CalibrationExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Any
    label: Union[str, bool, int]             # option/level name, yes/no bool (noul), or level index (score)


class ExplainRequest(BaseModel):
    """POST /v1/explain (opt-in, separate from /v1/decide): explain ONE atomic question's decision with the backbone."""
    model_config = ConfigDict(extra="forbid")
    state: Any
    question: QuestionSpec
    samples: int = Field(1, ge=1, le=8)                    # >1: best-of-N by faithfulness
    min_faithfulness: float = Field(0.0, ge=0.0, le=1.0)   # below this the explanation is withheld (null)
    evidence: int = Field(3, ge=0, le=10)


class CalibrateRequest(BaseModel):
    """Fit a per-question calibration profile from labelled examples (the per-deployment decision head)."""
    model_config = ConfigDict(extra="forbid")
    question: QuestionSpec
    examples: list[CalibrationExample] = Field(min_length=10, max_length=5000)
    # auto: CV-select among identity / shrunk temperature / beta / ets / vector / histogram (binary) / iso_shrunk
    # (src/opendecider/calibrators.py). temperature, vector: the plain logit-level fits (vector = T + per-option bias)
    method: Literal["auto", "temperature", "vector", "identity", "beta", "ets", "histogram", "iso_shrunk"] = "auto"
