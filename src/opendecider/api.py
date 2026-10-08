"""POST /v1/decide returns probabilities only; it has no text-generation path. POST /v1/explain (opt-in, separate)
generates an explanation of one decision with the same backbone.

    OPENDECIDER_CKPT=runs/<run>/model.pt python -m opendecider.api
    python -m opendecider.serve --model checkpoints/opendecider-9b --host 0.0.0.0 --api-key-file ... --log-dir ...

Serving extras (all optional, see opendecider.serve):
  api_key   requests to /v1/* need `Authorization: Bearer <key>` (or `X-API-Key`); /healthz and the playground page
            (GET /) stay open, the page asks for the key.
  log_dir   every /v1/decide is appended to requests.jsonl with an id (returned in the X-Request-Id header, the body
            is unchanged); POST /v1/feedback records the right answer for a logged request, including answers that
            were not among the options; GET /v1/feedback/export turns the feedback for one question into a
            /v1/calibrate request.
Concurrency: one thread owns the model (its caches, adapter flags and CUDA graphs are not thread-safe). Every
/v1/decide that is waiting when the GPU frees up joins the next batch (continuous batching, no artificial delay):
one state pass over all their states and one row pass per dependency wave (Decider.decide_many). A request that
arrives alone runs through Decider.decide and keeps the cross-request prefix cache. If a batch fails, its requests
are retried one by one so a bad request only fails itself.
"""
from __future__ import annotations

import contextlib
import fcntl
import hmac
import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from importlib import resources
from typing import Any, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .guards import free_gb
from .schema import CalibrateRequest, DecideRequest, ExplainRequest

LOG_FLOOR_GB = 4.0          # stop logging (never the serving) below this much free disk


class FeedbackRequest(BaseModel):
    """The right answer for one question of a logged /v1/decide request. `label` is an option/level name, a yes/no
    bool (noul) or a level index (score). When the right answer was not among the options, send `label: null` and
    `missing: "<what it should have been>"`: it is recorded, never forced onto a listed option."""
    model_config = ConfigDict(extra="forbid")
    request_id: str
    question: str
    label: Union[str, bool, int, None] = None
    missing: str | None = Field(None, max_length=500)
    note: str | None = Field(None, max_length=2000)


class _Log:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.req_path = os.path.join(log_dir, "requests.jsonl")
        self.fb_path = os.path.join(log_dir, "feedback.jsonl")
        self.warned = False

    @contextlib.contextmanager
    def _locked(self, path, mode):
        """flock, not a thread lock: with --workers N several processes append to the same files."""
        with open(path, mode) as f:
            fcntl.flock(f, fcntl.LOCK_EX if "a" in mode else fcntl.LOCK_SH)
            try:
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _append(self, path, row):
        if free_gb(path) < LOG_FLOOR_GB:
            if not self.warned:
                print(f"[opendecider] request log paused: under {LOG_FLOOR_GB} GB free", flush=True)
                self.warned = True
            return
        with self._locked(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def request(self, rid, req: dict, out: dict):
        self._append(self.req_path, {"id": rid, "ts": time.time(), "request": req, "response": out})

    def find_many(self, ids) -> dict:
        """Logged requests by id, in one pass over the log (the id is the first key of every row)."""
        want, got = set(ids), {}
        if not os.path.exists(self.req_path):
            return got
        with self._locked(self.req_path, "r") as f:
            for line in f:
                rid = line[8:24]
                if rid in want:
                    got[rid] = json.loads(line)
        return got

    def find(self, rid) -> dict | None:
        return self.find_many([rid]).get(rid)

    def feedback(self, row):
        self._append(self.fb_path, {"ts": time.time(), **row})

    def feedback_rows(self) -> list:
        if not os.path.exists(self.fb_path):
            return []
        with self._locked(self.fb_path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]


class ModelWorker:
    """The single owner of the decider. submit_decide() requests are batched; call() runs anything else alone."""

    def __init__(self, decider, max_batch: int = 16, max_batch_tokens: int = 16384):
        self.dec, self.max_batch, self.max_tokens = decider, max(1, max_batch), max_batch_tokens
        self.jobs: deque = deque()
        self.cv = threading.Condition()
        self.stats = {"batches": 0, "batched_requests": 0, "largest_batch": 0}
        threading.Thread(target=self._loop, daemon=True, name="opendecider-model").start()

    def _submit(self, kind, payload):
        fut = Future()
        with self.cv:
            self.jobs.append((kind, payload, fut))
            self.cv.notify()
        return fut.result()

    def submit_decide(self, req: DecideRequest) -> dict:
        return self._submit("decide", req)

    def call(self, fn):
        return self._submit("call", fn)

    @staticmethod
    def _key(req):
        return (req.sampler.mode, req.sampler.k, None if req.conformal is None else
                (req.conformal.alpha, req.conformal.method))

    @staticmethod
    def _size(req) -> int:
        """Rough state tokens (chars / 3), only to bound the padded batch."""
        st = req.state if isinstance(req.state, str) else json.dumps(req.state, default=str)
        return len(st) // 3 + sum(len(json.dumps(e.state, default=str)) // 3 for e in req.examples)

    def _take(self):
        """Next job; for a decide, also every queued decide with the same settings that fits the batch."""
        kind, payload, fut = self.jobs.popleft()
        if kind != "decide" or self.max_batch == 1:
            return kind, [(payload, fut)]
        group, key, longest = [(payload, fut)], self._key(payload), self._size(payload)
        rest = deque()
        while self.jobs:
            job = self.jobs.popleft()
            if len(group) < self.max_batch and job[0] == "decide" and self._key(job[1]) == key and \
                    max(longest, self._size(job[1])) * (len(group) + 1) <= self.max_tokens:
                group.append((job[1], job[2]))
                longest = max(longest, self._size(job[1]))
            else:
                rest.append(job)
        self.jobs.extendleft(reversed(rest))
        return kind, group

    def _loop(self):
        while True:
            with self.cv:
                while not self.jobs:
                    self.cv.wait()
                kind, group = self._take()
            if kind == "call":
                fn, fut = group[0]
                self._run(fut, fn)
            elif len(group) == 1:
                req, fut = group[0]
                self._run(fut, lambda: self.dec.decide(req))
            else:
                self.stats["batches"] += 1
                self.stats["batched_requests"] += len(group)
                self.stats["largest_batch"] = max(self.stats["largest_batch"], len(group))
                try:
                    outs = self.dec.decide_many([r for r, _ in group])
                except Exception:                            # isolate the failing request(s)
                    for req, fut in group:
                        self._run(fut, lambda req=req: self.dec.decide(req))
                    continue
                for (_, fut), out in zip(group, outs):
                    fut.set_result(out)

    @staticmethod
    def _run(fut, fn):
        try:
            fut.set_result(fn())
        except BaseException as e:                           # noqa: BLE001 -- delivered to the waiting request
            fut.set_exception(e)


def create_app(decider, api_key: str | None = None, log_dir: str | None = None, info: dict | None = None,
               max_batch: int = 16, max_batch_tokens: int = 16384) -> FastAPI:
    app = FastAPI(title="OpenDecider", version="0.1.0")
    worker = ModelWorker(decider, max_batch, max_batch_tokens)
    log = _Log(log_dir) if log_dir else None
    started = time.time()
    counts = {"decide": 0, "explain": 0, "calibrate": 0, "feedback": 0}

    def auth(request: Request):
        if not api_key:
            return
        given = request.headers.get("x-api-key") or ""
        h = request.headers.get("authorization") or ""
        if h.lower().startswith("bearer "):
            given = h[7:].strip()
        if not hmac.compare_digest(given.encode(), api_key.encode()):
            raise HTTPException(401, "missing or wrong API key")

    @app.exception_handler(RequestValidationError)
    async def _bad(_, exc):  # invalid schema -> 400 (not FastAPI's default 422)
        return JSONResponse(status_code=400, content={"error": "invalid request",
                                     "detail": jsonable_encoder(exc.errors(), custom_encoder={Exception: str})})

    @app.post("/v1/decide", dependencies=[Depends(auth)])
    def decide(req: DecideRequest, response: Response):
        try:
            out = worker.submit_decide(req)
        except (ValidationError, ValueError) as e:
            raise HTTPException(400, str(e))
        counts["decide"] += 1
        if log:
            rid = uuid.uuid4().hex[:16]
            log.request(rid, req.model_dump(mode="json", exclude_defaults=True), out)
            response.headers["X-Request-Id"] = rid
        return out

    @app.post("/v1/calibrate", dependencies=[Depends(auth)])
    def calibrate(req: CalibrateRequest):
        from .calibration import fit_profile
        try:
            prof = worker.call(lambda: fit_profile(decider, req, getattr(decider, "profiles_dir", "runs/profiles")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        counts["calibrate"] += 1
        return {"id": prof["id"], "method": prof["method"], "temperature": prof["temperature"], "n": prof["n"],
                "cv": prof["cv"], "fit_in_sample": prof["fit_in_sample"]}

    @app.post("/v1/explain", dependencies=[Depends(auth)])
    def explain(req: ExplainRequest):
        from .explain import Explainer
        if req.question.type == "composite":
            raise HTTPException(400, "explain the atomic parts of a composite")
        def run():
            ex = decider.__dict__.setdefault("_explainer", Explainer(decider))
            q = decider._atomic("q", req.question, {})
            return ex.explain(req.state, q, q.names, n_samples=req.samples, min_faithfulness=req.min_faithfulness,
                              k_evidence=req.evidence)
        out = worker.call(run)
        counts["explain"] += 1
        out.pop("candidates", None)
        return out

    @app.post("/v1/feedback", dependencies=[Depends(auth)])
    def feedback(fb: FeedbackRequest):
        if not log:
            raise HTTPException(400, "feedback needs the server's request log (--log-dir)")
        row = log.find(fb.request_id)
        if row is None:
            raise HTTPException(404, f"no logged request {fb.request_id}")
        if fb.question not in row["request"]["questions"]:
            raise HTTPException(400, f"request {fb.request_id} has no question {fb.question!r}")
        if fb.label is None and not fb.missing:
            raise HTTPException(400, "give a label, or `missing` when the right answer was not an option")
        log.feedback(fb.model_dump())
        counts["feedback"] += 1
        return {"ok": True}

    @app.get("/v1/feedback/export", dependencies=[Depends(auth)])
    def feedback_export(question: str, prompt: str | None = None):
        """Feedback for one question name as a ready /v1/calibrate body. The latest feedback per request wins.
        Requests whose question spec differs from the newest one (other options or prompt) are left out and counted,
        so a profile is never fitted on mixed questions; `missing` answers are counted, not fitted."""
        if not log:
            raise HTTPException(400, "feedback needs the server's request log (--log-dir)")
        latest = {}
        for fb in log.feedback_rows():
            if fb["question"] == question:
                latest[fb["request_id"]] = fb
        found = log.find_many(latest)
        rows = [(found.get(rid), fb) for rid, fb in latest.items()]
        rows = [(r, fb) for r, fb in rows if r is not None]
        if not rows:
            raise HTTPException(404, f"no feedback for question {question!r}")
        rows.sort(key=lambda x: x[0]["ts"])
        spec = rows[-1][0]["request"]["questions"][question]
        examples, other_spec, missing = [], 0, 0
        for r, fb in rows:
            if r["request"]["questions"][question] != spec:
                other_spec += 1
            elif fb["label"] is None:
                missing += 1
            else:
                examples.append({"state": r["request"]["state"], "label": fb["label"]})
        spec = {k: v for k, v in spec.items() if k != "calibration"}
        return {"calibrate_request": {"question": spec, "examples": examples},
                "n_examples": len(examples), "n_answer_not_listed": missing, "n_other_question_versions": other_spec,
                "ready": len(examples) >= 10}

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def server_info():
        return {**(info or {}), "uptime_s": round(time.time() - started), "requests": counts, "batching": worker.stats,
                "logging": bool(log), "auth": bool(api_key)}

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def playground():
        return resources.files("opendecider").joinpath("static/playground.html").read_text()

    return app


def main():
    import uvicorn
    from .checkpoint import load_decider
    decider = load_decider(os.environ["OPENDECIDER_CKPT"])
    uvicorn.run(create_app(decider), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
