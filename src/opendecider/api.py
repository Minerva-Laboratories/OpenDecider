"""POST /v1/decide returns probabilities only; it has no text-generation path. POST /v1/explain (opt-in, separate)
generates an explanation of one decision with the same backbone.

    OPENDECIDER_CKPT=runs/<run>/model.pt python -m opendecider.api
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .schema import CalibrateRequest, DecideRequest, ExplainRequest


def create_app(decider) -> FastAPI:
    app = FastAPI(title="OpenDecider", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def _bad(_, exc):  # invalid schema -> 400 (not FastAPI's default 422)
        return JSONResponse(status_code=400, content={"error": "invalid request",
                                     "detail": jsonable_encoder(exc.errors(), custom_encoder={Exception: str})})

    @app.post("/v1/decide")
    def decide(req: DecideRequest):
        try:
            return decider.decide(req)
        except (ValidationError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/v1/calibrate")
    def calibrate(req: CalibrateRequest):
        from .calibration import fit_profile
        try:
            prof = fit_profile(decider, req, getattr(decider, "profiles_dir", "runs/profiles"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"id": prof["id"], "method": prof["method"], "temperature": prof["temperature"], "n": prof["n"],
                "cv": prof["cv"], "fit_in_sample": prof["fit_in_sample"]}

    @app.post("/v1/explain")
    def explain(req: ExplainRequest):
        from .explain import Explainer
        if req.question.type == "composite":
            raise HTTPException(400, "explain the atomic parts of a composite")
        ex = decider.__dict__.setdefault("_explainer", Explainer(decider))
        q = decider._atomic("q", req.question, {})
        out = ex.explain(req.state, q, q.names, n_samples=req.samples, min_faithfulness=req.min_faithfulness,
                         k_evidence=req.evidence)
        out.pop("candidates", None)
        return out

    @app.get("/healthz")
    def health():
        return {"ok": True}

    return app


def main():
    import uvicorn
    from .checkpoint import load_decider
    decider = load_decider(os.environ["OPENDECIDER_CKPT"])
    uvicorn.run(create_app(decider), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))


if __name__ == "__main__":
    main()
