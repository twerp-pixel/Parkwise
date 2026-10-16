"""HTTP API (Starlette + pydantic) and static dashboard host.

Endpoints
---------
GET  /api/health              liveness + artifact status
GET  /api/meta                zones, dates, guardrail defaults, assumptions
GET  /api/overview?date=      network KPIs, heatmap and per-zone recommendations
GET  /api/forecast?zone_id=&date=   hourly forecast with p10-p90 band (+ actuals in backtest)
POST /api/simulate            what-if pricing plan under guardrails
GET  /api/history?zone_id=    daily utilisation / revenue history
GET  /api/report              model health, elasticity, policy replay, ticket classifier
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import __version__
from . import config as C
from . import pricing as P
from .service import ArtifactsMissing, Engine

log = logging.getLogger("parkwise.api")
STATIC = Path(__file__).parent / "static"


class GuardrailsIn(BaseModel):
    min_mult: float = Field(0.70, ge=0.3, le=1.0, description="Lowest allowed hourly multiplier")
    max_mult: float = Field(1.40, ge=1.0, le=3.0, description="Highest allowed hourly multiplier")
    max_step: float = Field(0.15, ge=0.05, le=2.0, description="Max hour-to-hour change in multiplier")
    avg_cap: float = Field(1.15, ge=1.0, le=3.0, description="Cap on the daily average multiplier")
    turn_cost: float = Field(0.50, ge=0.0, le=5.0, description="Goodwill cost of a driver turned away at a full lot")
    deter_cost: float = Field(0.50, ge=0.0, le=5.0, description="Goodwill cost of a driver priced out")

    @model_validator(mode="after")
    def _consistent(self):
        if self.avg_cap > self.max_mult:
            raise ValueError("avg_cap cannot exceed max_mult")
        return self

    def to_domain(self) -> P.Guardrails:
        return P.Guardrails(**self.model_dump()).validate()


class SimRequest(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    strategy: Literal["status_quo", "flat", "optimized", "custom"] = "optimized"
    scenario: Literal["p10", "p50", "p90"] = "p50"
    guardrails: GuardrailsIn = GuardrailsIn()
    spill_share: float = Field(0.25, ge=0.0, le=0.8)
    locked_zones: list[int] = Field(default_factory=list, max_length=C.N_ZONES)
    adjustments: dict[int, float] = Field(default_factory=dict)

    @field_validator("locked_zones")
    @classmethod
    def _zones_ok(cls, v):
        if any(z < 0 or z >= C.N_ZONES for z in v):
            raise ValueError(f"zone ids must be within 0..{C.N_ZONES - 1}")
        return sorted(set(v))

    @field_validator("adjustments")
    @classmethod
    def _adj_ok(cls, v):
        for z, a in v.items():
            if not 0 <= z < C.N_ZONES:
                raise ValueError(f"zone id {z} out of range")
            if not -0.5 <= a <= 1.0:
                raise ValueError("adjustments must be within [-0.5, 1.0]")
        return v


def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status)


def create_app(art_dir: Path | str | None = None) -> Starlette:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app):
        try:
            state["engine"] = Engine(art_dir or C.ARTIFACTS)
            log.info("artifacts loaded; live date %s", state["engine"].live_date)
        except ArtifactsMissing as e:
            state["error"] = str(e)
            log.error(str(e))
        yield

    def engine() -> Engine:
        if "engine" not in state:
            raise HTTPException(503, state.get("error", "engine not ready"))
        return state["engine"]

    def date_param(request: Request, eng: Engine) -> str:
        d = request.query_params.get("date") or eng.live_date
        if d not in eng.days:
            raise HTTPException(404, f"date {d} unavailable; valid: {eng.days[0]} .. {eng.days[-1]}")
        return d

    def zone_param(request: Request) -> int:
        try:
            z = int(request.query_params.get("zone_id", ""))
        except ValueError:
            raise HTTPException(422, "zone_id must be an integer")
        if not 0 <= z < C.N_ZONES:
            raise HTTPException(404, f"zone_id must be within 0..{C.N_ZONES - 1}")
        return z

    async def health(request: Request):
        ok = "engine" in state
        return JSONResponse({"status": "ok" if ok else "degraded", "version": __version__,
                             "artifacts_loaded": ok, "detail": state.get("error")}, status_code=200 if ok else 503)

    async def meta(request: Request):
        return JSONResponse(engine().meta())

    async def overview(request: Request):
        eng = engine()
        return JSONResponse(eng.overview(date_param(request, eng)))

    async def forecast(request: Request):
        eng = engine()
        return JSONResponse(eng.forecast(zone_param(request), date_param(request, eng)))

    async def history(request: Request):
        return JSONResponse(engine().history(zone_param(request)))

    async def report(request: Request):
        return JSONResponse(engine().report)

    async def simulate(request: Request):
        eng = engine()
        try:
            req = SimRequest.model_validate(await request.json())
        except ValidationError as e:
            return _err(422, "validation_error", "Invalid request body", e.errors(include_url=False, include_context=False))
        except ValueError:
            return _err(400, "bad_json", "Request body must be valid JSON")
        if req.date not in eng.days:
            return _err(404, "date_unavailable", f"date {req.date} unavailable; valid: {eng.days[0]} .. {eng.days[-1]}")
        try:
            out = eng.simulate(req.date, req.strategy, req.scenario, req.guardrails.to_domain(), req.spill_share,
                               req.locked_zones, req.adjustments)
        except ValueError as e:
            return _err(422, "invalid_guardrails", str(e))
        return JSONResponse(out)

    async def index(request: Request):
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    async def on_http_exc(request: Request, exc: HTTPException):
        code = {404: "not_found", 422: "unprocessable", 503: "unavailable"}.get(exc.status_code, "error")
        return _err(exc.status_code, code, str(exc.detail))

    async def on_unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return _err(500, "internal_error", "Unexpected server error")

    class Instrument:
        """Adds request ids, timing and conservative security headers."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.app(scope, receive, send)
            rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()

            async def send_wrapped(message):
                if message["type"] == "http.response.start":
                    h = message.setdefault("headers", [])
                    h += [(b"x-request-id", rid.encode()),
                          (b"server-timing", f"app;dur={(time.perf_counter() - t0) * 1000:.1f}".encode()),
                          (b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer"),
                          (b"content-security-policy", b"default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:")]
                await send(message)

            await self.app(scope, receive, send_wrapped)

    routes = [
        Route("/api/health", health), Route("/api/meta", meta), Route("/api/overview", overview),
        Route("/api/forecast", forecast), Route("/api/history", history), Route("/api/report", report),
        Route("/api/simulate", simulate, methods=["POST"]),
        Route("/", index),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    return Starlette(routes=routes, lifespan=lifespan,
                     middleware=[Middleware(Instrument), Middleware(GZipMiddleware, minimum_size=1024)],
                     exception_handlers={HTTPException: on_http_exc, 500: on_unhandled})


app = create_app()
