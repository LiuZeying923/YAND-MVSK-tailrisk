"""FastAPI application: serves the JSON API and the static frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..behavioral import PROFILE_BANDS, BehavioralGammaOptimizer, questionnaire_schema
from ..data import catalog
from .schemas import GammaRequest, HealthResponse, OptimizeRequest
from .service import optimize_portfolio

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

app = FastAPI(
    title="YAND-MVSK Tail-Risk Studio",
    version=__version__,
    description="Higher-moment portfolio optimization with behavioral gamma calibration "
                "and a tail-risk guard, across HK / US / A-share markets.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/api/questionnaire")
def get_questionnaire() -> dict:
    return {
        "questions": questionnaire_schema(),
        "profiles": [
            {"label": name, "gamma_low": lo, "gamma_high": hi, "blurb": blurb}
            for (lo, hi, name, blurb) in PROFILE_BANDS
        ],
    }


@app.post("/api/gamma")
def post_gamma(req: GammaRequest) -> dict:
    try:
        bd = BehavioralGammaOptimizer(req.base_gamma).calculate(req.answers)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return bd.as_dict()


@app.get("/api/catalog")
def get_catalog() -> dict:
    return {"assets": catalog.as_records(), "markets": catalog.MARKET_LABELS}


@app.get("/api/search")
def get_search(q: str = Query("", description="query string"), limit: int = 12) -> dict:
    return {"results": catalog.search(q, limit=limit)}


@app.post("/api/optimize")
def post_optimize(req: OptimizeRequest) -> dict:
    try:
        return optimize_portfolio(
            tickers=req.tickers,
            gamma=req.gamma,
            answers=req.answers,
            base_gamma=req.base_gamma,
            max_weight=req.max_weight,
            horizon=req.horizon,
            lookback=req.lookback,
            offline=req.offline,
            apply_guard=req.apply_guard,
            rf=req.rf,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # data/solver failures -> 500 with a readable message
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---- static frontend (mounted last so /api/* wins) -----------------------
if _FRONTEND_DIR.is_dir():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="static")
