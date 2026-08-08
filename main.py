"""
Structural Reliability Service
================================
FastAPI microservice that computes structural reliability (Pf, beta)
via direct Monte Carlo simulation, using verified closed-form capacity
and demand formulas supplied by the Recipe Agent (not hardcoded here).

Endpoints:
  POST /reliability - run direct Monte Carlo reliability analysis
  GET  /health       - health check
"""

import numpy as np
from scipy.stats import norm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Structural Reliability Service", version="3.1.0")

EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant, for Gumbel conversion


class RandomVariableSpec(BaseModel):
    dist: str       # "normal" | "lognormal" | "gumbel"
    mean: float
    cov: float


class ReliabilityRequest(BaseModel):
    N: int
    timeStep_years: float

    # ── Supplied by the Recipe Agent — the executable law, NOT hardcoded here ──
    capacity_formula: str
    demand_formula: str
    corrosion_formula: Optional[str] = None

    # ── Random variables — distribution model from Recipe Agent (JCSS/code) ──
    variables: Dict[str, RandomVariableSpec]

    # ── Fixed, deterministic values — auto-mapped from IFC geometry + Excel ──
    fixed_params: Dict[str, float]


class ReliabilityResponse(BaseModel):
    timeStep_years: float
    N: int
    n_fail: int
    Pf: float
    beta: float
    cov_Pf: Optional[float]
    mean_capacity_kNm: float
    mean_demand_kNm: float


# ── Restricted namespace for eval — ONLY numpy math, no builtins, no I/O ──
SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "max": np.maximum,
    "min": np.minimum,
}


def sample_variable(spec: RandomVariableSpec, N: int, rng) -> np.ndarray:
    if spec.dist == "normal":
        # scale IS the standard deviation directly — already correct.
        return rng.normal(spec.mean, spec.mean * spec.cov, size=N)

    if spec.dist == "lognormal":
        # Convert arithmetic (mean, CoV) -> lognormal's own (mu_ln, sigma_ln)
        # parameters, so the SAMPLED distribution actually has the intended
        # arithmetic mean and CoV — not a biased approximation.
        sigma_ln = np.sqrt(np.log(1 + spec.cov ** 2))
        mu_ln = np.log(spec.mean) - 0.5 * sigma_ln ** 2
        return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=N)

    if spec.dist == "gumbel":
        # NumPy's gumbel(loc, scale) has mean = loc + scale*EULER_GAMMA and
        # std = scale*pi/sqrt(6). "scale" is an ABSOLUTE dispersion (same
        # units as the variable), NOT the CoV. Convert properly from
        # arithmetic (mean, CoV) instead of passing CoV as scale directly.
        std = spec.mean * spec.cov
        scale = std * np.sqrt(6) / np.pi
        loc = spec.mean - scale * EULER_GAMMA
        return rng.gumbel(loc=loc, scale=scale, size=N)

    raise HTTPException(400, f"Unsupported distribution type: {spec.dist}")


def safe_eval_formula(formula: str, namespace: dict):
    try:
        return eval(formula, SAFE_GLOBALS, namespace)
    except Exception as e:
        raise HTTPException(400, f"Error evaluating formula '{formula}': {e}")


def run_reliability(req: ReliabilityRequest) -> ReliabilityResponse:
    rng = np.random.default_rng(seed=42)

    # 1. Sample every random variable according to the Recipe Agent's distribution model
    sampled = {name: sample_variable(spec, req.N, rng) for name, spec in req.variables.items()}

    # 2. Merge sampled arrays + fixed deterministic params + time step into ONE namespace
    namespace = {**sampled, **req.fixed_params, "t": req.timeStep_years}

    # 3. Apply corrosion degradation, if the recipe includes one
    if req.corrosion_formula:
        namespace["degradation_factor"] = safe_eval_formula(req.corrosion_formula, namespace)
    else:
        namespace["degradation_factor"] = 1.0

    # 4. Evaluate demand and capacity — BOTH formulas come from the Recipe Agent
    E = safe_eval_formula(req.demand_formula, namespace)
    R = safe_eval_formula(req.capacity_formula, namespace)

    g = R - E
    n_fail = int(np.sum(g < 0))
    Pf = min(max(n_fail / req.N, 0.5 / req.N), 1 - 0.5 / req.N)
    beta = float(-norm.ppf(Pf))
    cov_Pf = float(np.sqrt((1 - Pf) / (Pf * req.N))) if Pf > 0 else None

    return ReliabilityResponse(
        timeStep_years=req.timeStep_years, N=req.N, n_fail=n_fail, Pf=Pf, beta=beta,
        cov_Pf=cov_Pf, mean_capacity_kNm=float(np.mean(R)), mean_demand_kNm=float(np.mean(E)),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reliability", response_model=ReliabilityResponse)
def reliability_endpoint(req: ReliabilityRequest):
    return run_reliability(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
