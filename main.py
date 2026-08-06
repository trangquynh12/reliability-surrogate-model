import numpy as np
from scipy.stats import norm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Structural Reliability Service", version="3.0.0")


class RandomVariableSpec(BaseModel):
    dist: str       # "normal" | "lognormal" | "gumbel"
    mean: float
    cov: float


class ReliabilityRequest(BaseModel):
    N: int
    timeStep_years: float

    # ── Supplied by the Recipe Agent — the executable law, NOT hardcoded here ──
    capacity_formula: str          # e.g. "A_s * (f_yk/1.15) * (d_mm - (A_s*(f_yk/1.15))/(2*0.85*f_ck*b_mm)) / 1e6"
    demand_formula: str            # e.g. "B_D * demand_dead_mean_kNm + B_L * demand_live_mean_kNm"
    corrosion_formula: Optional[str] = None   # e.g. "max(0.05, 1 - k_A*max(0, t-T_i))" — multiplies a named variable

    # ── Random variables — distribution model from Recipe Agent (JCSS/code) ──
    variables: Dict[str, RandomVariableSpec]

    # ── Fixed, deterministic values — auto-mapped from IFC geometry + Excel ──
    # (per-member values: b_mm, d_mm, f_ck, demand_dead_mean_kNm, ... — whatever
    # the formula strings above reference by name)
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
    "max": np.maximum,   # vectorized max, so formulas can write max(0, x) on arrays
    "min": np.minimum,
}


def sample_variable(spec: RandomVariableSpec, N: int, rng) -> np.ndarray:
    if spec.dist == "normal":
        return rng.normal(spec.mean, spec.mean * spec.cov, size=N)
    if spec.dist == "lognormal":
        return rng.lognormal(mean=np.log(spec.mean), sigma=spec.cov, size=N)
    if spec.dist == "gumbel":
        return rng.gumbel(loc=spec.mean, scale=spec.cov, size=N)
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

    # 3. Apply corrosion degradation, if the recipe includes one, to whichever
    #    variable it targets (the formula itself decides — e.g. multiplies A_s)
    if req.corrosion_formula:
        namespace["degradation_factor"] = safe_eval_formula(req.corrosion_formula, namespace)
    else:
        namespace["degradation_factor"] = 1.0

    # 4. Evaluate demand and capacity — BOTH formulas come from the Recipe Agent,
    #    not from any code written in this service.
    E = safe_eval_formula(req.demand_formula, namespace)
    R = safe_eval_formula(req.capacity_formula, namespace)

    g = R - E
    n_fail = int(np.sum(g < 0))
    Pf = max(n_fail / req.N, 0.5 / req.N)
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
