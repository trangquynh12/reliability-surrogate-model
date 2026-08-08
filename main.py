"""
Structural Reliability Service
================================
FastAPI microservice that computes structural reliability (Pf, beta) via:
  1. Direct Monte Carlo simulation (crude MC) — fast, but cannot resolve
     very small Pf without enormous N (rare-event problem).
  2. FORM (First-Order Reliability Method, HL-RF algorithm) — finds the
     design point analytically, giving a reliable beta even when MC
     observes zero failures.

Both use capacity/demand/corrosion formulas supplied by the Recipe Agent
(never hardcoded here).

Endpoints:
  POST /reliability - run Monte Carlo + FORM reliability analysis
  GET  /health       - health check
"""

import numpy as np
from scipy.stats import norm, lognorm, gumbel_r
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Structural Reliability Service", version="4.0.0")

EULER_GAMMA = 0.5772156649015329


class RandomVariableSpec(BaseModel):
    dist: str       # "normal" | "lognormal" | "gumbel"
    mean: float
    cov: float


class ReliabilityRequest(BaseModel):
    N: int
    timeStep_years: float
    capacity_formula: str
    demand_formula: str
    corrosion_formula: Optional[str] = None
    variables: Dict[str, RandomVariableSpec]
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

    # STEP 1 — honesty flag: True means n_fail=0, so Pf/beta above are only
    # an artificial floor (0.5/N), NOT a resolved estimate.
    is_floor_value: bool

    # STEP 2 — FORM cross-check: a real point estimate, independent of N,
    # valid even when crude Monte Carlo observes zero failures.
    beta_FORM: Optional[float] = None
    Pf_FORM: Optional[float] = None
    form_converged: bool = False


SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "max": np.maximum,
    "min": np.minimum,
}


# --------------------------------------------------------------------------
# Distribution helpers — shared by Monte Carlo sampling AND FORM
# --------------------------------------------------------------------------

def sample_variable(spec: RandomVariableSpec, N: int, rng) -> np.ndarray:
    if spec.dist == "normal":
        return rng.normal(spec.mean, spec.mean * spec.cov, size=N)
    if spec.dist == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + spec.cov ** 2))
        mu_ln = np.log(spec.mean) - 0.5 * sigma_ln ** 2
        return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=N)
    if spec.dist == "gumbel":
        std = spec.mean * spec.cov
        scale = std * np.sqrt(6) / np.pi
        loc = spec.mean - scale * EULER_GAMMA
        return rng.gumbel(loc=loc, scale=scale, size=N)
    raise HTTPException(400, f"Unsupported distribution type: {spec.dist}")


def get_scipy_dist(spec: RandomVariableSpec):
    """Returns a frozen scipy.stats distribution with the SAME parameterization
    used for Monte Carlo sampling above, so FORM and MC agree on what each
    distribution actually means."""
    if spec.dist == "normal":
        return norm(loc=spec.mean, scale=spec.mean * spec.cov)
    if spec.dist == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + spec.cov ** 2))
        mu_ln = np.log(spec.mean) - 0.5 * sigma_ln ** 2
        return lognorm(s=sigma_ln, scale=np.exp(mu_ln))
    if spec.dist == "gumbel":
        std = spec.mean * spec.cov
        scale = std * np.sqrt(6) / np.pi
        loc = spec.mean - scale * EULER_GAMMA
        return gumbel_r(loc=loc, scale=scale)
    raise HTTPException(400, f"Unsupported distribution type: {spec.dist}")


def safe_eval_formula(formula: str, namespace: dict):
    try:
        return eval(formula, SAFE_GLOBALS, namespace)
    except Exception as e:
        raise HTTPException(400, f"Error evaluating formula '{formula}': {e}")


# --------------------------------------------------------------------------
# STEP 2 — FORM (HL-RF algorithm)
# --------------------------------------------------------------------------

def rackwitz_fiessler_equivalent_normal(dist, x: float):
    """At point x, returns (mu_N, sigma_N): the mean/std of a NORMAL
    distribution that matches dist's CDF and PDF at exactly this point.
    This lets FORM handle any distribution shape (Normal/Lognormal/Gumbel)
    using the same normal-space algorithm."""
    p = np.clip(dist.cdf(x), 1e-15, 1 - 1e-15)
    z = norm.ppf(p)
    pdf_x = max(dist.pdf(x), 1e-300)
    sigma_N = norm.pdf(z) / pdf_x
    mu_N = x - sigma_N * z
    return mu_N, sigma_N


def compute_form(variables: Dict[str, RandomVariableSpec], fixed_params: dict,
                  capacity_formula: str, demand_formula: str,
                  corrosion_formula: Optional[str], t: float,
                  max_iter: int = 50, tol: float = 1e-4):
    """Hasofer-Lind Rackwitz-Fiessler (HL-RF) algorithm — finds the design
    point (most probable failure point) and returns beta_FORM. Formulas are
    treated as black boxes (evaluated via eval()), so gradients are computed
    numerically (finite differences) rather than symbolically."""

    names = list(variables.keys())
    dists = {name: get_scipy_dist(variables[name]) for name in names}
    x = np.array([variables[name].mean for name in names], dtype=float)

    def g_func(x_vec: np.ndarray) -> float:
        ns = {name: x_vec[i] for i, name in enumerate(names)}
        ns.update(fixed_params)
        ns["t"] = t
        ns["degradation_factor"] = (
            safe_eval_formula(corrosion_formula, ns) if corrosion_formula else 1.0
        )
        R = safe_eval_formula(capacity_formula, ns)
        E = safe_eval_formula(demand_formula, ns)
        return float(R - E)

    beta_prev = 0.0
    converged = False

    for _ in range(max_iter):
        mu_N = np.zeros(len(names))
        sigma_N = np.zeros(len(names))
        for i, name in enumerate(names):
            mu_N[i], sigma_N[i] = rackwitz_fiessler_equivalent_normal(dists[name], x[i])

        u = (x - mu_N) / sigma_N
        g0 = g_func(x)

        grad_x = np.zeros(len(names))
        for i in range(len(names)):
            h = max(abs(x[i]) * 1e-4, 1e-6)
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad_x[i] = (g_func(x_plus) - g_func(x_minus)) / (2 * h)

        grad_u = grad_x * sigma_N  # chain rule: dg/du = dg/dx * dx/du
        norm_grad = np.linalg.norm(grad_u)
        if norm_grad < 1e-12:
            break  # flat gradient — cannot proceed further

        alpha = -grad_u / norm_grad
        beta_new = (np.dot(grad_u, u) - g0) / norm_grad
        u_new = beta_new * alpha
        x_new = mu_N + sigma_N * u_new

        if abs(beta_new - beta_prev) < tol:
            beta_prev = beta_new
            converged = True
            break

        x = x_new
        beta_prev = beta_new

    Pf_FORM = float(norm.cdf(-beta_prev))
    return float(beta_prev), Pf_FORM, converged


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------

def run_reliability(req: ReliabilityRequest) -> ReliabilityResponse:
    rng = np.random.default_rng(seed=42)

    sampled = {name: sample_variable(spec, req.N, rng) for name, spec in req.variables.items()}
    namespace = {**sampled, **req.fixed_params, "t": req.timeStep_years}

    if req.corrosion_formula:
        namespace["degradation_factor"] = safe_eval_formula(req.corrosion_formula, namespace)
    else:
        namespace["degradation_factor"] = 1.0

    E = safe_eval_formula(req.demand_formula, namespace)
    R = safe_eval_formula(req.capacity_formula, namespace)

    g = R - E
    n_fail = int(np.sum(g < 0))

    # STEP 1 — honesty flag: n_fail=0 means Pf/beta below are an artificial
    # floor (0.5/N), not a resolved estimate.
    is_floor_value = (n_fail == 0)

    Pf = min(max(n_fail / req.N, 0.5 / req.N), 1 - 0.5 / req.N)
    beta = float(-norm.ppf(Pf))
    cov_Pf = float(np.sqrt((1 - Pf) / (Pf * req.N))) if Pf > 0 else None

    # STEP 2 — always compute FORM as a cross-check (cheap: only a few
    # formula evaluations, independent of N). Most valuable exactly when
    # is_floor_value=True, since it gives a real estimate MC cannot resolve.
    try:
        beta_FORM, Pf_FORM, form_converged = compute_form(
            req.variables, req.fixed_params, req.capacity_formula,
            req.demand_formula, req.corrosion_formula, req.timeStep_years
        )
    except Exception:
        beta_FORM, Pf_FORM, form_converged = None, None, False

    return ReliabilityResponse(
        timeStep_years=req.timeStep_years, N=req.N, n_fail=n_fail, Pf=Pf, beta=beta,
        cov_Pf=cov_Pf, mean_capacity_kNm=float(np.mean(R)), mean_demand_kNm=float(np.mean(E)),
        is_floor_value=is_floor_value,
        beta_FORM=beta_FORM, Pf_FORM=Pf_FORM, form_converged=form_converged,
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
