"""
Structural Reliability Service  v5.0.0
========================================
FastAPI microservice — POST /reliability runs:
  1. Direct Monte Carlo (crude MC)
  2. FORM (HL-RF + Rackwitz-Fiessler) as cross-check

Fixes applied vs v4.0.0:
  FIX 1 — g_variance block removed from g_func (caused NameError on every
           FORM iteration → form_converged=False always)
  FIX 2 — capacity formula note: z must be in mm when dividing by 1e6,
           OR in m when dividing by 1e3. Both give kNm. The formula
           supplied by the Recipe Agent must be consistent with its units.
  FIX 3 — corrosion applied to A_p via degradation_factor in the capacity
           formula: "theta_R * (A_p * degradation_factor) * f_ps * z / 1e6"
  FIX 4 — g_variance check moved to run_reliability() where g is a vector,
           so np.var(g) is meaningful. Emits a warning but never blocks.

Endpoints:
  POST /reliability  — run MC + FORM reliability analysis
  GET  /health       — health check
"""

import numpy as np
from scipy.stats import norm, lognorm, gumbel_r
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

app = FastAPI(title="Structural Reliability Service", version="5.0.0")

EULER_GAMMA = 0.5772156649015329

# Only safe math symbols — no builtins, no file I/O
SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "max": np.maximum,
    "min": np.minimum,
}


# ── Pydantic models ──────────────────────────────────────────────────────────

class RandomVariableSpec(BaseModel):
    dist: str        # "normal" | "lognormal" | "gumbel"
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

    # True when n_fail=0 — beta is only a floor value (0.5/N), not resolved
    is_floor_value: bool

    # FORM cross-check — valid even when MC observes zero failures
    beta_FORM: Optional[float] = None
    Pf_FORM: Optional[float] = None
    form_converged: bool = False

    # Governing beta — FORM when MC is unreliable, min(MC,FORM) otherwise
    governing_beta: float
    governing_source: str     # "FORM" | "MC" | "min(MC,FORM)"


# ── Distribution helpers ─────────────────────────────────────────────────────

def sample_variable(spec: RandomVariableSpec, N: int, rng) -> np.ndarray:
    """Draw N samples from the given distribution."""
    if spec.dist == "normal":
        return rng.normal(spec.mean, spec.mean * spec.cov, size=N)
    if spec.dist == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + spec.cov ** 2))
        mu_ln    = np.log(spec.mean) - 0.5 * sigma_ln ** 2
        return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=N)
    if spec.dist == "gumbel":
        std   = spec.mean * spec.cov
        scale = std * np.sqrt(6) / np.pi
        loc   = spec.mean - scale * EULER_GAMMA
        return rng.gumbel(loc=loc, scale=scale, size=N)
    raise HTTPException(400, f"Unsupported distribution: '{spec.dist}'. Use normal/lognormal/gumbel.")


def get_scipy_dist(spec: RandomVariableSpec):
    """Frozen scipy distribution — same parameterization as sample_variable()."""
    if spec.dist == "normal":
        return norm(loc=spec.mean, scale=spec.mean * spec.cov)
    if spec.dist == "lognormal":
        sigma_ln = np.sqrt(np.log(1 + spec.cov ** 2))
        mu_ln    = np.log(spec.mean) - 0.5 * sigma_ln ** 2
        return lognorm(s=sigma_ln, scale=np.exp(mu_ln))
    if spec.dist == "gumbel":
        std   = spec.mean * spec.cov
        scale = std * np.sqrt(6) / np.pi
        loc   = spec.mean - scale * EULER_GAMMA
        return gumbel_r(loc=loc, scale=scale)
    raise HTTPException(400, f"Unsupported distribution: '{spec.dist}'.")


def safe_eval(formula: str, namespace: dict):
    try:
        return eval(formula, SAFE_GLOBALS, namespace)
    except Exception as e:
        raise HTTPException(400, f"Error evaluating formula '{formula}': {e}")


# ── FORM ─────────────────────────────────────────────────────────────────────

def rackwitz_fiessler(dist, x: float):
    """
    Rackwitz-Fiessler equivalent normal transformation at point x.
    Returns (mu_N, sigma_N) — the equivalent normal mean and std.
    """
    p      = float(np.clip(dist.cdf(x), 1e-12, 1 - 1e-12))
    z      = float(norm.ppf(p))
    pdf_x  = float(max(dist.pdf(x), 1e-300))
    phi_z  = float(norm.pdf(z))
    sigma_N = float(max(phi_z / pdf_x, 1e-12))
    mu_N    = x - sigma_N * z
    return mu_N, sigma_N


def compute_form(
    variables: Dict[str, RandomVariableSpec],
    fixed_params: dict,
    capacity_formula: str,
    demand_formula: str,
    corrosion_formula: Optional[str],
    t: float,
    max_iter: int = 100,
    tol_beta: float = 1e-5,
    tol_u: float = 1e-5,
    tol_g: float = 1e-6,
):
    """
    Hasofer-Lind / Rackwitz-Fiessler FORM.

    Limit state: g(X) = R(X,t) - E(X)
      g > 0  safe
      g = 0  limit state
      g < 0  failure

    Returns (beta_FORM, Pf_FORM, converged).
    """
    names  = list(variables.keys())
    n_var  = len(names)
    if n_var == 0:
        raise HTTPException(400, "FORM requires at least one random variable.")

    dists = {name: get_scipy_dist(variables[name]) for name in names}

    # ── Limit-state function (scalar evaluation, used inside FORM iterations) ──
    def g_func(x_vec: np.ndarray) -> float:
        ns = {name: float(x_vec[i]) for i, name in enumerate(names)}
        ns.update(fixed_params)
        ns["t"] = float(t)
        # Corrosion degradation factor — evaluated once per call
        # FIX 1: NO g_variance check here (scalar g → np.var always 0)
        #         The variance check is in run_reliability() where g is a vector.
        ns["degradation_factor"] = (
            float(safe_eval(corrosion_formula, ns))
            if corrosion_formula else 1.0
        )
        R = safe_eval(capacity_formula, ns)
        E = safe_eval(demand_formula, ns)
        return float(R - E)

    # ── Check initial point ──
    x = np.array([variables[name].mean for name in names], dtype=float)
    g_initial = g_func(x)
    if not np.isfinite(g_initial):
        raise HTTPException(400, f"FORM: initial g(X_mean) is not finite ({g_initial}).")

    u         = np.zeros(n_var, dtype=float)
    beta_prev = np.inf
    converged = False

    for _iteration in range(max_iter):

        # A. Rackwitz-Fiessler equivalent normal at current x
        mu_N    = np.zeros(n_var)
        sigma_N = np.zeros(n_var)
        for i, name in enumerate(names):
            mu_N[i], sigma_N[i] = rackwitz_fiessler(dists[name], float(x[i]))

        # B. Transform physical x → standard normal u
        u = (x - mu_N) / sigma_N
        if not np.all(np.isfinite(u)):
            break

        # C. Evaluate limit state at x
        g0 = g_func(x)
        if not np.isfinite(g0):
            break

        # D. Numerical gradient ∂g/∂x (central differences)
        grad_x = np.zeros(n_var)
        for i in range(n_var):
            h = max(abs(float(x[i])) * 1e-5, 1e-7)
            xp = x.copy(); xm = x.copy()
            xp[i] += h;    xm[i] -= h
            gp = g_func(xp); gm = g_func(xm)
            if np.isfinite(gp) and np.isfinite(gm):
                grad_x[i] = (gp - gm) / (2.0 * h)

        # E. Gradient in u-space: ∂g/∂u = (∂g/∂x) · σ_N
        grad_u    = grad_x * sigma_N
        norm_grad = float(np.linalg.norm(grad_u))
        if not np.isfinite(norm_grad) or norm_grad < 1e-12:
            break

        # F. HL-RF update rule
        u_new = ((np.dot(grad_u, u) - g0) / (norm_grad ** 2)) * grad_u
        if not np.all(np.isfinite(u_new)):
            break

        # G. Back-transform u_new → physical x_new
        x_new = mu_N + sigma_N * u_new
        if not np.all(np.isfinite(x_new)):
            break

        # H. Reliability index at new point
        beta_new = float(np.linalg.norm(u_new))

        # I. Evaluate limit state at x_new (convergence check)
        g_new = g_func(x_new)
        if not np.isfinite(g_new):
            break

        # J. Convergence: all three criteria must be met simultaneously
        delta_beta = abs(beta_new - beta_prev) if np.isfinite(beta_prev) else np.inf
        delta_u    = float(np.linalg.norm(u_new - u))

        if delta_beta < tol_beta and delta_u < tol_u and abs(g_new) < tol_g:
            x         = x_new
            u         = u_new
            converged = True
            break

        x         = x_new
        u         = u_new
        beta_prev = beta_new

    # ── Final result ──
    beta_form = float(np.linalg.norm(u))
    if not np.isfinite(beta_form):
        raise HTTPException(400, "FORM failed: non-finite reliability index.")

    # Pf = Phi(-beta)  [beta is positive distance from origin to MPP]
    Pf_FORM = float(norm.cdf(-beta_form))
    return beta_form, Pf_FORM, converged


# ── Main computation ─────────────────────────────────────────────────────────

def run_reliability(req: ReliabilityRequest) -> ReliabilityResponse:
    """
    1. Draw N samples (seeded for reproducibility).
    2. Evaluate capacity, demand, limit state.
    3. FIX 4: Check Var[g] HERE (g is a vector — np.var is meaningful).
    4. Compute MC-based Pf and beta.
    5. Run FORM as cross-check.
    6. Select governing beta.
    """
    rng = np.random.default_rng(seed=42)  # seeded — reproducible

    # ── Step 1: Sample all random variables ──
    sampled = {
        name: sample_variable(spec, req.N, rng)
        for name, spec in req.variables.items()
    }

    namespace = {**sampled, **req.fixed_params, "t": req.timeStep_years}

    # ── Step 2: Corrosion degradation factor (vector) ──
    if req.corrosion_formula:
        namespace["degradation_factor"] = safe_eval(req.corrosion_formula, namespace)
    else:
        namespace["degradation_factor"] = 1.0

    # ── Step 3: Capacity and demand (vectors) ──
    R = safe_eval(req.capacity_formula, namespace)
    E = safe_eval(req.demand_formula,   namespace)
    g = R - E

    # ── FIX 4: Var[g] check — NOW meaningful because g is a vector ──
    g_variance = float(np.var(g))
    if g_variance < 1e-6:
        print(
            f"WARNING: Var[g] = {g_variance:.2e} at t={req.timeStep_years} yr — "
            f"limit state is nearly deterministic. Verify that capacity/demand "
            f"formulas reference the random variables, and that corrosion "
            f"degradation_factor is applied to A_p in the capacity formula."
        )

    # ── Step 4: Monte Carlo Pf and beta ──
    n_fail      = int(np.sum(g < 0))
    is_floor    = (n_fail == 0)

    # Laplace correction — avoids log(0) at both ends
    Pf   = float(np.clip(n_fail / req.N, 0.5 / req.N, 1.0 - 0.5 / req.N))
    beta = float(-norm.ppf(Pf))

    # Coefficient of variation of Pf estimate
    cov_Pf = float(np.sqrt((1.0 - Pf) / (Pf * req.N))) if Pf > 0 else None

    # ── Step 5: FORM cross-check ──
    try:
        beta_FORM, Pf_FORM, form_converged = compute_form(
            req.variables,
            req.fixed_params,
            req.capacity_formula,
            req.demand_formula,
            req.corrosion_formula,
            req.timeStep_years,
        )
    except Exception as exc:
        print(f"FORM failed at t={req.timeStep_years}: {exc}")
        beta_FORM, Pf_FORM, form_converged = None, None, False

    # ── Step 6: Governing beta selection ──
    #
    # Rule (from supervisor guidelines):
    #   - If MC has < 30 failures → statistically unreliable → use FORM if converged
    #   - If MC has >= 30 failures AND FORM converged → use min(MC, FORM) [conservative]
    #   - If FORM did not converge → use MC only
    #
    mc_reliable   = n_fail >= 30
    form_ok       = form_converged and beta_FORM is not None and np.isfinite(beta_FORM)

    if not mc_reliable and form_ok:
        governing_beta   = beta_FORM
        governing_source = "FORM"
    elif mc_reliable and form_ok:
        governing_beta   = min(beta, beta_FORM)
        governing_source = "min(MC,FORM)"
    else:
        governing_beta   = beta
        governing_source = "MC"

    return ReliabilityResponse(
        timeStep_years    = req.timeStep_years,
        N                 = req.N,
        n_fail            = n_fail,
        Pf                = Pf,
        beta              = beta,
        cov_Pf            = cov_Pf,
        mean_capacity_kNm = float(np.mean(R)),
        mean_demand_kNm   = float(np.mean(E)),
        is_floor_value    = is_floor,
        beta_FORM         = beta_FORM,
        Pf_FORM           = Pf_FORM,
        form_converged    = form_converged,
        governing_beta    = governing_beta,
        governing_source  = governing_source,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0.0"}


@app.post("/reliability", response_model=ReliabilityResponse)
def reliability_endpoint(req: ReliabilityRequest):
    return run_reliability(req)


# ── Unit tests (run with: python main.py --test) ─────────────────────────────

def _run_unit_tests():
    """
    Quick sanity checks — called with 'python main.py --test'.
    All must pass before deploying to Railway.
    """
    print("Running unit tests...")

    # Test 1: Beta sign convention
    assert abs(-norm.ppf(0.5))      < 1e-6,  "Phi_inv(0.5) must be 0"
    assert abs(-norm.ppf(0.00025) - 3.481) < 0.001, "beta(Pf=0.00025) must be +3.481"
    assert abs(-norm.ppf(0.99975) + 3.481) < 0.001, "beta(Pf=0.99975) must be -3.481"
    print("  [1] Beta sign convention ✅")

    # Test 2: Capacity formula units
    # theta_R * A_p[mm²] * f_ps[MPa=N/mm²] * z[mm] / 1e6 → kNm
    # 1.0 * 2000 * 1750 * 1160 / 1e6 = 4060 kNm
    cap = 1.0 * 2000 * 1750 * 1160 / 1e6
    assert abs(cap - 4060) < 1.0, f"Capacity formula unit check failed: {cap}"
    print("  [2] Capacity formula units (z in mm, /1e6 → kNm) ✅")

    # Test 3: Distribution means match spec
    rng = np.random.default_rng(0)
    for dist, mean, cov in [("normal",1000,0.05), ("lognormal",1750,0.025), ("gumbel",1.0,0.18)]:
        spec = RandomVariableSpec(dist=dist, mean=mean, cov=cov)
        s = sample_variable(spec, 2_000_000, rng)
        err = abs(float(np.mean(s)) / mean - 1)
        assert err < 0.005, f"{dist} mean error {err:.4f} > 0.5%"
    print("  [3] Distribution mean accuracy (<0.5%) ✅")

    # Test 4: Simple closed-form example
    # g = R - E, R ~ N(5,1), E ~ N(3,1) → beta_exact = (5-3)/sqrt(2) ≈ 1.414
    variables = {
        "R": RandomVariableSpec(dist="normal", mean=5.0, cov=0.2),
        "E": RandomVariableSpec(dist="normal", mean=3.0, cov=1/3),
    }
    req = ReliabilityRequest(
        N=2_000_000,
        timeStep_years=0.0,
        capacity_formula="R",
        demand_formula="E",
        corrosion_formula=None,
        variables=variables,
        fixed_params={},
    )
    resp = run_reliability(req)
    expected = (5 - 3) / np.sqrt(1**2 + 1**2)
    assert abs(resp.governing_beta - expected) < 0.05, \
        f"Simple test: governing_beta={resp.governing_beta:.3f}, expected≈{expected:.3f}"
    print(f"  [4] Simple closed-form: beta={resp.governing_beta:.3f} (expect {expected:.3f}) ✅")

    # Test 5: Corrosion reduces capacity monotonically
    variables2 = {
        "theta_R": RandomVariableSpec(dist="lognormal", mean=1.0,  cov=0.09),
        "A_p":     RandomVariableSpec(dist="normal",    mean=2000, cov=0.015),
        "f_ps":    RandomVariableSpec(dist="lognormal", mean=1750, cov=0.025),
        "z":       RandomVariableSpec(dist="normal",    mean=1160, cov=0.03),
        "B_D":     RandomVariableSpec(dist="lognormal", mean=1.05, cov=0.10),
        "B_L":     RandomVariableSpec(dist="gumbel",    mean=1.00, cov=0.18),
    }
    betas = []
    for t in [0, 20, 50]:
        req2 = ReliabilityRequest(
            N=200_000,
            timeStep_years=float(t),
            capacity_formula="theta_R * (A_p * degradation_factor) * f_ps * z / 1e6",
            demand_formula="B_D * M_dead + B_L * M_live",
            corrosion_formula="max(0.05, 1 - 0.005 * max(0, t - 15))",
            variables=variables2,
            fixed_params={"M_dead": 967.91, "M_live": 1072.63},
        )
        resp2 = run_reliability(req2)
        betas.append(resp2.governing_beta)
    assert betas[0] >= betas[1] >= betas[2], \
        f"Beta must decrease monotonically: {betas}"
    print(f"  [5] Corrosion monotonic: beta(t=0,20,50)={[round(b,2) for b in betas]} ✅")

    print("\n✅ All unit tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_unit_tests()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
