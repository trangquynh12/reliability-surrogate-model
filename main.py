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

# 1. Rackwitz-Fiessler Equivalent Normal Transformation
# ============================================================
def rackwitz_fiessler_equivalent_normal(dist, x: float):
    """
    At the current point x, transform an arbitrary distribution
    into an equivalent NORMAL distribution having the same CDF
    and PDF at x.

    Returns:
        mu_N    : equivalent normal mean
        sigma_N : equivalent normal standard deviation

    This allows FORM to work in standard normal space even when
    the original random variable is Normal, Lognormal, or Gumbel.
    """

    # CDF at current physical-space point
    p = float(dist.cdf(x))

    # Avoid exactly 0 or 1 because norm.ppf(0) and norm.ppf(1)
    # would produce +/- infinity.
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))

    # Equivalent standard-normal coordinate
    z = float(norm.ppf(p))

    # PDF at current physical-space point
    pdf_x = float(max(dist.pdf(x), 1e-300))

    # Standard normal PDF at z
    phi_z = float(norm.pdf(z))

    # Rackwitz-Fiessler equivalent normal standard deviation
    sigma_N = phi_z / pdf_x

    # Protect against numerical problems
    sigma_N = float(max(sigma_N, 1e-12))

    # Equivalent normal mean
    mu_N = x - sigma_N * z

    return mu_N, sigma_N


# ============================================================
# 2. FORM
# ============================================================
def compute_form(
    variables: Dict[str, "RandomVariableSpec"],
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

    Limit-state convention used by this project:

        g(X) = R(X) - E(X)

        g > 0  -> SAFE
        g = 0  -> LIMIT STATE
        g < 0  -> FAILURE

    FORM searches for the Most Probable Point (MPP) of failure.

    Returns:
        beta_FORM
        Pf_FORM
        converged
    """

    # --------------------------------------------------------
    # 1. Prepare random variables and their distributions
    # --------------------------------------------------------

    names = list(variables.keys())

    if len(names) == 0:
        raise HTTPException(
            400,
            "FORM requires at least one random variable."
        )

    dists = {
        name: get_scipy_dist(variables[name])
        for name in names
    }

    n_var = len(names)

    # --------------------------------------------------------
    # 2. Initial point = mean values in physical space
    # --------------------------------------------------------

    x = np.array(
        [variables[name].mean for name in names],
        dtype=float
    )

    # --------------------------------------------------------
    # 3. Limit-state function
    # --------------------------------------------------------

    def g_func(x_vec: np.ndarray) -> float:

        ns = {
            name: float(x_vec[i])
            for i, name in enumerate(names)
        }

        # Add deterministic parameters
        ns.update(fixed_params)

        # Time
        ns["t"] = float(t)

        # Corrosion degradation
        if corrosion_formula:
            ns["degradation_factor"] = safe_eval_formula(
                corrosion_formula,
                ns
            )
        else:
            ns["degradation_factor"] = 1.0

        # Resistance
        R = safe_eval_formula(
            capacity_formula,
            ns
        )

        # Load effect
        E = safe_eval_formula(
            demand_formula,
            ns
        )

        # IMPORTANT:
        # Positive = safe
        # Negative = failure
        g = float(R - E)

        return g

    # --------------------------------------------------------
    # 4. Check initial point
    # --------------------------------------------------------

    g_initial = g_func(x)

    if not np.isfinite(g_initial):
        raise HTTPException(
            400,
            f"FORM initial limit-state value is not finite: {g_initial}"
        )

    # --------------------------------------------------------
    # 5. HL-RF iteration
    # --------------------------------------------------------

    u = np.zeros(n_var, dtype=float)

    beta_previous = np.inf

    converged = False

    last_g = g_initial

    for iteration in range(max_iter):

        # ====================================================
        # A. Equivalent normal transformation
        # ====================================================

        mu_N = np.zeros(n_var, dtype=float)
        sigma_N = np.zeros(n_var, dtype=float)

        for i, name in enumerate(names):

            mu_i, sigma_i = (
                rackwitz_fiessler_equivalent_normal(
                    dists[name],
                    float(x[i])
                )
            )

            mu_N[i] = mu_i
            sigma_N[i] = sigma_i

        # ====================================================
        # B. Transform physical x -> normal-space u
        # ====================================================

        u = (x - mu_N) / sigma_N

        if not np.all(np.isfinite(u)):
            break

        # ====================================================
        # C. Evaluate limit state
        # ====================================================

        g0 = g_func(x)

        if not np.isfinite(g0):
            break

        last_g = g0

        # ====================================================
        # D. Numerical gradient dg/dx
        # ====================================================

        grad_x = np.zeros(n_var, dtype=float)

        for i in range(n_var):

            # Relative perturbation
            h = max(
                abs(float(x[i])) * 1e-5,
                1e-7
            )

            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[i] += h
            x_minus[i] -= h

            g_plus = g_func(x_plus)
            g_minus = g_func(x_minus)

            if (
                not np.isfinite(g_plus)
                or not np.isfinite(g_minus)
            ):
                grad_x[i] = 0.0
                continue

            grad_x[i] = (
                g_plus - g_minus
            ) / (2.0 * h)

        # ====================================================
        # E. Transform gradient from x-space to u-space
        #
        # dx/du = sigma_N
        #
        # dg/du = dg/dx * dx/du
        # ====================================================

        grad_u = grad_x * sigma_N

        norm_grad = float(
            np.linalg.norm(grad_u)
        )

        if (
            not np.isfinite(norm_grad)
            or norm_grad < 1e-12
        ):
            # Gradient is effectively zero.
            # FORM cannot determine a direction.
            break

        # ====================================================
        # F. HL-RF update
        #
        # IMPORTANT:
        #
        # g = R - E
        #
        # Therefore:
        #   g > 0 -> safe
        #   g < 0 -> failure
        #
        # The update below moves toward g = 0.
        # ====================================================

        u_new = (
            (
                np.dot(grad_u, u) - g0
            )
            / (norm_grad ** 2)
        ) * grad_u

        if not np.all(np.isfinite(u_new)):
            break

        # ====================================================
        # G. Convert new u back to physical x
        # ====================================================

        x_new = (
            mu_N
            + sigma_N * u_new
        )

        if not np.all(np.isfinite(x_new)):
            break

        # ====================================================
        # H. Reliability index at new point
        # ====================================================

        beta_new = float(
            np.linalg.norm(u_new)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # beta is ALWAYS reported as a positive distance
        # from origin to the MPP.
        #
        # Do NOT use a signed beta here.
        # ----------------------------------------------------

        # ====================================================
        # I. Convergence checks
        # ====================================================

        delta_beta = (
            abs(beta_new - beta_previous)
            if np.isfinite(beta_previous)
            else np.inf
        )

        delta_u = float(
            np.linalg.norm(u_new - u)
        )

        g_new = g_func(x_new)

        if not np.isfinite(g_new):
            break

        # ====================================================
        # J. Check convergence
        # ====================================================

        if (
            delta_beta < tol_beta
            and delta_u < tol_u
            and abs(g_new) < tol_g
        ):
            x = x_new
            u = u_new
            last_g = g_new
            beta_previous = beta_new
            converged = True
            break

        # ====================================================
        # K. Prepare next iteration
        # ====================================================

        x = x_new
        u = u_new
        beta_previous = beta_new

    # ========================================================
    # 6. Final FORM result
    # ========================================================

    beta_form = float(
        np.linalg.norm(u)
    )

    # If the algorithm never got a meaningful point,
    # don't return nonsense.
    if not np.isfinite(beta_form):
        raise HTTPException(
            400,
            "FORM failed: non-finite reliability index."
        )

    # ========================================================
    # 7. FORM failure probability
    # ========================================================

    # Standard FORM relationship:
    #
    #       Pf = Phi(-beta)
    #
    # Since beta is a positive distance from origin,
    # Pf is always <= 0.5.
    #
    Pf_FORM = float(
        norm.cdf(-beta_form)
    )

    return (
        beta_form,
        Pf_FORM,
        converged
    )


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
