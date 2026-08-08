import numpy as np
from scipy.stats import norm, lognorm, gumbel_r
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Structural Reliability Service",
    version="4.0.0"
)


# ============================================================
# INPUT MODELS
# ============================================================

class RandomVariableSpec(BaseModel):
    dist: str
    mean: float
    cov: float


class ReliabilityRequest(BaseModel):
    N: int
    timeStep_years: float

    # Supplied by Recipe Agent
    capacity_formula: str
    demand_formula: str
    corrosion_formula: Optional[str] = None

    # Random variables
    variables: Dict[str, RandomVariableSpec]

    # Deterministic parameters
    fixed_params: Dict[str, float]


# ============================================================
# OUTPUT MODEL
# ============================================================

class ReliabilityResponse(BaseModel):
    timeStep_years: float
    N: int

    n_fail: int

    Pf: float
    beta: float

    cov_Pf: Optional[float]

    mean_capacity_kNm: float
    mean_demand_kNm: float

    # FORM
    beta_FORM: Optional[float]
    Pf_FORM: Optional[float]
    FORM_converged: bool

    FORM_g_residual: Optional[float]
    FORM_u_norm: Optional[float]
    FORM_iterations: int


# ============================================================
# SAFE EVAL
# ============================================================

SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "max": np.maximum,
    "min": np.minimum,
    "abs": np.abs,
}


def safe_eval_formula(
    formula: str,
    namespace: dict
):
    try:
        result = eval(
            formula,
            SAFE_GLOBALS,
            namespace
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Error evaluating formula "
                f"'{formula}': {e}"
            )
        )


# ============================================================
# DISTRIBUTION CONVERSION
# ============================================================

EULER_GAMMA = 0.5772156649015329


def get_scipy_dist(
    spec: RandomVariableSpec
):
    """
    Convert:

        arithmetic mean
        COV

    into a scipy probability distribution.
    """

    dist_name = spec.dist.lower()

    mean = float(spec.mean)
    cov = float(spec.cov)

    if cov < 0:

        raise HTTPException(
            status_code=400,
            detail=f"COV cannot be negative: {cov}"
        )

    # ========================================================
    # NORMAL
    # ========================================================

    if dist_name == "normal":

        sigma = abs(mean) * cov

        return norm(
            loc=mean,
            scale=max(sigma, 1e-12)
        )

    # ========================================================
    # LOGNORMAL
    #
    # Arithmetic mean:
    #
    #     m
    #
    # COV:
    #
    #     v
    #
    # Underlying normal:
    #
    #     sigma_ln = sqrt(log(1 + v^2))
    #
    #     mu_ln = log(m) - 0.5*sigma_ln^2
    # ========================================================

    if dist_name == "lognormal":

        if mean <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Lognormal variable requires "
                    f"mean > 0. Got {mean}"
                )
            )

        sigma_ln = np.sqrt(
            np.log(
                1.0 + cov ** 2
            )
        )

        mu_ln = (
            np.log(mean)
            - 0.5 * sigma_ln ** 2
        )

        return lognorm(
            s=sigma_ln,
            scale=np.exp(mu_ln)
        )

    # ========================================================
    # GUMBEL
    #
    # For scipy gumbel_r:
    #
    # mean = loc + gamma*scale
    #
    # std = pi/sqrt(6)*scale
    # ========================================================

    if dist_name == "gumbel":

        if mean <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Gumbel variable requires "
                    f"positive mean. Got {mean}"
                )
            )

        std = mean * cov

        scale = (
            std
            * np.sqrt(6.0)
            / np.pi
        )

        loc = (
            mean
            - EULER_GAMMA * scale
        )

        return gumbel_r(
            loc=loc,
            scale=max(scale, 1e-12)
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported distribution type: "
            f"{spec.dist}"
        )
    )


# ============================================================
# MONTE CARLO SAMPLING
# ============================================================

def sample_variable(
    spec: RandomVariableSpec,
    N: int,
    rng
) -> np.ndarray:

    dist_name = spec.dist.lower()

    mean = float(spec.mean)
    cov = float(spec.cov)

    # ========================================================
    # NORMAL
    # ========================================================

    if dist_name == "normal":

        sigma = abs(mean) * cov

        return rng.normal(
            loc=mean,
            scale=max(sigma, 1e-12),
            size=N
        )

    # ========================================================
    # LOGNORMAL
    #
    # IMPORTANT:
    #
    # numpy.lognormal() takes the parameters of the
    # underlying NORMAL distribution.
    #
    # It does NOT take arithmetic mean and COV.
    # ========================================================

    if dist_name == "lognormal":

        if mean <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Lognormal variable requires "
                    f"mean > 0. Got {mean}"
                )
            )

        sigma_ln = np.sqrt(
            np.log(
                1.0 + cov ** 2
            )
        )

        mu_ln = (
            np.log(mean)
            - 0.5 * sigma_ln ** 2
        )

        return rng.lognormal(
            mean=mu_ln,
            sigma=sigma_ln,
            size=N
        )

    # ========================================================
    # GUMBEL
    # ========================================================

    if dist_name == "gumbel":

        if mean <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Gumbel variable requires "
                    f"positive mean. Got {mean}"
                )
            )

        std = mean * cov

        scale = (
            std
            * np.sqrt(6.0)
            / np.pi
        )

        loc = (
            mean
            - EULER_GAMMA * scale
        )

        return rng.gumbel(
            loc=loc,
            scale=max(scale, 1e-12),
            size=N
        )

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported distribution type: "
            f"{spec.dist}"
        )
    )


# ============================================================
# LIMIT STATE FUNCTION
# ============================================================

def make_limit_state_function(
    variables,
    fixed_params,
    capacity_formula,
    demand_formula,
    corrosion_formula,
    t
):
    """
    Limit-state function:

        g(X) = R(X) - E(X)

    Failure:

        g(X) < 0
    """

    names = list(
        variables.keys()
    )

    def g_func(
        x_vec: np.ndarray
    ) -> float:

        namespace = {}

        for i, name in enumerate(names):

            namespace[name] = float(
                x_vec[i]
            )

        # Add deterministic values
        namespace.update(
            fixed_params
        )

        # Time
        namespace["t"] = float(t)

        # ====================================================
        # CORROSION
        # ====================================================

        if corrosion_formula:

            degradation_factor = (
                safe_eval_formula(
                    corrosion_formula,
                    namespace
                )
            )

            namespace[
                "degradation_factor"
            ] = degradation_factor

        else:

            namespace[
                "degradation_factor"
            ] = 1.0

        # ====================================================
        # CAPACITY
        # ====================================================

        R = safe_eval_formula(
            capacity_formula,
            namespace
        )

        # ====================================================
        # DEMAND
        # ====================================================

        E = safe_eval_formula(
            demand_formula,
            namespace
        )

        # ====================================================
        # LIMIT STATE
        # ====================================================

        return float(
            R - E
        )

    return g_func


# ============================================================
# RACKWITZ-FIESSLER TRANSFORMATION
# ============================================================

def rackwitz_fiessler_equivalent_normal(
    dist,
    x
):
    """
    Determine equivalent normal distribution at x.

    Match:

        CDF
        PDF
    """

    # --------------------------------------------------------
    # CDF
    # --------------------------------------------------------

    p = float(
        np.clip(
            dist.cdf(x),
            1e-12,
            1.0 - 1e-12
        )
    )

    # --------------------------------------------------------
    # Equivalent normal z
    # --------------------------------------------------------

    z = float(
        norm.ppf(p)
    )

    # --------------------------------------------------------
    # Original PDF
    # --------------------------------------------------------

    pdf_x = max(
        float(dist.pdf(x)),
        1e-300
    )

    # --------------------------------------------------------
    # Equivalent sigma
    #
    # sigma_N = phi(z) / f_X(x)
    # --------------------------------------------------------

    sigma_N = (
        norm.pdf(z)
        / pdf_x
    )

    sigma_N = max(
        sigma_N,
        1e-12
    )

    # --------------------------------------------------------
    # Equivalent mean
    #
    # x = mu_N + sigma_N*z
    #
    # therefore:
    #
    # mu_N = x - sigma_N*z
    # --------------------------------------------------------

    mu_N = (
        x
        - sigma_N * z
    )

    return (
        float(mu_N),
        float(sigma_N)
    )


# ============================================================
# NUMERICAL GRADIENT
# ============================================================

def numerical_gradient(
    g_func,
    x
):
    """
    Central finite difference.
    """

    n = len(x)

    grad = np.zeros(
        n,
        dtype=float
    )

    for i in range(n):

        h = max(
            abs(x[i]) * 1e-5,
            1e-6
        )

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        g_plus = g_func(
            x_plus
        )

        g_minus = g_func(
            x_minus
        )

        grad[i] = (
            g_plus
            - g_minus
        ) / (
            2.0 * h
        )

    return grad


# ============================================================
# FORM
# ============================================================

def compute_form(
    variables,
    fixed_params,
    capacity_formula,
    demand_formula,
    corrosion_formula,
    t,
    max_iter=100,
    tol_beta=1e-4,
    tol_u=1e-4,
    tol_g=1e-3
):
    """
    Rackwitz-Fiessler / HL-RF FORM.

    g(X) = R(X) - E(X)

    Failure:

        g < 0

    FORM result:

        beta_FORM
        Pf_FORM = Phi(-beta_FORM)
    """

    names = list(
        variables.keys()
    )

    # --------------------------------------------------------
    # scipy distributions
    # --------------------------------------------------------

    dists = {
        name: get_scipy_dist(
            variables[name]
        )
        for name in names
    }

    # --------------------------------------------------------
    # Limit-state function
    # --------------------------------------------------------

    g_func = make_limit_state_function(
        variables=variables,
        fixed_params=fixed_params,
        capacity_formula=capacity_formula,
        demand_formula=demand_formula,
        corrosion_formula=corrosion_formula,
        t=t
    )

    # --------------------------------------------------------
    # Initial point
    #
    # Start at physical means.
    # --------------------------------------------------------

    x = np.array(
        [
            variables[name].mean
            for name in names
        ],
        dtype=float
    )

    beta_prev = 0.0

    converged = False

    final_u = None
    final_g = None

    iteration_count = 0

    # ========================================================
    # HL-RF LOOP
    # ========================================================

    for iteration in range(
        1,
        max_iter + 1
    ):

        iteration_count = iteration

        # ----------------------------------------------------
        # 1. Equivalent normal distributions
        # ----------------------------------------------------

        mu_N = np.zeros(
            len(names)
        )

        sigma_N = np.zeros(
            len(names)
        )

        for i, name in enumerate(names):

            (
                mu_N[i],
                sigma_N[i]
            ) = (
                rackwitz_fiessler_equivalent_normal(
                    dists[name],
                    x[i]
                )
            )

        # ----------------------------------------------------
        # 2. Current standard normal coordinates
        # ----------------------------------------------------

        u = (
            x - mu_N
        ) / sigma_N

        # ----------------------------------------------------
        # 3. Current g
        # ----------------------------------------------------

        g0 = g_func(
            x
        )

        if not np.isfinite(g0):

            break

        # ----------------------------------------------------
        # 4. Gradient in physical space
        # ----------------------------------------------------

        grad_x = numerical_gradient(
            g_func,
            x
        )

        if not np.all(
            np.isfinite(grad_x)
        ):

            break

        # ----------------------------------------------------
        # 5. Gradient in u-space
        #
        # dg/du = dg/dx * dx/du
        #
        # dx/du = sigma_N
        # ----------------------------------------------------

        grad_u = (
            grad_x
            * sigma_N
        )

        norm_grad = np.linalg.norm(
            grad_u
        )

        if (
            not np.isfinite(norm_grad)
            or norm_grad < 1e-12
        ):

            break

        # ----------------------------------------------------
        # 6. Direction cosine
        #
        # Keep one consistent sign convention:
        #
        # alpha = grad_g / |grad_g|
        # ----------------------------------------------------

        alpha = (
            grad_u
            / norm_grad
        )

        # ----------------------------------------------------
        # 7. HL-RF beta
        #
        # beta =
        #
        # alpha.u - g/|grad g|
        # ----------------------------------------------------

        beta_new = (
            np.dot(
                alpha,
                u
            )
            - (
                g0
                / norm_grad
            )
        )

        if not np.isfinite(
            beta_new
        ):

            break

        # ----------------------------------------------------
        # 8. New design point
        # ----------------------------------------------------

        u_new = (
            beta_new
            * alpha
        )

        # ----------------------------------------------------
        # 9. Transform back to physical space
        # ----------------------------------------------------

        x_new = (
            mu_N
            + sigma_N * u_new
        )

        if not np.all(
            np.isfinite(x_new)
        ):

            break

        # ----------------------------------------------------
        # 10. New limit-state value
        # ----------------------------------------------------

        g_new = g_func(
            x_new
        )

        if not np.isfinite(
            g_new
        ):

            break

        # ----------------------------------------------------
        # 11. Convergence metrics
        # ----------------------------------------------------

        beta_change = abs(
            beta_new
            - beta_prev
        )

        u_change = np.linalg.norm(
            u_new
            - u
        )

        g_residual = abs(
            g_new
        )

        # ----------------------------------------------------
        # 12. Save current result
        # ----------------------------------------------------

        final_u = u_new.copy()
        final_g = g_new

        # ----------------------------------------------------
        # 13. Convergence
        #
        # ALL THREE must pass.
        # ----------------------------------------------------

        if (
            beta_change <= tol_beta
            and
            u_change <= tol_u
            and
            g_residual <= tol_g
        ):

            converged = True

            x = x_new

            beta_prev = beta_new

            break

        # ----------------------------------------------------
        # 14. Continue
        # ----------------------------------------------------

        x = x_new

        beta_prev = beta_new

    # ========================================================
    # DO NOT ACCEPT NON-CONVERGED FORM
    # ========================================================

    if not converged:

        return {
            "beta": None,
            "Pf": None,
            "converged": False,

            "g_residual": (
                float(abs(final_g))
                if final_g is not None
                else None
            ),

            "u_norm": (
                float(
                    np.linalg.norm(
                        final_u
                    )
                )
                if final_u is not None
                else None
            ),

            "iterations": iteration_count
        }

    # ========================================================
    # FORM FAILURE PROBABILITY
    # ========================================================

    beta_FORM = float(
        beta_prev
    )

    Pf_FORM = float(
        norm.cdf(
            -beta_FORM
        )
    )

    return {
        "beta": beta_FORM,
        "Pf": Pf_FORM,
        "converged": True,

        "g_residual": float(
            abs(final_g)
        ),

        "u_norm": float(
            np.linalg.norm(
                final_u
            )
        ),

        "iterations": iteration_count
    }


# ============================================================
# MONTE CARLO
# ============================================================

def run_monte_carlo(
    req
):
    """
    Standard crude Monte Carlo.
    """

    rng = np.random.default_rng(
        seed=42
    )

    # --------------------------------------------------------
    # Sample variables
    # --------------------------------------------------------

    sampled = {
        name: sample_variable(
            spec,
            req.N,
            rng
        )
        for name, spec
        in req.variables.items()
    }

    # --------------------------------------------------------
    # Namespace
    # --------------------------------------------------------

    namespace = {
        **sampled,
        **req.fixed_params,
        "t": req.timeStep_years
    }

    # --------------------------------------------------------
    # Corrosion
    # --------------------------------------------------------

    if req.corrosion_formula:

        namespace[
            "degradation_factor"
        ] = safe_eval_formula(
            req.corrosion_formula,
            namespace
        )

    else:

        namespace[
            "degradation_factor"
        ] = 1.0

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    R = safe_eval_formula(
        req.capacity_formula,
        namespace
    )

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    E = safe_eval_formula(
        req.demand_formula,
        namespace
    )

    # --------------------------------------------------------
    # Limit state
    # --------------------------------------------------------

    g = R - E

    # --------------------------------------------------------
    # Failure
    # --------------------------------------------------------

    n_fail = int(
        np.sum(
            g < 0
        )
    )

    # --------------------------------------------------------
    # Raw Monte Carlo Pf
    # --------------------------------------------------------

    raw_Pf = (
        n_fail
        / req.N
    )

    # --------------------------------------------------------
    # Stabilized Pf for beta calculation
    #
    # This avoids +/- infinity when zero failures occur.
    # --------------------------------------------------------

    Pf = min(
        max(
            raw_Pf,
            0.5 / req.N
        ),
        1.0 - 0.5 / req.N
    )

    # --------------------------------------------------------
    # beta
    # --------------------------------------------------------

    beta = float(
        -norm.ppf(
            Pf
        )
    )

    # --------------------------------------------------------
    # COV of MC Pf estimator
    # --------------------------------------------------------

    cov_Pf = None

    if (
        raw_Pf > 0
        and
        raw_Pf < 1
    ):

        cov_Pf = float(
            np.sqrt(
                (
                    1.0
                    - raw_Pf
                )
                /
                (
                    raw_Pf
                    * req.N
                )
            )
        )

    # --------------------------------------------------------
    # Mean response
    # --------------------------------------------------------

    mean_R = float(
        np.mean(R)
    )

    mean_E = float(
        np.mean(E)
    )

    return {
        "n_fail": n_fail,
        "Pf": float(Pf),
        "beta": beta,
        "cov_Pf": cov_Pf,
        "mean_capacity": mean_R,
        "mean_demand": mean_E
    }


# ============================================================
# MAIN RELIABILITY FUNCTION
# ============================================================

def run_reliability(
    req
):

    if req.N <= 0:

        raise HTTPException(
            status_code=400,
            detail="N must be greater than zero."
        )

    # ========================================================
    # MONTE CARLO
    # ========================================================

    mc = run_monte_carlo(
        req
    )

    # ========================================================
    # FORM
    # ========================================================

    form = compute_form(
        variables=req.variables,

        fixed_params=req.fixed_params,

        capacity_formula=req.capacity_formula,

        demand_formula=req.demand_formula,

        corrosion_formula=req.corrosion_formula,

        t=req.timeStep_years
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return ReliabilityResponse(

        timeStep_years=req.timeStep_years,

        N=req.N,

        n_fail=mc[
            "n_fail"
        ],

        Pf=mc[
            "Pf"
        ],

        beta=mc[
            "beta"
        ],

        cov_Pf=mc[
            "cov_Pf"
        ],

        mean_capacity_kNm=mc[
            "mean_capacity"
        ],

        mean_demand_kNm=mc[
            "mean_demand"
        ],

        beta_FORM=form[
            "beta"
        ],

        Pf_FORM=form[
            "Pf"
        ],

        FORM_converged=form[
            "converged"
        ],

        FORM_g_residual=form[
            "g_residual"
        ],

        FORM_u_norm=form[
            "u_norm"
        ],

        FORM_iterations=form[
            "iterations"
        ]
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# RELIABILITY ENDPOINT
# ============================================================

@app.post(
    "/reliability",
    response_model=ReliabilityResponse
)
def reliability_endpoint(
    req: ReliabilityRequest
):

    return run_reliability(
        req
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
