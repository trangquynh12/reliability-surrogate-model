"""
VALIDATION SCRIPT — run manually and locally only.
This file is NEVER imported by main.py and is NOT deployed as an API route.
It exists solely to prove run_reliability() reproduces a known, independently
verified worked example before the formula is trusted on real project data.
"""

from main import ReliabilityRequest, run_reliability

def test_golden_example():
    req = ReliabilityRequest(
        N=100_000,
        timeStep_years=0.0,
        demand_dead_mean_kNm=967.91,
        demand_live_mean_kNm=1072.63,
        demand_dead_cov=0.10,
        demand_live_cov=0.18,
        A_p_mean_mm2=2000.0,
        A_p_cov=0.015,
        f_ps_mean_MPa=1750.0,
        f_ps_cov=0.025,
        f_ck_MPa=40.0,
        b_f_m=1.5,
        d_p_m=1.22,
        theta_R_mean=1.00,
        theta_R_cov=0.09,
        T_i_years=20.0,
        k_A=0.02
    )
    result = run_reliability(req)

    print(f"Mean capacity: {result.mean_capacity_kNm:.0f} kN·m (expected ~ 4,150)")
    print(f"Mean demand:   {result.mean_demand_kNm:.0f} kN·m (expected ~ 2,040)")
    print(f"Margin:        {result.mean_capacity_kNm / result.mean_demand_kNm:.2f} (expected ~ 2.0)")
    print(f"Beta:          {result.beta:.2f}")

    assert 3800 < result.mean_capacity_kNm < 4500, "Capacity does not match the reference example!"
    print("PASSED")

if __name__ == "__main__":
    test_golden_example()
