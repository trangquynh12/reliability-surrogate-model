"""
Sanity test for main.py using FastAPI TestClient with synthetic data
that mimics the structure produced by the n8n Monte Carlo pipeline.
"""
import random
import time
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

random.seed(42)

IFC_CLASSES = ["ifcbeam", "ifccolumn", "ifcslab"]


def make_record(with_targets=True):
    ifc_class = random.choice(IFC_CLASSES)
    h = round(random.uniform(0.3, 0.8), 3)
    w = round(random.uniform(0.2, 0.5), 3)
    area = round(h * w, 4)
    f_ck = round(random.uniform(25, 40), 1) if ifc_class == "ifccolumn" else None
    f_yk = round(random.uniform(200, 500), 1)
    dead_load = round(random.uniform(5, 20), 2)
    live_load = round(random.uniform(2, 15), 2)
    corrosion = round(random.uniform(0.05, 0.5), 3)
    t = random.choice([0, 5, 10, 20, 30, 50])

    demand = dead_load + live_load
    capacity_healthy = f_yk * area * 10  # fake but consistent relationship
    capacity_loss = corrosion * t / 100.0
    capacity_damaged = max(0.0001, capacity_healthy * (1 - capacity_loss))

    dcr_healthy = demand / capacity_healthy
    dcr_damaged = demand / capacity_damaged

    record = {
        "ifc_class": ifc_class,
        "profile_height_m": h,
        "profile_width_m": w,
        "cross_section_area_m2": area,
        "f_ck": f_ck,
        "f_yk": f_yk,
        "dead_load": dead_load,
        "live_load": live_load,
        "corrosion_rate_pct_per_year": corrosion,
        "timeStep": t,
    }
    if with_targets:
        record["dcr_healthy"] = round(dcr_healthy, 4)
        record["dcr_damaged"] = round(dcr_damaged, 4)
    return record


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    print("HEALTH:", r.json())


def test_predict_before_train_fails():
    r = client.post("/predict", json={"model_type": "xgboost", "records": [make_record(with_targets=False)]})
    assert r.status_code == 404
    print("PREDICT-BEFORE-TRAIN (expected 404):", r.json())


def test_train_and_predict(model_type, n_train=2000, max_samples_gpr=500):
    records = [make_record() for _ in range(n_train)]
    payload = {"model_type": model_type, "records": records, "max_samples_gpr": max_samples_gpr}

    t0 = time.perf_counter()
    r = client.post("/train", json=payload)
    wall_time = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    result = r.json()
    print(f"\n=== TRAIN [{model_type}] ===")
    print(json.dumps({k: v for k, v in result.items() if k != "feature_importance"}, indent=2))
    print(f"(wall clock train time incl. HTTP overhead: {wall_time:.3f}s)")

    # Predict
    predict_records = [make_record(with_targets=False) for _ in range(50)]
    t0 = time.perf_counter()
    r = client.post("/predict", json={"model_type": model_type, "records": predict_records})
    wall_time = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    result = r.json()
    print(f"=== PREDICT [{model_type}] ===")
    print(f"n_predictions={result['n_predictions']}, "
          f"server_prediction_time={result['prediction_time_seconds']:.5f}s, "
          f"wall_clock_incl_http={wall_time:.5f}s")
    print("Sample prediction:", result["predictions"][0])


if __name__ == "__main__":
    import json
    test_health()
    test_predict_before_train_fails()
    test_train_and_predict("xgboost", n_train=5000)
    test_train_and_predict("gpr", n_train=5000, max_samples_gpr=500)
    print("\nALL TESTS PASSED")
