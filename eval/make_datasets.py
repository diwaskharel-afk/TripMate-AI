"""Generates the 3 small, deterministic benchmark datasets used by run_eval.py
— one per pipeline branch: pure EDA, supervised regression, time-series
forecasting. Re-run with `python -m eval.make_datasets` if you need to
regenerate them; they're committed to the repo so eval runs are reproducible
without regenerating data each time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "datasets"
SEED = 42


def make_eda_survey(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    departments = rng.choice(["Engineering", "Sales", "Support", "Marketing"], size=n)
    satisfaction = np.clip(rng.normal(7.0, 1.5, size=n), 1, 10).round(1)
    tenure_years = np.clip(rng.exponential(3.0, size=n), 0, 25).round(1)
    remote = rng.choice(["yes", "no"], size=n, p=[0.4, 0.6])
    return pd.DataFrame(
        {
            "employee_id": range(1, n + 1),
            "department": departments,
            "tenure_years": tenure_years,
            "remote": remote,
            "satisfaction_score": satisfaction,
        }
    )


def make_regression(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    age = rng.integers(18, 65, size=n)
    bmi = np.clip(rng.normal(27, 5, size=n), 15, 50).round(1)
    children = rng.integers(0, 5, size=n)
    smoker = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], size=n)
    smoker_penalty = np.where(smoker == "yes", 20000, 0)
    charges = (
        250 * age + 300 * bmi + 500 * children + smoker_penalty
        + rng.normal(0, 2000, size=n)
        + 1000
    ).round(2)
    charges = np.clip(charges, 500, None)
    return pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges,
        }
    )


def make_timeseries(n_days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    trend = 50 + 0.05 * t
    weekly_season = 10 * np.sin(2 * np.pi * t / 7)
    yearly_season = 20 * np.sin(2 * np.pi * t / 365.25)
    marketing_spend = np.clip(rng.normal(100, 20, size=n_days), 0, None).round(1)
    noise = rng.normal(0, 5, size=n_days)
    sales = trend + weekly_season + yearly_season + 0.3 * marketing_spend + noise
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "marketing_spend": marketing_spend,
            "daily_sales": sales.round(2),
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_eda_survey().to_csv(OUT_DIR / "eda_survey.csv", index=False)
    make_regression().to_csv(OUT_DIR / "insurance_like.csv", index=False)
    make_timeseries().to_csv(OUT_DIR / "daily_sales.csv", index=False)
    print(f"Wrote 3 datasets to {OUT_DIR}")


if __name__ == "__main__":
    main()
