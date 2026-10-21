"""Data loading, cleaning and feature engineering for the Kaggle Cardiovascular Disease dataset.

Expected file: data/cardio_train.csv (semicolon separated), columns:
id, age (days), gender (1=female, 2=male), height (cm), weight (kg), ap_hi, ap_lo,
cholesterol (1-3), gluc (1-3), smoke, alco, active, cardio (target)
"""
import numpy as np
import pandas as pd

FEATURES = [
    "age_years", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo", "pulse_pressure",
    "cholesterol", "gluc", "smoke", "alco", "active",
]
TARGET = "cardio"


def load_raw(path):
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:  # fallback if comma separated
        df = pd.read_csv(path)
    return df


def clean(df):
    """Convert units, remove physiologically implausible rows, add engineered features."""
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "age" in df.columns:  # age is stored in days in the Kaggle file
        df["age_years"] = (df["age"] / 365.25).round(1)
        df = df.drop(columns=["age"])
    n0 = len(df)
    df = df[(df.height.between(120, 220)) & (df.weight.between(30, 200))]
    df = df[(df.ap_hi.between(70, 250)) & (df.ap_lo.between(40, 200)) & (df.ap_hi > df.ap_lo)]
    df = df.drop_duplicates()
    df["bmi"] = df.weight / (df.height / 100) ** 2
    df["pulse_pressure"] = df.ap_hi - df.ap_lo
    print(f"[clean] kept {len(df)} of {n0} rows ({n0 - len(df)} removed as implausible/duplicate)")
    return df.reset_index(drop=True)


def make_synthetic(n=20000, seed=0):
    """Synthetic stand-in with the same schema; ONLY for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(30, 65, n)
    gender = rng.choice([1, 2], n, p=[0.65, 0.35])
    height = rng.normal(165, 8, n) + (gender == 2) * 8
    weight = rng.normal(74, 14, n)
    ap_hi = rng.normal(125, 15, n) + (age - 50) * 0.5
    ap_lo = ap_hi - rng.normal(45, 8, n)
    chol = rng.choice([1, 2, 3], n, p=[0.75, 0.13, 0.12])
    gluc = rng.choice([1, 2, 3], n, p=[0.85, 0.07, 0.08])
    smoke = rng.binomial(1, 0.09, n)
    alco = rng.binomial(1, 0.05, n)
    active = rng.binomial(1, 0.8, n)
    z = (0.06 * (age - 50) + 0.05 * (ap_hi - 125) + 0.5 * (chol - 1) + 0.2 * (gluc - 1)
         + 0.03 * (weight - 74) - 0.3 * active + 0.2 * smoke)
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({"age": age * 365.25, "gender": gender, "height": height.round(),
                         "weight": weight.round(1), "ap_hi": ap_hi.round(), "ap_lo": ap_lo.round(),
                         "cholesterol": chol, "gluc": gluc, "smoke": smoke, "alco": alco,
                         "active": active, "cardio": y})
