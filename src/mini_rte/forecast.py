"""Prévisions de demande (baseline ML / fallback régression simple)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _poly_forecast(demand: pd.Series, horizon: int) -> pd.Series:
    """Prévision simple par régression polynomiale sur l'heure."""
    hours = np.arange(len(demand))
    coeffs = np.polyfit(hours, demand.values, deg=2)
    poly = np.poly1d(coeffs)
    future_hours = np.arange(len(demand), len(demand) + horizon)
    forecast_vals = poly(future_hours)
    forecast_vals = np.clip(forecast_vals, a_min=0, a_max=None)
    return pd.Series(forecast_vals, index=range(len(demand), len(demand) + horizon))


def forecast_demand(
    demand: pd.Series,
    horizon: Optional[int] = None,
    method: str = "auto",
    pretrained_model: Optional[object] = None,
) -> pd.Series:
    """Prévoit la demande sur l'horizon souhaité.

    - Essaie RandomForestRegressor si scikit-learn est disponible.
    - Sinon, utilise une régression polynomiale simple.
    - Si un modèle pré-entraîné est passé, il est utilisé directement.
    """
    horizon = horizon or len(demand)
    if horizon <= 0:
        raise ValueError("L'horizon de prévision doit être positif")

    if pretrained_model is not None:
        try:
            hours_future = np.arange(len(demand), len(demand) + horizon).reshape(-1, 1)
            preds = pretrained_model.predict(hours_future)
            preds = np.clip(preds, a_min=0, a_max=None)
            return pd.Series(preds, index=range(len(demand), len(demand) + horizon))
        except Exception as exc:  # pragma: no cover
            logger.warning("Échec du modèle pré-entraîné, fallback auto (%s)", exc)

    if method == "auto":
        try:
            from sklearn.ensemble import RandomForestRegressor

            hours = np.arange(len(demand)).reshape(-1, 1)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(hours, demand.values)
            future_hours = np.arange(len(demand), len(demand) + horizon).reshape(-1, 1)
            preds = model.predict(future_hours)
            preds = np.clip(preds, a_min=0, a_max=None)
            return pd.Series(preds, index=range(len(demand), len(demand) + horizon))
        except Exception as exc:  # pragma: no cover - fallback prévu
            logger.info("RandomForest indisponible (%s), fallback polyfit", exc)
            return _poly_forecast(demand, horizon)
    elif method == "poly":
        return _poly_forecast(demand, horizon)
    else:
        raise ValueError("Méthode de prévision inconnue")


def build_features(demand: pd.Series) -> pd.DataFrame:
    """Construit des features simples à partir d'une série de demande indexée par heure."""
    if demand.index.dtype.kind not in {"i", "u"}:
        demand = demand.reset_index(drop=True)
    df = pd.DataFrame({"load": demand.values})
    df["hour"] = df.index % 24
    df["day"] = (df.index // 24) % 365
    # Encodage sin/cos pour capturer la saisonnalité horaire
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def train_forecast_model(demand: pd.Series) -> object:
    """Entraîne un RandomForest simple sur l'historique fourni."""
    from sklearn.ensemble import RandomForestRegressor

    features = build_features(demand)
    X = features.drop(columns=["load"])
    y = features["load"]
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model


def load_pretrained_model(path: Path) -> object:
    """Charge un modèle pré-entraîné (pickle/joblib)."""
    import joblib

    return joblib.load(path)
