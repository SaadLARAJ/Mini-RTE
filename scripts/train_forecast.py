#!/usr/bin/env python3
"""Télécharge la demande ENTSO-E et entraîne un modèle de prévision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from mini_rte.forecast import build_features, train_forecast_model


def download_entsoe(api_key: str, start: str, end: str) -> pd.Series:
    """Télécharge la charge France depuis ENTSO-E et renvoie une série horaire."""
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=api_key)
    start_ts = pd.Timestamp(start, tz="Europe/Paris")
    end_ts = pd.Timestamp(end, tz="Europe/Paris")
    load_qh = client.query_load(country_code="FR", start=start_ts, end=end_ts)
    load_h = load_qh.resample("H").mean()
    load_h.name = "demand_mw"
    return load_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharger et entraîner le modèle de prévision.")
    parser.add_argument("--api-key", default=os.getenv("ENTSOE_API_KEY"), help="Clé API ENTSO-E")
    parser.add_argument("--start", default="2022-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="Date de fin (YYYY-MM-DD)")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/historical_demand_fr.csv"),
        help="Chemin de sortie du CSV historique",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/forecast_model.pkl"),
        help="Chemin de sortie du modèle entraîné",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Clé API ENTSO-E manquante. Renseignez ENTSOE_API_KEY.")

    args.data_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement ENTSO-E de {args.start} à {args.end}...")
    demand = download_entsoe(args.api_key, args.start, args.end)
    demand.to_csv(args.data_path, header=True)
    print(f"Historique enregistré dans {args.data_path}")

    print("Entraînement du modèle (RandomForest)...")
    model = train_forecast_model(demand.reset_index(drop=True))

    import joblib

    joblib.dump(model, args.model_path)
    print(f"Modèle sauvegardé dans {args.model_path}")


if __name__ == "__main__":
    main()
