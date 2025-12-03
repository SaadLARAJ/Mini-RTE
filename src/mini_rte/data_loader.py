"""
Chargement et validation des données (demande, disponibilité renouvelables).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_demand(demand_path: Path) -> pd.Series:
    """
    Charge le profil de demande depuis un CSV.
    
    Formats supportés:
    - Colonnes "hour" et "demand_mw"
    - 1ère colonne = index, 2ème colonne = valeurs
    """
    if not demand_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {demand_path}")

    try:
        df = pd.read_csv(demand_path)
    except Exception as e:
        raise ValueError(f"Erreur lecture CSV: {e}") from e

    # Format 1 : colonnes "hour" et "demand_mw"
    if "hour" in df.columns and "demand_mw" in df.columns:
        demand = pd.Series(
            df["demand_mw"].values, index=df["hour"].values, name="demand_mw"
        )
    # Format 2 : première colonne = index, deuxième = valeurs
    elif len(df.columns) >= 2:
        demand = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values, name="demand_mw")
    else:
        raise ValueError("Format CSV invalide (attendu: 'hour', 'demand_mw')")

    if len(demand) == 0:
        raise ValueError("Fichier demande vide")

    if (demand < 0).any():
        raise ValueError("Valeurs négatives dans la demande")

    if demand.isna().any():
        raise ValueError("NaN détectés dans la demande")

    logger.info(f"Demande chargée: {len(demand)} pts, range=[{demand.min():.0f}, {demand.max():.0f}] MW")
    return demand


def load_availability(
    availability_path: Path, plant_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Charge les disponibilités ENR (facteurs 0-1).
    """
    if not availability_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {availability_path}")

    try:
        df = pd.read_csv(availability_path)
    except Exception as e:
        raise ValueError(f"Erreur lecture CSV: {e}") from e

    if "hour" not in df.columns:
        raise ValueError("Colonne 'hour' manquante")

    df = df.set_index("hour")

    if plant_names is not None:
        available_cols = [col for col in df.columns if col in plant_names]
        if not available_cols:
            logger.warning(f"Aucune colonne correspondante dans {availability_path}")
            return pd.DataFrame()
        df = df[available_cols]

    if len(df) == 0:
        raise ValueError("Données vides après filtrage")

    if (df < 0).any().any():
        raise ValueError("Disponibilité négative détectée")

    if (df > 1).any().any():
        logger.warning("Valeurs > 1 détectées, écrêtage à 1.0")
        df = df.clip(upper=1.0)

    if df.isna().any().any():
        raise ValueError("NaN détectés dans la disponibilité")

    logger.info(
        f"Disponibilité chargée: {len(df)} pts, {len(df.columns)} centrales"
    )
    return df


def load_all_availability(
    data_dir: Path, renewable_plant_names: list[str]
) -> pd.DataFrame:
    """
    Agrège les disponibilités (wind + solar).
    """
    availability = pd.DataFrame()
    availability_files = ["wind_availability.csv", "solar_availability.csv"]

    for filename in availability_files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = load_availability(filepath, plant_names=renewable_plant_names)
                if not df.empty:
                    availability = pd.concat([availability, df], axis=1)
            except Exception as e:
                logger.warning(f"Echec chargement {filename}: {e}")

    if availability.empty:
        logger.warning("Aucune donnée de disponibilité chargée")
        return pd.DataFrame()

    # Dédoublonnage colonnes
    availability = availability.loc[:, ~availability.columns.duplicated()]

    logger.info(f"Total disponibilités chargées: {len(availability.columns)} centrales")
    return availability


class DataLoader:
    """Façade pour le chargement des données."""

    @staticmethod
    def demand(path: Path) -> pd.Series:
        return load_demand(path)

    @staticmethod
    def availability(path: Path, plant_names: Optional[list[str]] = None) -> pd.DataFrame:
        return load_availability(path, plant_names=plant_names)

    @staticmethod
    def all_availability(data_dir: Path, renewable_plant_names: list[str]) -> pd.DataFrame:
        return load_all_availability(data_dir, renewable_plant_names)
