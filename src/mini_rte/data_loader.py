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
    Charge le profil de demande depuis un fichier CSV.

    Le fichier doit contenir au minimum une colonne avec les heures et une colonne
    avec la demande en MW. Formats acceptés :
    - Colonnes "hour" et "demand_mw"
    - Première colonne = heures, deuxième colonne = demande

    Args:
        demand_path: Chemin vers le fichier CSV

    Returns:
        Série pandas avec la demande horaire (index = heures, valeurs = MW)

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format du fichier est invalide
    """
    if not demand_path.exists():
        raise FileNotFoundError(f"Fichier de demande non trouvé: {demand_path}")

    try:
        df = pd.read_csv(demand_path)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier CSV: {e}") from e

    # Format 1 : colonnes "hour" et "demand_mw"
    if "hour" in df.columns and "demand_mw" in df.columns:
        demand = pd.Series(
            df["demand_mw"].values, index=df["hour"].values, name="demand_mw"
        )
    # Format 2 : première colonne = index, deuxième = valeurs
    elif len(df.columns) >= 2:
        demand = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values, name="demand_mw")
    else:
        raise ValueError(
            "Format de fichier invalide. Attendu: colonnes 'hour' et 'demand_mw' "
            "ou au moins 2 colonnes (heures, demande)"
        )

    # Validation
    if len(demand) == 0:
        raise ValueError("Le fichier de demande est vide")

    if (demand < 0).any():
        raise ValueError("La demande ne peut pas contenir de valeurs négatives")

    if demand.isna().any():
        raise ValueError("La demande contient des valeurs manquantes (NaN)")

    logger.info(f"Demande chargée: {len(demand)} heures, min={demand.min():.0f} MW, max={demand.max():.0f} MW")
    return demand


def load_availability(
    availability_path: Path, plant_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Charge les données de disponibilité des énergies renouvelables depuis un fichier CSV.

    Le fichier doit contenir une colonne "hour" et une colonne par centrale renouvelable.
    Les valeurs doivent être entre 0 et 1 (facteur de disponibilité).

    Args:
        availability_path: Chemin vers le fichier CSV
        plant_names: Liste optionnelle des noms de centrales à charger (filtre les colonnes)

    Returns:
        DataFrame avec les heures en index et les centrales en colonnes

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si le format du fichier est invalide
    """
    if not availability_path.exists():
        raise FileNotFoundError(f"Fichier de disponibilité non trouvé: {availability_path}")

    try:
        df = pd.read_csv(availability_path)
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier CSV: {e}") from e

    if "hour" not in df.columns:
        raise ValueError("Le fichier de disponibilité doit contenir une colonne 'hour'")

    # Utiliser "hour" comme index
    df = df.set_index("hour")

    # Filtrer par noms de centrales si spécifié
    if plant_names is not None:
        available_cols = [col for col in df.columns if col in plant_names]
        if not available_cols:
            logger.warning(f"Aucune colonne correspondant aux centrales {plant_names} dans {availability_path}")
            return pd.DataFrame()
        df = df[available_cols]

    # Validation
    if len(df) == 0:
        raise ValueError("Le fichier de disponibilité est vide après filtrage")

    if (df < 0).any().any():
        raise ValueError("La disponibilité ne peut pas contenir de valeurs négatives")

    if (df > 1).any().any():
        logger.warning("Certaines valeurs de disponibilité sont > 1, elles seront limitées à 1")
        df = df.clip(upper=1.0)

    if df.isna().any().any():
        raise ValueError("La disponibilité contient des valeurs manquantes (NaN)")

    logger.info(
        f"Disponibilité chargée: {len(df)} heures, {len(df.columns)} centrales: {list(df.columns)}"
    )
    return df


def load_all_availability(
    data_dir: Path, renewable_plant_names: list[str]
) -> pd.DataFrame:
    """
    Charge toutes les données de disponibilité depuis le répertoire data.

    Cherche les fichiers wind_availability.csv et solar_availability.csv
    et les combine en un seul DataFrame.

    Args:
        data_dir: Répertoire contenant les fichiers de disponibilité
        renewable_plant_names: Liste des noms de centrales renouvelables

    Returns:
        DataFrame combiné avec toutes les disponibilités
    """
    availability = pd.DataFrame()

    # Fichiers à chercher
    availability_files = ["wind_availability.csv", "solar_availability.csv"]

    for filename in availability_files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = load_availability(filepath, plant_names=renewable_plant_names)
                if not df.empty:
                    availability = pd.concat([availability, df], axis=1)
            except Exception as e:
                logger.warning(f"Impossible de charger {filename}: {e}")

    if availability.empty:
        logger.warning("Aucune donnée de disponibilité chargée")
        return pd.DataFrame()

    # Supprimer les doublons de colonnes (si une centrale apparaît dans plusieurs fichiers)
    availability = availability.loc[:, ~availability.columns.duplicated()]

    logger.info(f"Disponibilité totale chargée: {len(availability.columns)} centrales")
    return availability


class DataLoader:
    """Interface simple pour charger demande et disponibilité."""

    @staticmethod
    def demand(path: Path) -> pd.Series:
        return load_demand(path)

    @staticmethod
    def availability(path: Path, plant_names: Optional[list[str]] = None) -> pd.DataFrame:
        return load_availability(path, plant_names=plant_names)

    @staticmethod
    def all_availability(data_dir: Path, renewable_plant_names: list[str]) -> pd.DataFrame:
        return load_all_availability(data_dir, renewable_plant_names)
