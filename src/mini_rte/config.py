"""
Chargement et validation de la configuration YAML.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from mini_rte.models.plant import PowerPlant

logger = logging.getLogger(__name__)

EMISSION_FACTORS = {
    "nuclear": 0.006,
    "hydro": 0.006,
    "gas": 0.350,
    "oil": 0.500,
    "coal": 0.900,
    "wind": 0.0,
    "solar": 0.0,
}


@dataclass
class PlantConfig:
    """Configuration d'une centrale électrique."""

    name: str
    type: str
    p_min: float
    p_max: float
    cost_variable: float
    cost_fixed: float
    cost_startup: float
    ramp_rate: float
    is_renewable: bool = False
    emission_factor: float = 0.0


@dataclass
class SolverConfig:
    """Configuration du solveur."""

    name: str = "cbc"
    timeout: int = 300
    mip_gap: float = 0.01
    threads: int = 0
    log_level: str = "INFO"


@dataclass
class ScenarioConfig:
    """Configuration d'un scénario."""

    name: str
    modifications: Dict[str, any]


@dataclass
class GlobalParams:
    """Paramètres globaux (CO2, VoLL, réserve)."""

    co2_price: float = 80.0
    voll: float = 20_000.0
    reserve_margin: float = 0.0


class ConfigLoader:
    """Chargeur et validateur de configuration."""

    @staticmethod
    def load_yaml(config_path: Path) -> Dict:
        """
        Charge un fichier YAML.

        Args:
            config_path: Chemin vers le fichier YAML

        Returns:
            Dictionnaire de configuration

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            yaml.YAMLError: Si le fichier YAML est invalide
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Erreur de parsing YAML: {e}") from e

        logger.info(f"Configuration chargée depuis {config_path}")
        return config

    @staticmethod
    def load_plants(config: Dict) -> List[PowerPlant]:
        """
        Crée la liste des centrales à partir de la configuration.

        Args:
            config: Dictionnaire de configuration

        Returns:
            Liste des objets PowerPlant

        Raises:
            ValueError: Si la configuration est invalide
        """
        if "plants" not in config:
            raise ValueError("La configuration doit contenir une section 'plants'")

        plants = []
        for plant_config in config["plants"]:
            try:
                plant = PowerPlant(
                    name=plant_config["name"],
                    plant_type=plant_config["type"],
                    p_min=float(plant_config["p_min"]),
                    p_max=float(plant_config["p_max"]),
                    cost_variable=float(plant_config["cost_variable"]),
                    cost_fixed=float(plant_config["cost_fixed"]),
                    cost_startup=float(plant_config["cost_startup"]),
                    ramp_rate=float(plant_config["ramp_rate"]),
                    is_renewable=plant_config.get("is_renewable", False),
                    emission_factor=float(
                        plant_config.get(
                            "emission_factor",
                            EMISSION_FACTORS.get(plant_config["type"], 0.0),
                        )
                    ),
                )
                plants.append(plant)
            except KeyError as e:
                raise ValueError(f"Paramètre manquant dans la configuration de la centrale: {e}")
            except ValueError as e:
                raise ValueError(f"Erreur de validation pour la centrale {plant_config.get('name', 'unknown')}: {e}")

        logger.info(f"{len(plants)} centrales chargées depuis la configuration")
        return plants

    @staticmethod
    def load_solver_config(config: Dict) -> SolverConfig:
        """
        Charge la configuration du solveur.

        Args:
            config: Dictionnaire de configuration

        Returns:
            Configuration du solveur
        """
        solver_dict = config.get("solver", {})
        return SolverConfig(
            name=solver_dict.get("name", "cbc"),
            timeout=solver_dict.get("timeout", 300),
            mip_gap=solver_dict.get("mip_gap", 0.01),
            threads=solver_dict.get("threads", 0),
            log_level=solver_dict.get("log_level", "INFO"),
        )

    @staticmethod
    def load_global_params(config: Dict) -> GlobalParams:
        """Charge les paramètres globaux (CO2, VoLL, réserve)."""
        params = config.get("global_params", {})
        return GlobalParams(
            co2_price=float(params.get("co2_price", 80.0)),
            voll=float(params.get("voll", 20_000.0)),
            reserve_margin=float(params.get("reserve_margin", 0.0)),
        )

    @staticmethod
    def validate_config(config: Dict) -> None:
        """
        Valide la structure de la configuration.

        Args:
            config: Dictionnaire de configuration

        Raises:
            ValueError: Si la configuration est invalide
        """
        required_sections = ["plants"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Section '{section}' manquante dans la configuration")

        if not isinstance(config["plants"], list):
            raise ValueError("La section 'plants' doit être une liste")

        if len(config["plants"]) == 0:
            raise ValueError("Au moins une centrale doit être définie")

        # Valider chaque centrale
        required_plant_params = [
            "name",
            "type",
            "p_min",
            "p_max",
            "cost_variable",
            "cost_fixed",
            "cost_startup",
            "ramp_rate",
        ]

        for i, plant in enumerate(config["plants"]):
            for param in required_plant_params:
                if param not in plant:
                    raise ValueError(
                        f"Paramètre '{param}' manquant pour la centrale {i+1} ({plant.get('name', 'unknown')})"
                    )

        logger.info("Configuration validée avec succès")


def load_config(config_path: Path) -> Dict:
    """Alias pratique pour charger un YAML de configuration."""
    return ConfigLoader.load_yaml(config_path)
