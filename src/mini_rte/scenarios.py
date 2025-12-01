"""
Générateur de scénarios pour l'analyse de sensibilité.
"""

import copy
import logging
from typing import Any, Dict, List

from mini_rte.config import ConfigLoader
from mini_rte.models.plant import PowerPlant

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """Générateur de scénarios pour modifier la configuration de base."""

    @staticmethod
    def apply_scenario_modifications(
        plants: List[PowerPlant], modifications: Dict[str, Any]
    ) -> List[PowerPlant]:
        """
        Applique des modifications à une liste de centrales.

        Args:
            plants: Liste des centrales originales
            modifications: Dictionnaire de modifications à appliquer

        Returns:
            Nouvelle liste de centrales modifiées
        """
        modified_plants = []

        for plant in plants:
            # Créer une copie de la centrale
            new_plant = PowerPlant(
                name=plant.name,
                plant_type=plant.plant_type,
                p_min=plant.p_min,
                p_max=plant.p_max,
                cost_variable=plant.cost_variable,
                cost_fixed=plant.cost_fixed,
                cost_startup=plant.cost_startup,
                ramp_rate=plant.ramp_rate,
                is_renewable=plant.is_renewable,
                emission_factor=plant.emission_factor,
            )

            # Appliquer les modifications selon le type de centrale
            plant_type = plant.plant_type.lower()

            # Multiplicateur de coût variable du gaz
            if "gas_cost_multiplier" in modifications and plant_type == "gas":
                new_plant.cost_variable *= modifications["gas_cost_multiplier"]
                logger.debug(
                    f"Coût variable du gaz modifié pour {plant.name}: "
                    f"{plant.cost_variable:.2f} -> {new_plant.cost_variable:.2f} €/MWh"
                )

            # Disponibilité nucléaire (réduire p_max)
            if "nuclear_availability" in modifications and plant_type == "nuclear":
                availability_factor = modifications["nuclear_availability"]
                new_plant.p_max *= availability_factor
                new_plant.p_min = min(new_plant.p_min, new_plant.p_max)
                logger.debug(
                    f"Disponibilité nucléaire modifiée pour {plant.name}: "
                    f"p_max = {new_plant.p_max:.0f} MW"
                )

            modified_plants.append(new_plant)

        return modified_plants

    @staticmethod
    def modify_availability(
        availability: "pd.DataFrame", modifications: Dict[str, Any]
    ) -> "pd.DataFrame":
        """
        Modifie les données de disponibilité des renouvelables.

        Args:
            availability: DataFrame de disponibilité original
            modifications: Dictionnaire de modifications

        Returns:
            DataFrame modifié
        """
        import pandas as pd

        modified_avail = availability.copy()

        # Facteur de disponibilité éolienne
        if "wind_availability_factor" in modifications:
            factor = modifications["wind_availability_factor"]
            wind_cols = [col for col in modified_avail.columns if "wind" in col.lower()]
            for col in wind_cols:
                modified_avail[col] *= factor
                logger.debug(f"Disponibilité éolienne modifiée pour {col}: facteur {factor}")

        # Facteur de disponibilité solaire
        if "solar_availability_factor" in modifications:
            factor = modifications["solar_availability_factor"]
            solar_cols = [col for col in modified_avail.columns if "solar" in col.lower()]
            for col in solar_cols:
                modified_avail[col] *= factor
                logger.debug(f"Disponibilité solaire modifiée pour {col}: facteur {factor}")

        # Limiter à [0, 1]
        modified_avail = modified_avail.clip(lower=0.0, upper=1.0)

        return modified_avail

    @staticmethod
    def generate_scenario(
        base_config: Dict,
        scenario_name: str,
        scenario_modifications: Dict[str, Any],
    ) -> Dict:
        """
        Génère une configuration de scénario à partir d'une configuration de base.

        Args:
            base_config: Configuration de base (dictionnaire)
            scenario_name: Nom du scénario
            scenario_modifications: Modifications à appliquer

        Returns:
            Nouvelle configuration de scénario
        """
        # Copie profonde de la configuration
        scenario_config = copy.deepcopy(base_config)

        # Appliquer les modifications
        if "plants" in scenario_config:
            plants = ConfigLoader.load_plants(scenario_config)
            modified_plants = ScenarioGenerator.apply_scenario_modifications(
                plants, scenario_modifications
            )

            # Reconstruire la liste de plantes dans la config
            scenario_config["plants"] = []
            for plant in modified_plants:
                scenario_config["plants"].append({
                    "name": plant.name,
                    "type": plant.plant_type,
                    "p_min": plant.p_min,
                    "p_max": plant.p_max,
                    "cost_variable": plant.cost_variable,
                    "cost_fixed": plant.cost_fixed,
                    "cost_startup": plant.cost_startup,
                    "ramp_rate": plant.ramp_rate,
                    "is_renewable": plant.is_renewable,
                    "emission_factor": plant.emission_factor,
                })

        logger.info(f"Scénario '{scenario_name}' généré avec {len(scenario_modifications)} modifications")
        return scenario_config


# Scénarios prédéfinis
PREDEFINED_SCENARIOS = {
    "low_wind": {
        "wind_availability_factor": 0.3,  # 30% de disponibilité normale
        "description": "Vent faible - disponibilité éolienne réduite à 30%",
    },
    "high_gas_price": {
        "gas_cost_multiplier": 1.5,  # +50% sur le coût variable du gaz
        "description": "Prix du gaz élevé - coût variable du gaz multiplié par 1.5",
    },
    "nuclear_outage": {
        "nuclear_availability": 0.5,  # 50% des centrales nucléaires disponibles
        "description": "Panne nucléaire - disponibilité réduite à 50%",
    },
    "summer_solar": {
        "solar_availability_factor": 1.3,  # +30% de disponibilité solaire
        "description": "Pic solaire estival - disponibilité solaire augmentée de 30%",
    },
    "combined_stress": {
        "wind_availability_factor": 0.4,
        "gas_cost_multiplier": 1.3,
        "nuclear_availability": 0.7,
        "description": "Scénario de stress combiné - vent faible, gaz cher, nucléaire réduit",
    },
    "low_co2": {
        "co2_price": 25,
        "description": "Prix CO2 pré-2021 (25 €/t)",
    },
    "high_co2": {
        "co2_price": 100,
        "description": "Prix CO2 crise 2022 (100 €/t)",
    },
    "crisis_2022": {
        "co2_price": 100,
        "gas_cost_multiplier": 3.0,
        "description": "Simulation crise énergétique 2022 (gaz x3, CO2 100 €/t)",
    },
    "backtest_2022": {
        "demand_profile": "demand_profile_2022.csv",
        "gas_cost_multiplier": 3.0,
        "co2_price": 100,
        "description": "Backtest 2022 (demande +10 %, gaz x3, CO2 100 €/t)",
    },
    "backtest_2023": {
        "demand_profile": "demand_profile_2023.csv",
        "co2_price": 80,
        "description": "Backtest 2023 (demande -5 %, prix normal)",
    },
}


def get_predefined_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Récupère un scénario prédéfini.

    Args:
        scenario_name: Nom du scénario

    Returns:
        Dictionnaire avec les modifications du scénario

    Raises:
        ValueError: Si le scénario n'existe pas
    """
    if scenario_name not in PREDEFINED_SCENARIOS:
        available = ", ".join(PREDEFINED_SCENARIOS.keys())
        raise ValueError(
            f"Scénario '{scenario_name}' non trouvé. Scénarios disponibles: {available}"
        )

    return PREDEFINED_SCENARIOS[scenario_name]
