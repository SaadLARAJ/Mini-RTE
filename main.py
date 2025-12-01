#!/usr/bin/env python3
"""
Script principal pour tester le modèle d'Unit Commitment en ligne de commande.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from mini_rte import ConfigLoader
from mini_rte.data_loader import load_all_availability, load_demand
from mini_rte.models.unit_commitment import UnitCommitmentModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Charge la configuration depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier de configuration

    Returns:
        Dictionnaire de configuration
    """
def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Résout le problème d'Unit Commitment pour un réseau électrique"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Chemin vers le fichier de configuration",
    )
    parser.add_argument(
        "--demand",
        type=str,
        default="data/demand_profile_24h.csv",
        help="Chemin vers le fichier de demande",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="appsi_highs",
        choices=["cbc", "glpk", "appsi_highs"],
        help="Solveur à utiliser",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout en secondes",
    )

    args = parser.parse_args()

    # Chemins
    project_root = Path(__file__).parent
    config_path = project_root / args.config
    demand_path = project_root / args.demand
    data_dir = project_root / "data"

    # Chargement des données
    logger.info(f"Chargement de la configuration depuis {config_path}")
    config = ConfigLoader.load_yaml(config_path)
    ConfigLoader.validate_config(config)

    logger.info(f"Chargement de la demande depuis {demand_path}")
    demand = load_demand(demand_path)

    logger.info("Création des centrales")
    plants = ConfigLoader.load_plants(config)
    global_params = ConfigLoader.load_global_params(config)
    logger.info(f"{len(plants)} centrales chargées")

    # Chargement de la disponibilité des renouvelables
    renewable_names = [p.name for p in plants if p.is_renewable]
    availability = pd.DataFrame()
    if renewable_names:
        logger.info("Chargement de la disponibilité des énergies renouvelables")
        availability = load_all_availability(data_dir, renewable_names)

    # Construction du modèle
    logger.info("Construction du modèle Pyomo...")
    model = UnitCommitmentModel(
        plants=plants,
        demand=demand,
        availability=availability,
        co2_price=global_params.co2_price,
        voll=global_params.voll,
        reserve_margin=global_params.reserve_margin,
    )
    model.build_model()

    # Résolution
    logger.info(f"Résolution avec le solveur {args.solver}...")
    result = model.solve(solver_name=args.solver, timeout=args.timeout)

    # Affichage des résultats
    print("\n" + "=" * 80)
    print("RÉSULTATS DE L'OPTIMISATION")
    print("=" * 80)
    print(f"\nStatut: {result.status}")
    print(f"Succès: {result.success}")
    print(f"\nCoût total: {result.objective_value:,.2f} €")
    print(f"Émissions CO2: {result.co2_emissions:,.2f} t")
    print(f"Délestage total: {result.total_load_shedding:,.2f} MWh")
    print(f"Prix marginal moyen: {result.marginal_prices.mean():,.2f} €/MWh")

    print("\n" + "-" * 80)
    print("DÉCOMPOSITION DES COÛTS")
    print("-" * 80)
    for cost_type, cost_value in result.cost_breakdown.items():
        if cost_type != "total":
            print(f"  {cost_type.capitalize()}: {cost_value:,.2f} €")
    print(f"  Total: {result.cost_breakdown.get('total', result.objective_value):,.2f} €")

    print("\n" + "-" * 80)
    print("PLANNING DE PRODUCTION (MW)")
    print("-" * 80)
    print(result.production_schedule.round(2))

    print("\n" + "-" * 80)
    print("ÉTAT DES CENTRALES (1=allumée, 0=éteinte)")
    print("-" * 80)
    print(result.commitment_schedule)

    print("\n" + "-" * 80)
    print("DÉMARRAGES (1=démarrage à cette heure)")
    print("-" * 80)
    print(result.startup_schedule)

    # Vérification de la satisfaction de la demande
    print("\n" + "-" * 80)
    print("VÉRIFICATION DE LA DEMANDE")
    print("-" * 80)
    total_production = result.production_schedule.sum(axis=1)
    shed = result.load_shedding
    for hour in range(len(demand)):
        prod = total_production.iloc[hour]
        shed_val = shed.iloc[hour]
        dem = demand.iloc[hour]
        coverage = prod + shed_val - dem
        print(
            f"Heure {hour:2d}: Prod={prod:8.2f} MW, Délestage={shed_val:6.2f} MW, "
            f"Demande={dem:8.2f} MW, Écart={coverage:7.2f} MW"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
