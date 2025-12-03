#!/usr/bin/env python3
"""
Point d'entrée CLI pour la résolution du problème d'Unit Commitment.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from mini_rte import ConfigLoader
from mini_rte.data_loader import load_all_availability, load_demand
from mini_rte.models.unit_commitment import UnitCommitmentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Résolution du problème d'Unit Commitment (MILP)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Fichier de configuration",
    )
    parser.add_argument(
        "--demand",
        type=str,
        default="data/demand_profile_24h.csv",
        help="Fichier de profil de demande",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="cbc",
        choices=["cbc", "glpk", "appsi_highs"],
        help="Solveur MILP",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout (secondes)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent
    config_path = project_root / args.config
    demand_path = project_root / args.demand
    data_dir = project_root / "data"

    logger.info(f"Chargement configuration: {config_path}")
    config = ConfigLoader.load_yaml(config_path)
    ConfigLoader.validate_config(config)

    logger.info(f"Chargement demande: {demand_path}")
    demand = load_demand(demand_path)

    plants = ConfigLoader.load_plants(config)
    global_params = ConfigLoader.load_global_params(config)
    logger.info(f"Centrales chargées: {len(plants)}")

    renewable_names = [p.name for p in plants if p.is_renewable]
    availability = pd.DataFrame()
    if renewable_names:
        logger.info("Chargement profils ENR")
        availability = load_all_availability(data_dir, renewable_names)

    logger.info("Construction du modèle MILP...")
    model = UnitCommitmentModel(
        plants=plants,
        demand=demand,
        availability=availability,
        co2_price=global_params.co2_price,
        voll=global_params.voll,
        reserve_margin=global_params.reserve_margin,
    )
    model.build_model()

    logger.info(f"Résolution (solver={args.solver})...")
    result = model.solve(solver_name=args.solver, timeout=args.timeout)

    print("\n" + "=" * 80)
    print("RÉSULTATS OPTIMISATION")
    print("=" * 80)
    print(f"\nStatut: {result.status}")
    print(f"Succès: {result.success}")
    print(f"\nCoût total: {result.objective_value:,.2f} €")
    print(f"Émissions CO2: {result.co2_emissions:,.2f} t")
    print(f"Délestage: {result.total_load_shedding:,.2f} MWh")
    print(f"Prix marginal moyen: {result.marginal_prices.mean():,.2f} €/MWh")

    print("\n" + "-" * 80)
    print("DÉTAIL DES COÛTS")
    print("-" * 80)
    for cost_type, cost_value in result.cost_breakdown.items():
        if cost_type != "total":
            print(f"  {cost_type.capitalize()}: {cost_value:,.2f} €")
    print(f"  Total: {result.cost_breakdown.get('total', result.objective_value):,.2f} €")

    print("\n" + "-" * 80)
    print("PLAN DE PRODUCTION (MW)")
    print("-" * 80)
    print(result.production_schedule.round(2))

    print("\n" + "-" * 80)
    print("ENGAGEMENT (ON/OFF)")
    print("-" * 80)
    print(result.commitment_schedule)

    print("\n" + "-" * 80)
    print("DÉMARRAGES")
    print("-" * 80)
    print(result.startup_schedule)

    print("\n" + "-" * 80)
    print("VÉRIFICATION ÉQUILIBRE")
    print("-" * 80)
    total_production = result.production_schedule.sum(axis=1)
    shed = result.load_shedding
    for hour in range(len(demand)):
        prod = total_production.iloc[hour]
        shed_val = shed.iloc[hour]
        dem = demand.iloc[hour]
        coverage = prod + shed_val - dem
        print(
            f"H{hour:02d}: Prod={prod:8.2f} | Délestage={shed_val:6.2f} | "
            f"Demande={dem:8.2f} | Écart={coverage:7.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
