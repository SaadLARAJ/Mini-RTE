"""
Tests unitaires pour le modèle d'Unit Commitment.
"""

import unittest
from pathlib import Path
import sys

# Ajouter le répertoire src au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd

from mini_rte.models.plant import PowerPlant
from mini_rte.models.unit_commitment import UnitCommitmentModel
from mini_rte.solver import SolverManager


class TestPowerPlant(unittest.TestCase):
    """Tests pour la classe PowerPlant."""

    def test_plant_creation(self):
        """Test la création d'une centrale."""
        plant = PowerPlant(
            name="Test_Plant",
            plant_type="gas",
            p_min=100,
            p_max=500,
            cost_variable=50,
            cost_fixed=1000,
            cost_startup=5000,
            ramp_rate=200,
            is_renewable=False,
        )
        self.assertEqual(plant.name, "Test_Plant")
        self.assertEqual(plant.p_max, 500)

    def test_plant_validation(self):
        """Test la validation des paramètres."""
        # p_max doit être > p_min
        with self.assertRaises(ValueError):
            PowerPlant(
                name="Invalid",
                plant_type="gas",
                p_min=500,
                p_max=100,  # Invalide
                cost_variable=50,
                cost_fixed=1000,
                cost_startup=5000,
                ramp_rate=200,
            )


class TestUnitCommitmentModel(unittest.TestCase):
    """Tests pour le modèle d'Unit Commitment."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        # Créer des centrales de test
        self.plants = [
            PowerPlant(
                name="Gas_1",
                plant_type="gas",
                p_min=100,
                p_max=500,
                cost_variable=50,
                cost_fixed=1000,
                cost_startup=5000,
                ramp_rate=200,
                emission_factor=0.35,
            ),
            PowerPlant(
                name="Nuclear_1",
                plant_type="nuclear",
                p_min=800,
                p_max=1200,
                cost_variable=20,
                cost_fixed=5000,
                cost_startup=50000,
                ramp_rate=100,
                emission_factor=0.006,
            ),
        ]

        # Créer une demande de test (24 heures)
        self.demand = pd.Series(
            [4000, 3800, 3600, 3500, 3400, 3300, 3500, 4000, 4500, 5000, 5200, 5300,
             5400, 5500, 5600, 5550, 5500, 5600, 5800, 6000, 5800, 5300, 4800, 4300],
            index=range(24),
            name="demand_mw",
        )

    def test_model_creation(self):
        """Test la création du modèle."""
        model = UnitCommitmentModel(plants=self.plants, demand=self.demand)
        self.assertEqual(len(model.plants), 2)
        self.assertEqual(len(model.demand), 24)

    def test_model_validation(self):
        """Test la validation des entrées."""
        # Demande vide
        with self.assertRaises(ValueError):
            UnitCommitmentModel(plants=self.plants, demand=pd.Series(dtype=float))

        # Pas de centrales
        with self.assertRaises(ValueError):
            UnitCommitmentModel(plants=[], demand=self.demand)

        # Demande négative
        negative_demand = pd.Series([-100, 200], index=[0, 1])
        with self.assertRaises(ValueError):
            UnitCommitmentModel(plants=self.plants, demand=negative_demand)

    def test_model_build(self):
        """Test la construction du modèle Pyomo."""
        model = UnitCommitmentModel(plants=self.plants, demand=self.demand)
        pyomo_model = model.build_model()

        # Vérifier que le modèle a été créé
        self.assertIsNotNone(pyomo_model)
        self.assertIsNotNone(pyomo_model.objective)

        # Vérifier les variables
        self.assertIn("u", pyomo_model.component_map())
        self.assertIn("p", pyomo_model.component_map())
        self.assertIn("start", pyomo_model.component_map())

        # Vérifier les contraintes
        self.assertIn("demand_balance", pyomo_model.component_map())
        self.assertIn("power_min", pyomo_model.component_map())
        self.assertIn("power_max", pyomo_model.component_map())

    def test_renewable_constraint(self):
        """Test la contrainte sur les énergies renouvelables."""
        # Ajouter une centrale renouvelable
        wind_plant = PowerPlant(
            name="Wind_1",
            plant_type="wind",
            p_min=0,
            p_max=300,
            cost_variable=5,
            cost_fixed=500,
            cost_startup=0,
            ramp_rate=300,
            is_renewable=True,
        )

        plants_with_wind = self.plants + [wind_plant]

        # Créer une disponibilité
        availability = pd.DataFrame({
            "Wind_1": [0.5] * 24
        }, index=range(24))

        model = UnitCommitmentModel(
            plants=plants_with_wind,
            demand=self.demand,
            availability=availability,
        )

        pyomo_model = model.build_model()
        # Vérifier que la contrainte renouvelable existe
        self.assertIn("renewable_limit", pyomo_model.component_map())

    def _get_solver(self) -> str:
        available = SolverManager.list_available_solvers()
        if not available:
            self.skipTest("Aucun solveur MILP disponible")
        return available[0]

    def test_never_infeasible(self):
        """Le modèle trouve toujours une solution grâce au load shedding."""
        solver_name = self._get_solver()
        demand = pd.Series([100_000] * 24)
        model = UnitCommitmentModel(self.plants, demand, co2_price=80)
        model.build_model()
        result = model.solve(solver_name=solver_name)
        self.assertTrue(result.success)
        self.assertGreater(result.total_load_shedding, 0)

    def test_co2_reduces_gas(self):
        """Un prix CO2 élevé doit réduire la production gaz."""
        solver_name = self._get_solver()
        demand = pd.Series([1500] * 24)
        plants = [
            PowerPlant(
                name="Nuke",
                plant_type="nuclear",
                p_min=500,
                p_max=1000,
                cost_variable=10,
                cost_fixed=0,
                cost_startup=0,
                ramp_rate=500,
                emission_factor=0.006,
            ),
            PowerPlant(
                name="Hydro",
                plant_type="hydro",
                p_min=0,
                p_max=800,
                cost_variable=40,
                cost_fixed=0,
                cost_startup=0,
                ramp_rate=800,
                emission_factor=0.006,
            ),
            PowerPlant(
                name="Gas",
                plant_type="gas",
                p_min=0,
                p_max=1000,
                cost_variable=30,
                cost_fixed=0,
                cost_startup=0,
                ramp_rate=1000,
                emission_factor=0.35,
            ),
        ]

        model_low = UnitCommitmentModel(plants, demand, co2_price=0)
        model_low.build_model()
        result_low = model_low.solve(solver_name=solver_name)

        model_high = UnitCommitmentModel(plants, demand, co2_price=150)
        model_high.build_model()
        result_high = model_high.solve(solver_name=solver_name)

        gas_low = result_low.production_schedule["Gas"].sum()
        gas_high = result_high.production_schedule["Gas"].sum()
        self.assertGreater(gas_low, gas_high)

    def test_marginal_prices_positive(self):
        """Les prix marginaux doivent être >= 0."""
        solver_name = self._get_solver()
        model = UnitCommitmentModel(self.plants, self.demand)
        model.build_model()
        result = model.solve(solver_name=solver_name)
        self.assertTrue((result.marginal_prices >= -1e-6).all())


if __name__ == "__main__":
    unittest.main()
