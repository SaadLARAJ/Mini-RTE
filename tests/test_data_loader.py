"""
Tests unitaires pour le chargement de données.
"""

import unittest
from pathlib import Path
import sys
import tempfile
import csv

# Ajouter le répertoire src au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd

from mini_rte.data_loader import load_demand, load_availability, load_all_availability


class TestDataLoader(unittest.TestCase):
    """Tests pour le chargement de données."""

    def setUp(self):
        """Configuration initiale pour les tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_load_demand_valid(self):
        """Test le chargement d'une demande valide."""
        # Créer un fichier CSV de test
        csv_path = self.temp_dir / "test_demand.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "demand_mw"])
            for hour in range(24):
                writer.writerow([hour, 4000 + hour * 100])

        demand = load_demand(csv_path)
        self.assertEqual(len(demand), 24)
        self.assertTrue((demand >= 0).all())

    def test_load_demand_missing_file(self):
        """Test le chargement avec fichier manquant."""
        missing_path = self.temp_dir / "missing.csv"
        with self.assertRaises(FileNotFoundError):
            load_demand(missing_path)

    def test_load_demand_negative_values(self):
        """Test la validation des valeurs négatives."""
        csv_path = self.temp_dir / "negative_demand.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "demand_mw"])
            writer.writerow([0, -100])  # Valeur négative

        with self.assertRaises(ValueError):
            load_demand(csv_path)

    def test_load_availability_valid(self):
        """Test le chargement d'une disponibilité valide."""
        csv_path = self.temp_dir / "test_availability.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "Wind_1"])
            for hour in range(24):
                writer.writerow([hour, 0.5])

        availability = load_availability(csv_path, plant_names=["Wind_1"])
        self.assertEqual(len(availability), 24)
        self.assertEqual(len(availability.columns), 1)
        self.assertTrue((availability >= 0).all().all())
        self.assertTrue((availability <= 1).all().all())

    def test_load_availability_clipping(self):
        """Test que les valeurs > 1 sont limitées à 1."""
        csv_path = self.temp_dir / "high_availability.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "Wind_1"])
            writer.writerow([0, 1.5])  # Valeur > 1

        availability = load_availability(csv_path, plant_names=["Wind_1"])
        self.assertTrue((availability <= 1).all().all())


if __name__ == "__main__":
    unittest.main()
