"""
Interface abstraite pour les solveurs MILP (factory pattern).
"""

import logging
from typing import Optional

from pyomo.environ import SolverFactory
from pyomo.opt import SolverResults

logger = logging.getLogger(__name__)


class SolverManager:
    """Gestionnaire de solveurs avec factory pattern."""

    SUPPORTED_SOLVERS = ["cbc", "glpk", "highs", "appsi_highs", "gurobi"]

    @staticmethod
    def create_solver(solver_name: str) -> Optional[object]:
        """
        Crée une instance du solveur demandé.

        Args:
            solver_name: Nom du solveur ("cbc", "glpk", "highs", "gurobi")

        Returns:
            Instance du solveur Pyomo, ou None si non disponible

        Raises:
            ValueError: Si le solveur n'est pas supporté
        """
        if solver_name not in SolverManager.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Solveur '{solver_name}' non supporté. "
                f"Solveurs disponibles: {SolverManager.SUPPORTED_SOLVERS}"
            )

        solver = SolverFactory(solver_name)

        if solver is None:
            logger.warning(f"Solveur '{solver_name}' non disponible sur ce système")
            return None

        if not solver.available():
            logger.warning(f"Solveur '{solver_name}' installé mais non disponible")
            return None

        logger.info(f"Solveur '{solver_name}' créé avec succès")
        return solver

    @staticmethod
    def configure_solver(solver: object, solver_name: str, timeout: int = 300, mip_gap: float = 0.01) -> None:
        """
        Configure les options du solveur.

        Args:
            solver: Instance du solveur Pyomo
            solver_name: Nom du solveur
            timeout: Timeout en secondes
            mip_gap: Gap d'optimalité accepté (0.01 = 1%)
        """
        if solver_name == "cbc":
            solver.options["seconds"] = timeout
            solver.options["ratioGap"] = mip_gap
            solver.options["threads"] = 0  # Auto
        elif solver_name == "glpk":
            solver.options["tmlim"] = timeout * 1000  # GLPK utilise des millisecondes
            solver.options["mipgap"] = mip_gap
        elif solver_name in {"highs", "appsi_highs"}:
            solver.options["time_limit"] = timeout
            solver.options["mip_rel_gap"] = mip_gap
        elif solver_name == "gurobi":
            solver.options["TimeLimit"] = timeout
            solver.options["MIPGap"] = mip_gap

        logger.debug(f"Options du solveur {solver_name} configurées: timeout={timeout}s, mip_gap={mip_gap}")

    @staticmethod
    def solve_model(
        model: object,
        solver_name: str = "cbc",
        timeout: int = 300,
        mip_gap: float = 0.01,
        tee: bool = False,
    ) -> SolverResults:
        """
        Résout un modèle Pyomo avec le solveur spécifié.

        Args:
            model: Modèle Pyomo à résoudre
            solver_name: Nom du solveur
            timeout: Timeout en secondes
            mip_gap: Gap d'optimalité accepté
            tee: Afficher les logs du solveur (True) ou non (False)

        Returns:
            Résultat du solveur

        Raises:
            ValueError: Si le solveur n'est pas disponible
        """
        solver = SolverManager.create_solver(solver_name)

        if solver is None:
            raise ValueError(
                f"Solveur '{solver_name}' non disponible. "
                "Vérifiez l'installation du solveur."
            )

        SolverManager.configure_solver(solver, solver_name, timeout, mip_gap)

        logger.info(f"Résolution du modèle avec {solver_name}...")
        results = solver.solve(model, tee=tee)

        return results

    @staticmethod
    def check_solver_availability(solver_name: str) -> bool:
        """
        Vérifie si un solveur est disponible.

        Args:
            solver_name: Nom du solveur

        Returns:
            True si le solveur est disponible, False sinon
        """
        try:
            solver = SolverManager.create_solver(solver_name)
            return solver is not None and solver.available()
        except Exception:
            return False

    @staticmethod
    def list_available_solvers() -> list[str]:
        """
        Liste tous les solveurs disponibles sur le système.

        Returns:
            Liste des noms de solveurs disponibles
        """
        available = []
        for solver_name in SolverManager.SUPPORTED_SOLVERS:
            if SolverManager.check_solver_availability(solver_name):
                available.append(solver_name)
        return available
