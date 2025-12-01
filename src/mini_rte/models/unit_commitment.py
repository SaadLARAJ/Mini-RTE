"""Modèle Pyomo pour le problème d'Unit Commitment (MILP)."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Set,
    Suffix,
    Var,
    minimize,
    value,
)

from .plant import PowerPlant

logger = logging.getLogger(__name__)

VOLL_DEFAULT = 20_000
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
class OptimizationResult:
    """Résultat de l'optimisation UC avec indicateurs enrichis."""

    success: bool
    objective_value: float
    production_schedule: pd.DataFrame
    startup_schedule: pd.DataFrame
    cost_breakdown: Dict[str, float]
    load_shedding: pd.Series
    total_load_shedding: float
    curtailment: pd.DataFrame
    total_curtailment: float
    marginal_prices: pd.Series
    co2_emissions: float
    termination_condition: str = ""
    commitment_schedule: Optional[pd.DataFrame] = None
    status: str = ""


class UnitCommitmentModel:
    """Modèle MILP pour décider du dispatch optimal heure par heure."""

    def __init__(
        self,
        plants: List[PowerPlant],
        demand: pd.Series,
        availability: Optional[pd.DataFrame] = None,
        co2_price: float = 80.0,
        voll: float = VOLL_DEFAULT,
        reserve_margin: float = 0.0,
    ) -> None:
        self.plants = plants
        self.demand = demand
        self.availability = availability if availability is not None else pd.DataFrame()
        self.model: Optional[ConcreteModel] = None
        self.solver_result = None
        self.co2_price = co2_price
        self.voll = voll
        self.reserve_margin = reserve_margin

        self._validate_inputs()
        logger.info(
            "Modèle initialisé avec %s centrales, %s heures, CO2=%s €/t, VoLL=%s €/MWh",
            len(plants),
            len(demand),
            co2_price,
            voll,
        )

    def _validate_inputs(self) -> None:
        if len(self.plants) == 0:
            raise ValueError("Au moins une centrale est requise")
        if len(self.demand) == 0:
            raise ValueError("La demande ne peut pas être vide")
        if self.demand.min() < 0:
            raise ValueError("La demande ne peut pas être négative")
        if self.voll <= 0:
            raise ValueError("La valeur de VoLL doit être positive")
        if self.co2_price < 0:
            raise ValueError("Le prix du CO2 doit être positif")
        if self.reserve_margin < 0:
            raise ValueError("La marge de réserve doit être positive ou nulle")

        renewable_plants = [p for p in self.plants if p.is_renewable]
        if renewable_plants and not self.availability.empty:
            renewable_names = [p.name for p in renewable_plants]
            missing = set(renewable_names) - set(self.availability.columns)
            if missing:
                logger.warning(
                    "Pas de disponibilité pour %s, elles seront traitées comme pilotables",
                    missing,
                )

    def build_model(self) -> ConcreteModel:
        """Construit le modèle Pyomo avec contraintes UC complètes."""
        model = ConcreteModel()
        self.model = model
        model.dual = Suffix(direction=Suffix.IMPORT)

        hours = list(range(len(self.demand)))
        plant_ids = list(range(len(self.plants)))

        model.I = Set(initialize=plant_ids, doc="Ensemble des centrales")
        model.T = Set(initialize=hours, doc="Ensemble des heures")
        model.I_renew = Set(
            initialize=[i for i, p in enumerate(self.plants) if p.is_renewable],
            doc="Centrales renouvelables",
        )

        model.u = Var(model.I, model.T, domain=Binary, doc="État on/off de la centrale")
        model.p = Var(
            model.I, model.T, domain=NonNegativeReals, doc="Puissance produite (MW)"
        )
        model.start = Var(model.I, model.T, domain=Binary, doc="Indicateur démarrage")
        model.shed = Var(model.T, within=NonNegativeReals, doc="Délestage (MW)")
        model.curtail = Var(
            model.I_renew, model.T, within=NonNegativeReals, doc="Écrêtement (MW)"
        )

        def objective_rule(m):
            cost_var = sum(
                (self.plants[i].cost_variable + self.plants[i].emission_factor * self.co2_price)
                * m.p[i, t]
                for i in m.I
                for t in m.T
            )
            cost_fix = sum(
                self.plants[i].cost_fixed * m.u[i, t] for i in m.I for t in m.T
            )
            cost_start = sum(
                self.plants[i].cost_startup * m.start[i, t] for i in m.I for t in m.T
            )
            shed_cost = sum(self.voll * m.shed[t] for t in m.T)
            return cost_var + cost_fix + cost_start + shed_cost

        model.objective = Objective(rule=objective_rule, sense=minimize)

        def demand_balance_rule(m, t):
            return (
                sum(m.p[i, t] for i in m.I) + m.shed[t]
                >= self.demand.iloc[t] * (1 + self.reserve_margin)
            )

        model.demand_balance = Constraint(model.T, rule=demand_balance_rule)

        def power_min_rule(m, i, t):
            return self.plants[i].p_min * m.u[i, t] <= m.p[i, t]

        def power_max_rule(m, i, t):
            return m.p[i, t] <= self.plants[i].p_max * m.u[i, t]

        model.power_min = Constraint(model.I, model.T, rule=power_min_rule)
        model.power_max = Constraint(model.I, model.T, rule=power_max_rule)

        def ramp_up_rule(m, i, t):
            if t == 0:
                return Constraint.Skip
            return m.p[i, t] - m.p[i, t - 1] <= self.plants[i].ramp_rate

        def ramp_down_rule(m, i, t):
            if t == 0:
                return Constraint.Skip
            return m.p[i, t - 1] - m.p[i, t] <= self.plants[i].ramp_rate

        model.ramp_up = Constraint(model.I, model.T, rule=ramp_up_rule)
        model.ramp_down = Constraint(model.I, model.T, rule=ramp_down_rule)

        def startup_1(m, i, t):
            if t == 0:
                return m.start[i, t] >= m.u[i, t]
            return m.start[i, t] >= m.u[i, t] - m.u[i, t - 1]

        def startup_2(m, i, t):
            return m.start[i, t] <= m.u[i, t]

        def startup_3(m, i, t):
            if t == 0:
                return Constraint.Skip
            return m.start[i, t] <= 1 - m.u[i, t - 1]

        model.startup_1 = Constraint(model.I, model.T, rule=startup_1)
        model.startup_2 = Constraint(model.I, model.T, rule=startup_2)
        model.startup_3 = Constraint(model.I, model.T, rule=startup_3)

        def renewable_balance(m, i, t):
            avail = 1.0
            plant = self.plants[i]
            if not self.availability.empty and plant.name in self.availability.columns:
                avail = float(self.availability.loc[t, plant.name])
            max_available = avail * plant.p_max
            return m.p[i, t] + m.curtail[i, t] <= max_available

        model.renewable_limit = Constraint(model.I_renew, model.T, rule=renewable_balance)

        logger.info("Modèle Pyomo construit avec succès")
        return model

    def solve(
        self, solver_name: str = "cbc", timeout: int = 300, mip_gap: float = 0.01
    ) -> OptimizationResult:
        """Résout le modèle et extrait les indicateurs principaux."""
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant d'être résolu")

        from pyomo.opt import SolverFactory

        solver = SolverFactory(solver_name)
        if solver is None or not solver.available():
            raise ValueError(
                f"Solveur '{solver_name}' non disponible. "
                f"Vérifiez son installation."
            )

        if solver_name == "cbc":
            solver.options["seconds"] = timeout
            solver.options["ratioGap"] = mip_gap
        elif solver_name == "glpk":
            solver.options["tmlim"] = timeout
            solver.options["mipgap"] = mip_gap
        elif solver_name == "highs" or solver_name == "appsi_highs":
            solver.options["time_limit"] = timeout
            solver.options["mip_rel_gap"] = mip_gap

        logger.info("Résolution avec %s...", solver_name)
        self.solver_result = solver.solve(self.model, tee=False)

        status = str(self.solver_result.solver.status)
        termination = str(self.solver_result.solver.termination_condition)
        term_lower = termination.lower()
        success = "optimal" in term_lower or "feasible" in term_lower
        objective_value = value(self.model.objective) if success else np.nan

        logger.info("Statut: %s / %s", status, termination)

        if not success:
            empty_hours = list(range(len(self.demand)))
            plant_names = [p.name for p in self.plants]
            empty_prod = pd.DataFrame(0.0, index=empty_hours, columns=plant_names)
            empty_commit = pd.DataFrame(0, index=empty_hours, columns=plant_names, dtype=int)
            empty_series = pd.Series(0.0, index=empty_hours)
            return OptimizationResult(
                success=False,
                objective_value=objective_value,
                production_schedule=empty_prod,
                startup_schedule=empty_commit.copy(),
                cost_breakdown={},
                load_shedding=empty_series,
                total_load_shedding=0.0,
                curtailment=pd.DataFrame(0.0, index=empty_hours, columns=[]),
                total_curtailment=0.0,
                marginal_prices=empty_series,
                co2_emissions=0.0,
                termination_condition=termination,
                commitment_schedule=empty_commit.copy(),
                status=f"{status} - {termination}",
            )

        production_schedule = self.get_production_schedule()
        commitment_schedule = self.get_commitment_schedule()
        startup_schedule = self.get_startup_schedule()
        load_shedding = self.get_load_shedding()
        curtailment = self.get_curtailment()
        
        # Tenter d'obtenir les prix marginaux via les duals (méthode la plus précise)
        # Si indisponible (cas fréquent avec les solveurs MILP gratuits), basculer sur une estimation.
        try:
            dual_prices = self.get_marginal_prices()
            if dual_prices.notna().any() and dual_prices.abs().sum() > 0:
                marginal_prices = dual_prices
            else: # Les duals sont nuls ou vides, on estime
                marginal_prices = self.estimate_marginal_prices_from_dispatch(production_schedule)
        except Exception: # Erreur lors de l'accès aux duals
            marginal_prices = self.estimate_marginal_prices_from_dispatch(production_schedule)

        co2_emissions = self.compute_co2_emissions(production_schedule)
        cost_breakdown = self.get_cost_breakdown(load_shedding)

        return OptimizationResult(
            success=True,
            objective_value=objective_value,
            production_schedule=production_schedule,
            startup_schedule=startup_schedule,
            cost_breakdown=cost_breakdown,
            load_shedding=load_shedding,
            total_load_shedding=float(load_shedding.sum()),
            curtailment=curtailment,
            total_curtailment=float(curtailment.sum().sum()) if not curtailment.empty else 0.0,
            marginal_prices=marginal_prices,
            co2_emissions=co2_emissions,
            termination_condition=termination,
            commitment_schedule=commitment_schedule,
            status=f"{status} - {termination}",
        )

    def get_production_schedule(self) -> pd.DataFrame:
        """Retourne la production par centrale et par heure."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")

        hours = list(range(len(self.demand)))
        plant_names = [p.name for p in self.plants]
        schedule = pd.DataFrame(index=hours, columns=plant_names, dtype=float)

        for i, plant in enumerate(self.plants):
            for t in hours:
                val = value(self.model.p[i, t])
                schedule.loc[t, plant.name] = val if val is not None else 0.0

        return schedule

    def get_commitment_schedule(self) -> pd.DataFrame:
        """Retourne l'état ON/OFF par centrale et par heure."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")

        hours = list(range(len(self.demand)))
        plant_names = [p.name for p in self.plants]
        schedule = pd.DataFrame(index=hours, columns=plant_names, dtype=int)

        for i, plant in enumerate(self.plants):
            for t in hours:
                val = value(self.model.u[i, t])
                schedule.loc[t, plant.name] = int(val) if val is not None else 0
        return schedule

    def get_startup_schedule(self) -> pd.DataFrame:
        """Retourne les démarrages par centrale et par heure."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")

        hours = list(range(len(self.demand)))
        plant_names = [p.name for p in self.plants]
        schedule = pd.DataFrame(index=hours, columns=plant_names, dtype=int)

        for i, plant in enumerate(self.plants):
            for t in hours:
                val = value(self.model.start[i, t])
                schedule.loc[t, plant.name] = int(val) if val is not None else 0
        return schedule

    def get_load_shedding(self) -> pd.Series:
        """Retourne le délestage par heure."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")
        hours = list(range(len(self.demand)))
        return pd.Series(
            [float(value(self.model.shed[t]) or 0.0) for t in hours], index=hours, name="shed_mw"
        )

    def get_curtailment(self) -> pd.DataFrame:
        """Retourne l'écrêtement par centrale renouvelable et heure."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")
        hours = list(range(len(self.demand)))
        renew_indices = list(self.model.I_renew.data())
        if not renew_indices:
            return pd.DataFrame(0.0, index=hours, columns=[])

        columns = [self.plants[i].name for i in renew_indices]
        curtail = pd.DataFrame(0.0, index=hours, columns=columns)
        for i in renew_indices:
            for t in hours:
                val = value(self.model.curtail[i, t])
                curtail.loc[t, self.plants[i].name] = val if val is not None else 0.0
        return curtail

    def get_marginal_prices(self) -> pd.Series:
        """Extrait le prix marginal (shadow price) de l'équilibre offre/demande."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")
        prices: Dict[int, float] = {}
        for t in self.model.T:
            try:
                dual_val = self.model.dual[self.model.demand_balance[t]]
                prices[int(t)] = abs(float(dual_val)) if dual_val is not None else 0.0
            except Exception:
                prices[int(t)] = 0.0
        return pd.Series(prices).sort_index()

    def estimate_marginal_prices_from_dispatch(self, production: pd.DataFrame) -> pd.Series:
        """Estimation de prix marginaux via le coût marginal de la dernière centrale appelée."""
        prices = {}
        costs = {
            p.name: p.cost_variable + p.emission_factor * self.co2_price for p in self.plants
        }
        for t in production.index:
            running = production.loc[t]
            active = running[running > 1e-3]
            if active.empty:
                prices[t] = 0.0
            else:
                last_cost = max(costs[name] for name in active.index)
                prices[t] = last_cost
        return pd.Series(prices).sort_index()

    def compute_co2_emissions(self, production: pd.DataFrame) -> float:
        """Calcule les émissions totales de CO2 (t)."""
        emissions = 0.0
        for plant in self.plants:
            if plant.name in production.columns:
                emissions += float(production[plant.name].sum()) * plant.emission_factor
        return emissions

    def get_cost_breakdown(self, load_shedding: Optional[pd.Series] = None) -> Dict[str, float]:
        """Décompose les coûts totaux (variable, CO2, fixes, démarrage, délestage)."""
        if self.model is None:
            raise ValueError("Le modèle doit être résolu avant d'extraire les résultats")

        cost_var = 0.0
        cost_co2 = 0.0
        cost_fix = 0.0
        cost_start = 0.0

        for i, plant in enumerate(self.plants):
            for t in range(len(self.demand)):
                p_val = value(self.model.p[i, t]) or 0.0
                cost_var += plant.cost_variable * p_val
                cost_co2 += plant.emission_factor * self.co2_price * p_val
                cost_fix += plant.cost_fixed * (value(self.model.u[i, t]) or 0)
                cost_start += plant.cost_startup * (value(self.model.start[i, t]) or 0)

        cost_shed = float(load_shedding.sum() * self.voll) if load_shedding is not None else 0.0

        return {
            "variable": cost_var,
            "co2": cost_co2,
            "fixed": cost_fix,
            "startup": cost_start,
            "shed": cost_shed,
            "total": cost_var + cost_co2 + cost_fix + cost_start + cost_shed,
        }
