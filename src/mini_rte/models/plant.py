"""
Définition des centrales électriques et leurs paramètres.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PowerPlant:
    """
    Représente une centrale électrique avec ses caractéristiques techniques et économiques.

    Attributes:
        name: Nom de la centrale
        plant_type: Type de centrale ("nuclear", "gas", "wind", "solar", "hydro")
        p_min: Puissance minimale de production (MW)
        p_max: Puissance maximale de production (MW)
        cost_variable: Coût variable de production (€/MWh)
        cost_fixed: Coût fixe horaire si la centrale est allumée (€/h)
        cost_startup: Coût de démarrage (€ par démarrage)
        ramp_rate: Rampe maximale de variation de puissance (MW/h)
        is_renewable: Indique si la centrale utilise une énergie renouvelable
        emission_factor: Facteur d'émission tCO2/MWh
    """

    name: str
    plant_type: Literal["nuclear", "gas", "wind", "solar", "hydro", "oil", "coal"]
    p_min: float
    p_max: float
    cost_variable: float  # €/MWh
    cost_fixed: float  # €/h si allumée
    cost_startup: float  # € par démarrage
    ramp_rate: float  # MW/h max
    is_renewable: bool = False
    emission_factor: float = 0.0

    def __post_init__(self) -> None:
        """
        Valide les paramètres de la centrale après initialisation.
        
        Raises:
            ValueError: Si les paramètres sont incohérents
        """
        if self.p_min < 0:
            raise ValueError(f"p_min doit être positif pour {self.name}")
        if self.p_max <= self.p_min:
            raise ValueError(
                f"p_max ({self.p_max}) doit être strictement supérieur à p_min ({self.p_min}) pour {self.name}"
            )
        if self.cost_variable < 0:
            raise ValueError(f"cost_variable doit être positif pour {self.name}")
        if self.cost_fixed < 0:
            raise ValueError(f"cost_fixed doit être positif pour {self.name}")
        if self.cost_startup < 0:
            raise ValueError(f"cost_startup doit être positif pour {self.name}")
        if self.ramp_rate < 0:
            raise ValueError(f"ramp_rate doit être positif pour {self.name}")
        if self.emission_factor < 0:
            raise ValueError(f"emission_factor doit être positif pour {self.name}")

    def __repr__(self) -> str:
        """Représentation lisible de la centrale."""
        return (
            f"PowerPlant(name='{self.name}', type='{self.plant_type}', "
            f"p=[{self.p_min}, {self.p_max}] MW, "
            f"cost_var={self.cost_variable} €/MWh)"
        )
