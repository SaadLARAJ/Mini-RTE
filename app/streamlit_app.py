"""
Application Streamlit interactive pour l'optimisation du réseau électrique.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from mini_rte import ConfigLoader, forecast_demand
from mini_rte.data_loader import load_all_availability, load_demand
from mini_rte.models.unit_commitment import UnitCommitmentModel
from mini_rte.scenarios import PREDEFINED_SCENARIOS, ScenarioGenerator, get_predefined_scenario
from mini_rte.solver import SolverManager
from mini_rte.visualization import (
    plot_commitment_schedule,
    plot_cost_breakdown,
    plot_forecast_vs_actual,
    plot_plant_clusters,
    plot_production_detailed,
    plot_production_mix,
)


def plot_merit_order(plants, co2_price: float) -> go.Figure:
    """Merit order dynamique intégrant le prix CO2."""
    data = []
    for p in plants:
        cost_co2 = p.emission_factor * co2_price
        cost = p.cost_variable + cost_co2
        data.append(
            {
                "name": p.name,
                "type": p.plant_type,
                "cost": cost,
                "capacity": p.p_max,
                "cost_fuel": p.cost_variable,
                "cost_co2": cost_co2,
            }
        )

    data.sort(key=lambda x: x["cost"])
    colors = {
        "nuclear": "#FF6B6B",
        "hydro": "#4ECDC4",
        "gas": "#FFE66D",
        "oil": "#FF8C42",
        "wind": "#95E1D3",
        "solar": "#F9ED69",
    }

    fig = go.Figure()
    cumul = 0
    for d in data:
        fig.add_trace(
            go.Bar(
                x=[cumul + d["capacity"] / 2],
                y=[d["cost"]],
                width=d["capacity"],
                name=d["name"],
                marker_color=colors.get(d["type"], "#888"),
                hovertemplate=(
                    f"<b>{d['name']}</b><br>"
                    f"Coût: {d['cost']:.0f} €/MWh<br>"
                    f"Combustible: {d['cost_fuel']:.0f} €/MWh<br>"
                    f"CO2: {d['cost_co2']:.1f} €/MWh<br>"
                    f"Capacité: {d['capacity']} MW"
                ),
            )
        )
        cumul += d["capacity"]

    fig.update_layout(
        title=f"📊 Merit Order (CO2 = {co2_price} €/t)",
        xaxis_title="Capacité cumulée (MW)",
        yaxis_title="Coût marginal (€/MWh)",
        showlegend=True,
    )
    return fig


def plot_co2_emissions(result, plants) -> go.Figure:
    """Répartition des émissions par filière."""
    emissions: dict[str, float] = {}
    for p in plants:
        prod = result.production_schedule[p.name].sum()
        emissions[p.plant_type] = emissions.get(p.plant_type, 0.0) + prod * p.emission_factor

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(emissions.keys()),
                values=list(emissions.values()),
                hole=0.4,
            )
        ]
    )
    fig.update_layout(title="🌍 Émissions CO2 par filière")
    return fig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="mini-rte - Optimiseur de Réseau Électrique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ mini-rte - Optimiseur de Réseau Électrique")
st.markdown(
    """
    **Optimiseur MILP pour la planification de production électrique (Unit Commitment)**
    
    Décidez quelles centrales allumer/éteindre heure par heure pour satisfaire la demande au moindre coût.
    """
)

st.sidebar.header("⚙️ Configuration")

config_path = project_root / "config" / "default_config.yaml"
data_dir = project_root / "data"

try:
    config = ConfigLoader.load_yaml(config_path)
    ConfigLoader.validate_config(config)
    plants = ConfigLoader.load_plants(config)
    solver_config = ConfigLoader.load_solver_config(config)
    global_params = ConfigLoader.load_global_params(config)
except Exception as e:
    st.error(f"Erreur chargement config: {e}")
    st.stop()

available_solvers = SolverManager.list_available_solvers()
if not available_solvers:
    st.error("Aucun solveur disponible (CBC, GLPK, HiGHS).")
    st.stop()

solver_name = st.sidebar.selectbox(
    "Solveur",
    options=available_solvers,
    index=0 if "cbc" in available_solvers else 0,
)

timeout = st.sidebar.slider("Timeout (s)", 60, 600, solver_config.timeout, 30)
mip_gap = st.sidebar.slider("Gap optimalité (%)", 0.1, 5.0, solver_config.mip_gap * 100, 0.1) / 100

st.sidebar.header("📊 Scénarios")

scenario_mode = st.sidebar.radio(
    "Mode",
    ["Configuration de base", "Scénario prédéfini", "Paramètres personnalisés"],
)

selected_scenario = None
custom_modifications = {}
scenario_mods: dict = {}

if scenario_mode == "Scénario prédéfini":
    scenario_names = list(PREDEFINED_SCENARIOS.keys())
    selected_scenario = st.sidebar.selectbox("Scénario", scenario_names)
    if selected_scenario:
        scenario_info = PREDEFINED_SCENARIOS[selected_scenario]
        st.sidebar.info(f"**{selected_scenario}**: {scenario_info['description']}")
        scenario_mods = {k: v for k, v in scenario_info.items() if k != "description"}

elif scenario_mode == "Paramètres personnalisés":
    st.sidebar.subheader("Modifications")
    custom_modifications["wind_availability_factor"] = st.sidebar.slider(
        "Facteur disponibilité éolienne", 0.0, 2.0, 1.0, 0.1
    )
    custom_modifications["solar_availability_factor"] = st.sidebar.slider(
        "Facteur disponibilité solaire", 0.0, 2.0, 1.0, 0.1
    )
    custom_modifications["gas_cost_multiplier"] = st.sidebar.slider(
        "Multiplicateur coût gaz", 0.5, 3.0, 1.0, 0.1
    )
    custom_modifications["nuclear_availability"] = st.sidebar.slider(
        "Disponibilité nucléaire", 0.0, 1.0, 1.0, 0.1
    )
    scenario_mods = custom_modifications

st.sidebar.markdown("---")
st.sidebar.subheader("💨 Marché carbone")
co2_price = st.sidebar.slider(
    "Prix CO2 (€/tonne)",
    min_value=0,
    max_value=200,
    value=int(global_params.co2_price),
    step=5,
    help="Prix du marché carbone européen (ETS)",
)
gas_factor = next((p.emission_factor for p in plants if p.plant_type == "gas"), 0.35)
base_gas_cost = next((p.cost_variable for p in plants if p.plant_type == "gas"), 50)
gas_effective = base_gas_cost + gas_factor * co2_price
st.sidebar.caption(f"→ Coût gaz avec CO2: {gas_effective:.0f} €/MWh")

st.sidebar.markdown("---")
use_forecast = st.sidebar.checkbox(
    "Prévoir la demande (ML)",
    value=False,
    help="Utilise RandomForest si disponible, sinon régression polynomiale.",
)

if st.sidebar.button("🚀 Optimiser", type="primary", use_container_width=True):
    st.session_state["optimize"] = True
else:
    st.session_state["optimize"] = False

if st.session_state.get("optimize", False):
    demand_forecast_plot = None
    with st.spinner("Chargement des données..."):
        demand_path = data_dir / "demand_profile_24h.csv"
        if scenario_mods.get("demand_profile"):
            demand_path = data_dir / scenario_mods["demand_profile"]

        try:
            demand = load_demand(demand_path)
        except Exception as e:
            st.error(f"Erreur chargement demande: {e}")
            st.stop()

        renewable_names = [p.name for p in plants if p.is_renewable]
        availability = pd.DataFrame()
        if renewable_names:
            availability = load_all_availability(data_dir, renewable_names)

        modified_plants = plants
        modified_availability = availability
        co2_price_active = co2_price

        if scenario_mods:
            modified_plants = ScenarioGenerator.apply_scenario_modifications(
                plants, scenario_mods
            )
            modified_availability = ScenarioGenerator.modify_availability(
                availability, scenario_mods
            )
            co2_price_active = scenario_mods.get("co2_price", co2_price_active)
            if scenario_mods.get("demand_profile"):
                st.sidebar.info(f"Profil de demande: {scenario_mods['demand_profile']}")

        if use_forecast:
            try:
                forecasted = forecast_demand(demand, horizon=len(demand), method="auto")
                forecasted.index = demand.index
                st.sidebar.success("Prévision ML appliquée.")
                demand_forecast_plot = plot_forecast_vs_actual(demand, forecasted)
                demand = forecasted
            except Exception as e:
                st.sidebar.warning(f"Prévision ML indisponible: {e}")

    with st.spinner("Résolution du modèle..."):
        model = UnitCommitmentModel(
            plants=modified_plants,
            demand=demand,
            availability=modified_availability if not modified_availability.empty else None,
            co2_price=co2_price_active,
            voll=global_params.voll,
            reserve_margin=global_params.reserve_margin,
        )

        try:
            model.build_model()
        except Exception as e:
            st.error(f"Erreur construction modèle: {e}")
            st.stop()

        
        if solver_name not in ["highs", "appsi_highs"]:
            from pyomo.environ import Suffix
            model.model.dual = Suffix(direction=Suffix.IMPORT)

        try:
            result = model.solve(
                solver_name=solver_name, timeout=timeout, mip_gap=mip_gap
            )
        except ValueError as e:
            error_msg = str(e)
            if "non disponible" in error_msg or "non disponible" in error_msg.lower():
                st.error(
                    f"❌ **Solveur non disponible**\n\n"
                    f"{error_msg}\n\n"
                    f"Installez le solveur {solver_name} (CBC, GLPK ou HiGHS)."
                )
            else:
                st.error(f"❌ **Erreur résolution**\n\n{error_msg}")
            st.stop()
        except Exception as e:
            st.error(f"❌ **Erreur inattendue**\n\n{str(e)}")
            import traceback
            with st.expander("Détails"):
                st.code(traceback.format_exc())
            st.stop()

    if result.success:
        st.success(f"✅ Optimisation réussie ! Coût total: **{result.objective_value:,.2f} €**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Coût Total", f"{result.objective_value:,.0f} €")
        with col2:
            if result.total_load_shedding > 0:
                st.metric(
                    "⚠️ Délestage",
                    f"{result.total_load_shedding:,.0f} MWh",
                    delta=f"-{result.total_load_shedding * global_params.voll:,.0f} €",
                    delta_color="inverse",
                )
            else:
                st.metric("✅ Délestage", "0 MWh")
        with col3:
            st.metric("Émissions CO2", f"{result.co2_emissions:,.0f} t")
        with col4:
            st.metric("Prix Marginal Moy.", f"{result.marginal_prices.mean():,.1f} €/MWh")

        if result.total_load_shedding > 0:
            st.error(
                f"""
                ⚠️ **DÉLESTAGE NÉCESSAIRE**

                Énergie non servie : **{result.total_load_shedding:,.0f} MWh**
                Coût économique : **{result.total_load_shedding * global_params.voll:,.0f} €** (VoLL = {global_params.voll:,.0f} €/MWh)

                Le système ne peut pas satisfaire toute la demande avec la capacité disponible.
                """
            )

        st.header("📈 Visualisations")
        hours = demand.index

        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
        fig_price.add_trace(
            go.Scatter(
                x=hours,
                y=result.marginal_prices,
                name="Prix marginal €/MWh",
                line=dict(color="red", width=2),
                fill="tozeroy",
            ),
            secondary_y=False,
        )
        fig_price.add_trace(
            go.Scatter(
                x=hours,
                y=demand,
                name="Demande MW",
                line=dict(color="blue", dash="dash"),
            ),
            secondary_y=True,
        )
        fig_price.update_layout(title="📈 Prix Marginal de l'Électricité")
        st.plotly_chart(fig_price, use_container_width=True)
        st.caption("Le prix marginal = coût de la dernière centrale appelée = prix du marché spot")

        if demand_forecast_plot is not None:
            st.plotly_chart(demand_forecast_plot, use_container_width=True)

        fig_mix = plot_production_mix(result.production_schedule, demand)
        st.plotly_chart(fig_mix, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            fig_costs = plot_cost_breakdown(result.cost_breakdown)
            st.plotly_chart(fig_costs, use_container_width=True)
            st.plotly_chart(plot_co2_emissions(result, modified_plants), use_container_width=True)
        with col_right:
            fig_commit = plot_commitment_schedule(result.commitment_schedule)
            st.plotly_chart(fig_commit, use_container_width=True)
            st.plotly_chart(plot_merit_order(modified_plants, co2_price_active), use_container_width=True)
            st.plotly_chart(plot_plant_clusters(modified_plants, co2_price_active, n_clusters=3), use_container_width=True)

        st.subheader("Production Détaillée par Centrale")
        fig_detailed = plot_production_detailed(result.production_schedule)
        st.plotly_chart(fig_detailed, use_container_width=True)

        st.header("📋 Données Détaillées")
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Production", "État Centrales", "Démarrages", "Délestage/Écrêtement"]
        )

        with tab1:
            st.dataframe(result.production_schedule.round(2), use_container_width=True)
        with tab2:
            st.dataframe(result.commitment_schedule, use_container_width=True)
        with tab3:
            st.dataframe(result.startup_schedule, use_container_width=True)
        with tab4:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Délestage (MW)**")
                st.dataframe(result.load_shedding, use_container_width=True)
            with col_b:
                st.markdown("**Écrêtement renouvelables (MW)**")
                st.dataframe(result.curtailment, use_container_width=True)

        st.subheader("✅ Vérification de la Satisfaction de la Demande")
        total_production = result.production_schedule.sum(axis=1)
        verification_df = pd.DataFrame(
            {
                "Heure": demand.index,
                "Demande (MW)": demand.values,
                "Production (MW)": total_production.values,
                "Délestage (MW)": result.load_shedding.values,
                "Prod + Délestage (MW)": total_production.values + result.load_shedding.values,
                "Écart (MW)": (
                    total_production.values + result.load_shedding.values - demand.values
                ),
            }
        )
        st.dataframe(verification_df, use_container_width=True, hide_index=True)

    else:
        st.error(f"❌ Échec de l'optimisation: {result.status}")

else:
    st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur 'Optimiser' pour lancer l'optimisation.")

    st.header("📋 Aperçu de la Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Centrales")
        plants_df = pd.DataFrame({
            "Nom": [p.name for p in plants],
            "Type": [p.plant_type for p in plants],
            "P_min (MW)": [p.p_min for p in plants],
            "P_max (MW)": [p.p_max for p in plants],
            "Coût Var (€/MWh)": [p.cost_variable for p in plants],
        })
        st.dataframe(plants_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Scénarios Disponibles")
        for name, info in PREDEFINED_SCENARIOS.items():
            st.markdown(f"**{name}**: {info['description']}")

if __name__ == "__main__":
    pass
