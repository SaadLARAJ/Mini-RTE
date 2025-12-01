"""
Fonctions de visualisation avec Plotly pour l'analyse des résultats.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_production_mix(
    schedule: pd.DataFrame,
    demand: Optional[pd.Series] = None,
    title: str = "Mix de Production Électrique",
) -> go.Figure:
    """
    Crée un graphique empilé du mix de production par heure.

    Args:
        schedule: DataFrame avec la production par centrale et par heure
        demand: Série avec la demande horaire (optionnel, pour vérification)
        title: Titre du graphique

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    # Couleurs par type de centrale (approximation basée sur le nom)
    colors = {
        "nuclear": "#FFD700",  # Or
        "gas": "#FF6B6B",  # Rouge
        "wind": "#4ECDC4",  # Turquoise
        "solar": "#FFE66D",  # Jaune
        "hydro": "#95E1D3",  # Vert clair
    }

    # Grouper par type de centrale
    plant_types = {}
    for col in schedule.columns:
        col_lower = col.lower()
        if "nuclear" in col_lower:
            plant_type = "nuclear"
        elif "gas" in col_lower:
            plant_type = "gas"
        elif "wind" in col_lower:
            plant_type = "wind"
        elif "solar" in col_lower:
            plant_type = "solar"
        elif "hydro" in col_lower:
            plant_type = "hydro"
        else:
            plant_type = "other"

        if plant_type not in plant_types:
            plant_types[plant_type] = []
        plant_types[plant_type].append(col)

    # Créer les graphiques empilés par type
    for plant_type, plant_names in plant_types.items():
        type_production = schedule[plant_names].sum(axis=1)
        color = colors.get(plant_type, "#CCCCCC")
        fig.add_trace(
            go.Scatter(
                x=schedule.index,
                y=type_production,
                mode="lines",
                name=plant_type.capitalize(),
                stackgroup="one",
                fillcolor=color,
                line=dict(width=0.5, color=color),
            )
        )

    # Ajouter la demande si fournie
    if demand is not None:
        fig.add_trace(
            go.Scatter(
                x=demand.index,
                y=demand.values,
                mode="lines",
                name="Demande",
                line=dict(color="black", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Heure",
        yaxis_title="Puissance (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    return fig


def plot_cost_breakdown(
    cost_breakdown: Dict[str, float], title: str = "Décomposition des Coûts"
) -> go.Figure:
    """
    Crée un graphique en barres de la décomposition des coûts.

    Args:
        cost_breakdown: Dictionnaire avec les coûts (variable, fixed, startup, total)
        title: Titre du graphique

    Returns:
        Figure Plotly
    """
    # Exclure "total" pour le graphique en barres
    costs = {k: v for k, v in cost_breakdown.items() if k != "total"}

    palette = ["#4ECDC4", "#FF6B6B", "#FFD700", "#95E1D3", "#6C5B7B"]
    colors = palette[: len(costs)] if costs else palette

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(costs.keys()),
                y=list(costs.values()),
                text=[f"{v:,.0f} €" for v in costs.values()],
                textposition="auto",
                marker_color=colors,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Type de coût",
        yaxis_title="Coût (€)",
        height=400,
    )

    return fig


def plot_scenario_comparison(
    results: List[Dict], title: str = "Comparaison Multi-Scénarios"
) -> go.Figure:
    """
    Compare plusieurs scénarios sur différents métriques.

    Args:
        results: Liste de dictionnaires avec les résultats de chaque scénario
                 Chaque dict doit contenir: "name", "cost", "production_schedule", etc.
        title: Titre du graphique

    Returns:
        Figure Plotly avec sous-graphiques
    """
    if not results:
        raise ValueError("La liste de résultats ne peut pas être vide")

    # Créer des sous-graphiques
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Coût Total",
            "Production Moyenne par Type",
            "Production Maximale",
            "Nombre de Démarrages",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    scenario_names = [r["name"] for r in results]
    costs = [r.get("cost", 0) for r in results]

    # Graphique 1: Coût total
    fig.add_trace(
        go.Bar(x=scenario_names, y=costs, name="Coût Total", marker_color="#4ECDC4"),
        row=1,
        col=1,
    )

    # Graphique 2: Production moyenne par type (simplifié)
    # Calculer la production moyenne par type pour chaque scénario
    for i, result in enumerate(results):
        schedule = result.get("production_schedule", pd.DataFrame())
        if not schedule.empty:
            # Production moyenne totale
            avg_prod = schedule.sum(axis=1).mean()
            fig.add_trace(
                go.Bar(
                    x=[scenario_names[i]],
                    y=[avg_prod],
                    name=f"Prod Moy {i}",
                    showlegend=False,
                    marker_color="#FF6B6B",
                ),
                row=1,
                col=2,
            )

    # Graphique 3: Production maximale
    max_prods = []
    for result in results:
        schedule = result.get("production_schedule", pd.DataFrame())
        if not schedule.empty:
            max_prod = schedule.sum(axis=1).max()
            max_prods.append(max_prod)
        else:
            max_prods.append(0)

    fig.add_trace(
        go.Bar(x=scenario_names, y=max_prods, name="Prod Max", marker_color="#FFD700"),
        row=2,
        col=1,
    )

    # Graphique 4: Nombre de démarrages
    startups = []
    for result in results:
        startup_schedule = result.get("startup_schedule", pd.DataFrame())
        if not startup_schedule.empty:
            total_startups = startup_schedule.sum().sum()
            startups.append(total_startups)
        else:
            startups.append(0)

    fig.add_trace(
        go.Bar(x=scenario_names, y=startups, name="Démarrages", marker_color="#95E1D3"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Scénario", row=2, col=1)
    fig.update_xaxes(title_text="Scénario", row=2, col=2)
    fig.update_yaxes(title_text="Coût (€)", row=1, col=1)
    fig.update_yaxes(title_text="Puissance (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Puissance (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Nombre", row=2, col=2)

    return fig


def plot_production_detailed(
    schedule: pd.DataFrame, title: str = "Production Détaillée par Centrale"
) -> go.Figure:
    """
    Crée un graphique avec la production de chaque centrale individuellement.

    Args:
        schedule: DataFrame avec la production par centrale et par heure
        title: Titre du graphique

    Returns:
        Figure Plotly
    """
    fig = go.Figure()

    for col in schedule.columns:
        fig.add_trace(
            go.Scatter(
                x=schedule.index,
                y=schedule[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Heure",
        yaxis_title="Puissance (MW)",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    return fig


def plot_commitment_schedule(
    commitment: pd.DataFrame, title: str = "Planning d'Allumage des Centrales"
) -> go.Figure:
    """
    Crée un graphique de type heatmap pour visualiser l'état on/off des centrales.

    Args:
        commitment: DataFrame avec l'état on/off (0/1) par centrale et par heure
        title: Titre du graphique

    Returns:
        Figure Plotly (heatmap)
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=commitment.values.T,
            x=commitment.index,
            y=commitment.columns,
            colorscale=[[0, "#FF6B6B"], [1, "#4ECDC4"]],
            showscale=True,
            colorbar=dict(title="État", tickvals=[0, 1], ticktext=["OFF", "ON"]),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Heure",
        yaxis_title="Centrale",
        height=400,
    )

    return fig


def _kmeans_simple(features: np.ndarray, k: int, iterations: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """K-means léger sans dépendance externe."""
    rng = np.random.default_rng(42)
    # Init aléatoire parmi les points
    centers = features[rng.choice(len(features), size=k, replace=False)]
    for _ in range(iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        for i in range(k):
            points = features[labels == i]
            if len(points) > 0:
                centers[i] = points.mean(axis=0)
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    return labels, centers


def plot_plant_clusters(
    plants: Sequence[object],
    co2_price: float,
    n_clusters: int = 3,
) -> go.Figure:
    """Clustérise les centrales (coût marginal + capacité) et affiche les groupes."""
    data = []
    for p in plants:
        cost = p.cost_variable + p.emission_factor * co2_price
        data.append(
            {
                "name": p.name,
                "type": p.plant_type,
                "cost": cost,
                "capacity": p.p_max,
            }
        )

    features = np.array([[d["cost"], d["capacity"]] for d in data], dtype=float)
    k = min(n_clusters, len(data))
    labels, centers = _kmeans_simple(features, k)

    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#6C5B7B", "#1A535C", "#FF8C42"]
    fig = go.Figure()
    for cluster in range(k):
        cluster_points = [d for d, lab in zip(data, labels) if lab == cluster]
        fig.add_trace(
            go.Scatter(
                x=[d["capacity"] for d in cluster_points],
                y=[d["cost"] for d in cluster_points],
                mode="markers+text",
                text=[d["name"] for d in cluster_points],
                textposition="top center",
                marker=dict(size=12, color=colors[cluster % len(colors)]),
                name=f"Cluster {cluster+1}",
                hovertemplate="<b>%{text}</b><br>Coût: %{y:.1f} €/MWh<br>Capacité: %{x:.0f} MW",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=centers[:, 1],
            y=centers[:, 0],
            mode="markers",
            marker_symbol="x",
            marker_size=14,
            marker_color="#000",
            name="Centres",
            hovertemplate="Centre<br>Coût: %{y:.1f}<br>Capacité: %{x:.0f}",
        )
    )

    fig.update_layout(
        title=f"🤖 Clustering des centrales (k={k})",
        xaxis_title="Capacité (MW)",
        yaxis_title="Coût marginal (€/MWh) incluant CO2",
        showlegend=True,
        height=450,
    )
    return fig


def plot_forecast_vs_actual(actual: pd.Series, forecast: pd.Series) -> go.Figure:
    """Compare la série réelle et la prévision."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=actual.index, y=actual.values, name="Demande réelle", line=dict(color="#1A535C"))
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name="Prévision ML",
            line=dict(color="#FF6B6B", dash="dash"),
        )
    )
    fig.update_layout(
        title="📈 Prévision de la demande (baseline ML)",
        xaxis_title="Heure",
        yaxis_title="Puissance (MW)",
        hovermode="x unified",
        height=400,
    )
    return fig
