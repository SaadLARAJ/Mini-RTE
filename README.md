# mini-rte — Optimiseur de Réseau Électrique (Unit Commitment)

## Explication
- **Ce que c’est** : un simulateur de dispatch électrique horaire qui choisit quelles centrales allumer pour servir la demande au moindre coût, en intégrant CO₂, délestage et contraintes techniques.
- **Pourquoi c’est crédible** : modèle MILP (Pyomo) avec prix marginal issu des duals, Value of Lost Load (VoLL), rampes linéaires, écrêtement renouvelables, scénarios de marché (crise gaz/CO₂).
- **Ce que vous voyez dans l’UI** : sliders CO₂, scénarios prêts (gaz cher, crise 2022, backtests 2022/2023), alertes délestage, graphiques prix marginal / merit order / émissions.
- **Compétences démontrées** : optimisation (MILP), culture marché énergie (merit order, ETS/CO₂, VoLL), data viz (Plotly), industrialisation (package Python, tests, scripts d’entraînement, Streamlit).
- **Prêt à l’usage** : interface Streamlit, CLI, tests Pytest, script pour télécharger/enrichir les données (ENTSO-E) et entraîner une prévision.

## En un coup d’œil
- **Dispatch horaire optimisé** (MILP Pyomo) avec rampes linéaires, démarrages bornés, délestage VoLL (20 000 €/MWh) et réserve optionnelle.
- **Prix du CO₂** dans le merit order, **prix marginal** via duals, bilan d’**émissions**, **écrêtement** explicite des renouvelables.
- **UI Streamlit** : scénarios (gaz cher, CO₂ haut/bas, crise 2022, backtests 2022/2023), slider CO₂, alertes délestage, graphiques prix marginal/merit order/émissions.
- **Data science** : clustering léger des centrales (merit order agrégé), prévision de demande baseline (RandomForest si dispo, fallback polyfit).

## Ce que ça démontre
- Optimisation & RO : UC (binaire/continu), rampes, VoLL, duals/prix marginal, écrêtement des intermittents.
- Marché énergie : merit order, ETS/CO₂, Value of Lost Load, scénarios de stress (gaz/CO₂).
- Data science : clustering (k-means léger) des centrales par coût marginal/puissance, prévision de charge (RandomForest ou polyfit).
- Math/Modélisation : formulation MILP avec contraintes linéarisées (ramp-up/down), start-up borné, réserve optionnelle, pénalisation du délestage.
- Ingénierie : packaging Python (`mini_rte`), tests Pytest, scripts de training/données, Streamlit UI.

## Architecture
```
mini-rte/
├── app/streamlit_app.py          # UI Streamlit
├── config/default_config.yaml    # Centrales, paramètres globaux, scénarios
├── data/                         # Profils de demande/renouvelables
├── src/mini_rte/
│   ├── __init__.py
│   ├── config.py                 # Chargement/validation YAML
│   ├── data_loader.py            # Lecture CSV
│   ├── forecast.py               # Prévision de demande
│   ├── solver.py                 # Gestion des solveurs
│   ├── scenarios.py              # Scénarios prédéfinis
│   ├── visualization.py          # Plotly (mix, coûts, clustering, forecast)
│   └── models/
│       ├── plant.py
│       └── unit_commitment.py    # Modèle UC (MILP)
├── scripts/train_forecast.py     # Téléchargement ENTSO-E + entraînement RF
├── tests/
│   ├── test_model.py
│   └── test_data_loader.py
└── main.py                       # CLI de résolution
```

## Modèle UC — points clés
- Variables u/p/start, ramp-up/down linéaires, démarrage borné.
- Délestage `shed[t]` (VoLL configurable) pour éviter l’infaisabilité, réserve optionnelle.
- Coûts CO₂ dynamiques (`emission_factor * co2_price`), bilan d’émissions.
- Écrêtement explicite des renouvelables, prix marginal tiré du dual d’équilibre offre/demande.

### Contrainte et coût (aperçu)
- **Objectif** : min (coût variable + fixe + démarrage + CO₂ + VoLL*délestage).
- **Équilibre** : Σ p[i,t] + shed[t] ≥ D[t]*(1+réserve).
- **Bornes** : u[i,t]*Pmin ≤ p[i,t] ≤ u[i,t]*Pmax.
- **Rampes** : p[i,t] - p[i,t-1] ≤ R[i], p[i,t-1] - p[i,t] ≤ R[i].
- **Start-up** : start ≥ u[t]-u[t-1], start ≤ u[t], start ≤ 1-u[t-1].
- **Renouvelables** : p + curtail ≤ dispo * Pmax.
- **Prix marginal** : dual de l’équilibre (shadow price) = signal de marché.

## Scénarios & paramètres
- Globaux : `co2_price`, `voll`, `reserve_margin` (`config/default_config.yaml`).
- Scénarios : low_wind, high_gas_price, nuclear_outage, summer_solar, combined_stress, low_co2, high_co2, crisis_2022, backtest_2022, backtest_2023.
- Facteurs d’émission renseignés, capacités hydro/gaz renforcées.

## Utilisation
- CLI : `python main.py --config config/default_config.yaml`
- Streamlit : `streamlit run app/streamlit_app.py` (CBC/GLPK/HiGHS/Gurobi)
- Prévision (optionnel) :
  1. `export ENTSOE_API_KEY="votre_cle"` ne pas oublier votre propre clé la mienne n'est pas présente.
  2. `python scripts/train_forecast.py --start 2022-01-01 --end 2023-12-31`
  3. Dans l’UI, cochez “Prévoir la demande”.

## Tests
```
pytest tests/ -v
```

## Extensions possibles
- Min up/down times.
- Réserves primaires/secondaires (marge 5 %).
- Contraintes réseau (transport inter-zones).
- Stockage : batteries ou STEP (cycles charge/décharge).
