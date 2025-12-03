# mini-rte — Optimiseur de Réseau Électrique (Unit Commitment)

> **Status: Prototype**  
> Ce projet est un démonstrateur technique d'optimisation énergétique.

## Description
**mini-rte** est un simulateur de dispatch électrique horaire (Unit Commitment) résolu par programmation linéaire mixte en nombres entiers (MILP). Il détermine le plan de production optimal pour satisfaire la demande au moindre coût, sous contraintes techniques et environnementales.

Le projet inclut une interface de visualisation et des outils d'analyse de scénarios (crise gaz, variation prix CO₂, etc.).

## Fonctionnalités Clés
*   **Modélisation MILP (Pyomo)** :
    *   Décisions binaires d'engagement (Unit Commitment).
    *   Contraintes de rampes (montée/descente).
    *   Coûts de démarrage et coûts fixes.
    *   Gestion de la réserve et du délestage (Value of Lost Load).
*   **Économie de l'Énergie** :
    *   Calcul du prix marginal via les variables duales.
    *   Intégration du marché carbone (ETS).
    *   Merit Order dynamique.
*   **Interface & Analyse** :
    *   Dashboard interactif (Streamlit).
    *   Comparaison de scénarios pré-configurés.
    *   Visualisation des flux, coûts et émissions.

## Architecture Technique
Le projet est structuré comme un package Python standard :

```
mini-rte/
├── app/                # Interface utilisateur (Streamlit)
├── config/             # Configuration (YAML) et définitions des centrales
├── data/               # Séries temporelles (demande, renouvelables)
├── src/mini_rte/       # Cœur du simulateur
│   ├── models/         # Modèles d'optimisation (Pyomo)
│   ├── solver.py       # Interface solveurs (HiGHS, CBC, GLPK)
│   └── ...
└── scripts/            # Outils (récupération données ENTSO-E, training)
```

## Installation

Pré-requis : Python 3.10+

```bash
pip install -r requirements.txt
```

Pour utiliser le solveur HiGHS (recommandé pour les performances) :
```bash
pip install highspy
```

## Utilisation

### Interface Graphique
Lancer le tableau de bord interactif :
```bash
streamlit run app/streamlit_app.py
```

### Ligne de Commande
Exécuter une simulation simple :
```bash
python main.py --config config/default_config.yaml
```

### Tests
Lancer la suite de tests :
```bash
pytest tests/
```

