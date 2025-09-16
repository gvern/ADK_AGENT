### Architecture d’agents

## root_agent (orchestrateur)
Routage : Compréhension ↔ Metadata ↔ SQL ↔ Viz.
Règles :

1. Détecter si la question est vague → passer à ux_agent (reformulation, précisions).

2. Si analytique → demander metadata_agent (tables/colonnes pertinentes) puis sql_agent (SQL + exécution).

3. Si l’utilisateur veut des graphes → viz_agent (optionnel).

## ux_agent (compréhension / glossaire / reformulation)

- Reformule, pose des questions de précision.

- Traduit vocabulaire métier → tables/champs (via un petit glossaire embarqué).

## metadata_agent (métadonnées utiles uniquement)

- Explore le schéma BigQuery ciblé (ou JSON statique fallback).

- Ne retourne que les tables/champs pertinents pour la question.

## sql_agent (génération + exécution)

- Utilise RAG d’exemples Q↔SQL (tool d’exemples) + schéma renvoyé par metadata.

- Génère un SQL SELECT-only, le valide (no DDL/DML), l’exécute via run_sql (tool).

## viz_agent (optionnel)

- Propose un chart spec (Vega-Lite) à partir des résultats tabulaires (sans rendre l’image côté LLM).

- Laisse le front ou un notebook afficher.