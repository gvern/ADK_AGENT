### Architecture d’agents

## root_agent (orchestrateur)
Routage : Compréhension ↔ Metadata ↔ SQL ↔ Viz.
Règles :

1. Détecter si la question est vague → passer à ux_agent (reformulation, précisions).

2. Si analytique → demander metadata_agent (tables/colonnes pertinentes) puis sql_agent (SQL + exécution).

3. Si l’utilisateur veut des graphes → root_agent appelle directement ses outils de visualisation (`chart_spec_tool` → `persist_viz_artifacts` → `render_vega_block`).

## ux_agent (compréhension / glossaire / reformulation)

- Reformule, pose des questions de précision.

- Traduit vocabulaire métier → tables/champs (via un petit glossaire embarqué).

## metadata_agent (métadonnées utiles uniquement)

- Explore le schéma BigQuery ciblé (ou JSON statique fallback).

- Ne retourne que les tables/champs pertinents pour la question.

## sql_agent (génération + exécution)

- Utilise RAG d’exemples Q↔SQL (tool d’exemples) + schéma renvoyé par metadata.

- Génère un SQL SELECT-only, le valide (no DDL/DML), l’exécute via run_sql (tool).

## Visualisation (pilotée par root_agent)

- Génère une spec Vega-Lite à partir des résultats tabulaires via `chart_spec_tool`.
- Persiste la spec et un PNG via `persist_viz_artifacts` quand c'est possible.
- Prépare le bloc d'affichage chat via `render_vega_block`.