# Reine des Maracas – Orchestrateur Data (BigQuery)

Orchestrateur multi-agents pour questions analytiques en langage naturel, ciblant BigQuery: UX → Metadata → SQL → Viz (gérée par root_agent).

```
root_agent_reine_des_maracas
  ├── ux_agent        # clarification / reformulation
  ├── metadata_agent  # contexte complet: schéma + exemples
  └── sql_agent       # génération + exécution SQL (BigQuery SELECT-only)
     ↳ outils de viz (appelés par root_agent): chart_spec_tool, persist_viz_artifacts, render_vega_block
```

Points clés

- BigQuery en lecture seule (SELECT/WITH seulement) avec qualification automatique des tables et backticks de sécurité
- Schéma enrichi (INFORMATION_SCHEMA live + `table_description.json` statique)
- Exemples SQL (RAG) et exemple prioritaire auto-retargeté quand très pertinent
- UI locale via ADK Web

Outils (côté agents)

- metadata_agent → `get_full_context_for_sql(question)`
- sql_agent → `run_sql(query)`
- root_agent → `chart_spec_tool(data_json, user_intent)`, `persist_viz_artifacts(...)`, `render_vega_block(...)`

Des helpers existent aussi: `rag_sql_examples(question)`, `get_full_schema_context()`.

## Sommaire

1. Prérequis
2. Installation
3. Configuration (.env)
4. Lancer l’app (ADK Web)
5. Exemples de questions
6. Dépannage
7. Notes de conception
8. Roadmap

---

## 1) Prérequis

- Python 3.12+
- Accès GCP BigQuery au dataset cible (droits lecture)
- gcloud CLI pour l’authentification ADC
- ADK installé (UI) : paquet Python « google-adk » et binaire `adk`
- Librairies data: BigQuery SDK; optionnel pour RAG sémantique: Vertex AI SDK

## 2) Installation

Dans un environnement virtuel Python:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r reine_des_maracas/requirements.txt
```

Authentification Google Cloud (ADC):

```bash
gcloud auth application-default login -q
gcloud auth application-default set-quota-project avisia-training -q
```

## 3) Configuration (.env)

Variables essentielles

- VERTEX_BQ_DATASET: dataset BigQuery (ex: reine_des_maracas)
- VERTEXAI_PROJECT ou VERTEX_PROJECT: projet GCP (l’un renseigne l’autre)
- VERTEXAI_LOCATION ou VERTEX_LOCATION: région (ex: europe-west1)
- BQ_LOCATION: emplacement des jobs BigQuery (ex: EU)
- DATA_MODEL: modèle LLM (défaut: gemini-2.5-pro)
- SQL_SCHEMA_PATH: chemin du schéma statique (fallback / enrichissement)
- SQL_EXAMPLES_PATH: chemin d’exemples Q↔SQL (RAG)
- EXAMPLE_MIN_SCORE: seuil (0-1) pour marquer un exemple “prioritaire” (défaut 0.85)

Flags utiles

- DISABLE_SCHEMA_CACHE=1 pour invalider le cache du schéma
- DISABLE_EXAMPLES_CACHE=1 pour invalider le cache des exemples

Exemple `.env` minimal:

```dotenv
VERTEX_BQ_DATASET=reine_des_maracas
VERTEXAI_PROJECT=avisia-training
VERTEXAI_LOCATION=europe-west1
BQ_LOCATION=EU
DATA_MODEL=gemini-2.5-pro
SQL_SCHEMA_PATH=${PWD}/reine_des_maracas/table_description.json
SQL_EXAMPLES_PATH=${PWD}/reine_des_maracas/sql_examples.json
```

Astuce: si vous utilisez VERTEX_PROJECT/LOCATION, elles seront mappées automatiquement vers VERTEXAI_PROJECT/LOCATION.

## 4) Lancer l’app (ADK Web)

```bash
export PYTHONPATH="$PWD"
adk web apps
```

Ensuite, ouvrez http://127.0.0.1:8000 et choisissez l’app « reine_des_maracas ».

Le fichier `apps/reine_des_maracas/adk_app.py` (ainsi que `reine_des_maracas/adk_app.py`) exporte `AGENT` conformément à la convention ADK.

## 5) Exemples de questions

- « Donne le CA par ville sur les 30 derniers jours. »  → metadata → sql (+ viz si demandé)
- « Quels segments génèrent le plus de CA ? »
- « Annulations immédiates par famille le mois dernier »
- « Évolution quotidienne du CA sur 2 semaines »

## 6) Dépannage

- Seules les requêtes SELECT/WITH sont autorisées: retirez UPDATE/INSERT/DDL
- Schéma vide: pas d’accès INFORMATION_SCHEMA → fallback statique utilisé; vérifiez IAM
- Dates NULL: `DATE_TICKET` est une STRING JJ/MM/YYYY → `SAFE.PARSE_DATE('%d/%m/%Y', t.DATE_TICKET)` est appliqué automatiquement aux comparaisons courantes
- Projets avec tirets: la qualification + backticks sont ajoutés automatiquement (voir log « Final SQL: »)
- Pas de tables proposées: question trop vague → laissez l’UX agent clarifier

Vérif rapide du client Vertex (facultatif):

```bash
python - <<'PY'
import os
from google import genai
c = genai.Client(vertexai=True, project=os.environ.get('VERTEXAI_PROJECT'), location=os.environ.get('VERTEXAI_LOCATION'))
print('OK?', bool(c.models.generate_content(model=os.getenv('DATA_MODEL','gemini-2.5-pro'), contents='ping').candidates))
PY
```

## 7) Notes de conception

- Garde-fous SQL (SELECT-only) + qualification auto `project.dataset.table` + backticks 3-parties
- Retarget automatique des références 3-parties vers `${VERTEXAI_PROJECT}.${VERTEX_BQ_DATASET}.<table>`
- Schéma enrichi live + statique; relations issues du JSON pour guider les JOINs
- Règles métier (concepts) exposées aux agents via le contexte complet
- RAG d’exemples + exemple prioritaire retargeté si score ≥ EXAMPLE_MIN_SCORE
- Spécification Vega-Lite générée directement par le root_agent (via ses outils) et artefacts persistés si possible

## 8) Roadmap

1. RAG sémantique embeddings (FAISS / Vertex Matching Engine)
2. Cache des résultats fréquents (clé de hash SQL)
3. Détection d’intentions de visualisation
4. Outil `explain_sql(query)`
5. Guardrail: LIMIT automatique si absence de GROUP BY

---

Notes

- Repo potentiellement basé sur des fonctionnalités ADK expérimentales (logs verbeux possibles)
- Gardez `requirements` minimal; n’ajoutez que ce qui est nécessaire aux outils

