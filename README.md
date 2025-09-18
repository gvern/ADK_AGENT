# Reine des Maracas – Orchestrateur Data (UX / Metadata / SQL / Viz)

Nouvelle architecture orientée analytics :

```
root_agent_reine_des_maracas
  ├── ux_agent (clarification / reformulation)
  ├── metadata_agent (sélection tables / colonnes pertinentes)
  ├── sql_agent (génération + exécution SQL BigQuery via run_sql)
  └── viz_agent (spec Vega-Lite facultative)
```

Tools exposés (JSON): `get_schema`, `suggest_relevant`, `rag_sql_examples`, `run_sql`, `chart_spec`.

Les anciens agents SAV / Sales ont été retirés (voir historique git si besoin de rollback).

---

## 1) Environment variables (Data)

### Obligatoires

| Variable | Rôle |
|----------|------|
| `VERTEX_PROJECT` | Projet GCP pour BigQuery |
| `VERTEX_BQ_DATASET` | Dataset BigQuery cible |
| `VERTEX_LOCATION` | Région (ex: europe-west1) |
| `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` | Mappées automatiquement si absentes |
| `GOOGLE_GENAI_USE_VERTEXAI` | Doit être `true` pour Vertex |

### Optionnels

| Variable | Rôle |
|----------|------|
| `DATA_MODEL` | Modèle Vertex (défaut `gemini-1.5-flash-002`) |
| `SQL_EXAMPLES_PATH` | Chemin vers le fichier d'exemples (défaut `./sql_examples.json`) |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | Cohérence outils gcloud |

---

## 2) Fichiers projet

```
.
├─ agent.py                # Orchestrateur data + sous-agents + tools analytics
├─ adk_app.py              # Exporte root_agent pour ADK Web
├─ sql_examples.json       # Exemples Q↔SQL (RAG lexical simple)
├─ table_description.json  # Schéma statique utilisé en fallback
├─ requirements.txt
└─ apps/reine_des_maracas/adk_app.py  # Variante pour ADK apps folder
```

---

## 3) .env template (data)

```dotenv
VERTEX_PROJECT=avisia-training
VERTEX_BQ_DATASET=reine_des_maracas
VERTEX_LOCATION=europe-west1
GOOGLE_GENAI_USE_VERTEXAI=true
VERTEXAI_PROJECT=${VERTEX_PROJECT}
VERTEXAI_LOCATION=${VERTEX_LOCATION}
GOOGLE_CLOUD_PROJECT=${VERTEX_PROJECT}
GOOGLE_CLOUD_LOCATION=${VERTEX_LOCATION}
DATA_MODEL=gemini-1.5-flash-002
SQL_EXAMPLES_PATH=./sql_examples.json
```

---

## 4) Installation & tests rapides

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD"
export GOOGLE_GENAI_USE_VERTEXAI=true
export VERTEX_PROJECT=avisia-training
export VERTEX_BQ_DATASET=reine_des_maracas
export VERTEX_LOCATION=europe-west1
export VERTEXAI_PROJECT=$VERTEX_PROJECT
export VERTEXAI_LOCATION=$VERTEX_LOCATION
adk web apps
```

Ouvrir http://127.0.0.1:8000 puis chosir l'app.

### Exemples de questions

1. « Donne le CA par ville sur les 30 derniers jours. »  → metadata → sql (+ viz si demandé)
2. « Quels segments génèrent le plus de CA ? »
3. « Annulations immédiates par famille le mois dernier »
4. « Evolution quotidienne du CA sur 2 semaines »

---

## 5) ADK Web UI

Le fichier `adk_app.py` exporte `root_agent` sous le nom `AGENT` (convention ADK).

---

## 6) (Optionnel) Vertex AI Agent Engine

Même logique : empaqueter `root_agent`. Ajouter les dépendances (BigQuery SDK) dans `requirements.txt` côté Reasoning Engine.

---

## 7) Agentspace (optionnel)

Si vous souhaitez intégrer dans une interface Agentspace, créez un Reasoning Engine puis référencez son resource name.

---

## 8) Troubleshooting (data)

| Problème | Cause probable | Piste |
|----------|----------------|-------|
| `Only pure SELECT queries are allowed` | Mot clé DML/DDL détecté | Retirer UPDATE/INSERT… |
| Schéma vide | Pas d'accès INFORMATION_SCHEMA | Fallback statique déjà utilisé, vérifier IAM | 
| Dates NULL | Format JJ/MM/YYYY non parsé | Vérifier usage de SAFE.PARSE_DATE | 
| Pas de suggestion de tables | Question trop vague | Passer par ux_agent | 

Export rapide pour vérifier modèle :

```bash
python - <<'PY'
import os; from google import genai; c=genai.Client(vertexai=True,project=os.environ['VERTEXAI_PROJECT'],location=os.environ['VERTEXAI_LOCATION']);print('OK?', bool(c.models.generate_content(model=os.getenv('DATA_MODEL','gemini-1.5-flash-002'), contents='ping').candidates))
PY
```

---

## 9) Notes de conception

* Sécurité SQL : regex SELECT-only + blacklist DML/DDL.
* RAG lexical simple (Jaccard) sur `sql_examples.json` pour éviter surcharge infra.
* Schéma dynamique si BigQuery accessible, sinon fallback statique compact.
* Viz minimaliste (Vega-Lite) → rendu côté front / notebook.

---

## 10) Roadmap potentielle

1. Embedding semantic RAG sur exemples SQL (FAISS / Vertex Matching Engine).
2. Caching résultats requêtes fréquentes (clé hash SQL).
3. Détection automatique d'intention de visualisation (classif légère).
4. Ajout d'un outil `explain_sql(query)` pour pédagogie utilisateur.
5. Guardrail supplémentaire pour LIMIT automatique si absence de GROUP BY.

---

Pour revenir à l'ancienne version (SAV/Sales), consulter l'historique git avant ce commit.

# Start MCP locally
uvicorn mcp_server:app --host 0.0.0.0 --port 8080

# Point agents to MCP
export MCP_HTTP_URL="http://localhost:8080"

# Run ADK Web UI
export PYTHONPATH="$PWD"
adk web apps
```

---

## 13) License / Notes

* This repo uses experimental ADK features (you’ll see warnings in logs).
* Keep `requirements.txt` lean; add only what your tools need (Cloud Run/Agent Engine will pip install it).
* If you later add auth to the MCP gateway, set `MCP_HTTP_BEARER` and forward it in `agent.py` MCP calls.

---

**You’re set.** Next time: export the env block, start MCP (or point to Cloud Run), and use either the CLI or `adk web apps`.


find . -name "__pycache__" -type d -exec rm -rf {} +              
export VERTEXAI_PROJECT="avisia-training"                                           
export VERTEXAI_LOCATION="europe-west1"
export VERTEX_BQ_DATASET="reine_des_maracas"
export SQL_SCHEMA_PATH="./table_description.json"
gcloud auth application-default login -q
gcloud auth application-default set-quota-project avisia-training -q
export PYTHONPATH="$PWD"
adk web apps