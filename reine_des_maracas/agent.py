# -*- coding: utf-8 -*-
"""
Data Orchestrator Agents (v3) for Reine des Maracas.
This version implements a flexible, multi-agent architecture where a root
orchestrator routes tasks to specialized sub-agents (UX, Metadata, SQL, Viz).
"""

from __future__ import annotations

import os
import re
import json
import logging
import asyncio
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from functools import lru_cache
from collections import defaultdict

import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError
from google.adk.agents.llm_agent import LlmAgent
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- Vérification et importation des dépendances ---
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

# --- Configuration ---
MODEL = os.getenv("DATA_MODEL", "gemini-1.5-flash-002")
if os.getenv("VERTEX_PROJECT") and not os.getenv("VERTEXAI_PROJECT"):
    os.environ["VERTEXAI_PROJECT"] = os.environ["VERTEX_PROJECT"]
if os.getenv("VERTEX_LOCATION") and not os.getenv("VERTEXAI_LOCATION"):
    os.environ["VERTEXAI_LOCATION"] = os.environ["VERTEXAI_LOCATION"]
PROJECT = os.getenv("VERTEXAI_PROJECT", "avisia-training")
LOCATION = os.getenv("VERTEXAI_LOCATION", "europe-west1")
DATASET = os.getenv("VERTEX_BQ_DATASET", "reine_des_maracas")
DESCRIPTION_PATH = os.getenv("SQL_SCHEMA_PATH", "./table_description.json")
EXAMPLES_PATH = os.getenv("SQL_EXAMPLES_PATH", "./sql_examples.json")

if VERTEXAI_AVAILABLE:
    try:
        if PROJECT:
            vertexai.init(project=PROJECT, location=LOCATION)
            logging.info(f"Vertex AI SDK initialized for project '{PROJECT}' in '{LOCATION}'.")
    except Exception as e:
        logging.warning(f"Could not initialize Vertex AI SDK. Error: {e}")
        VERTEXAI_AVAILABLE = False
else:
    logging.warning("Vertex AI SDK not found. Install 'google-cloud-aiplatform' to enable semantic search.")

# --- Gestion du Schéma ---
def _bq_client() -> Optional[bigquery.Client]:
    try:
        if not PROJECT: return None
        return bigquery.Client(project=PROJECT, location=os.getenv("BQ_LOCATION"))
    except Exception as e:
        logging.warning(f"Le client BigQuery n'a pas pu être initialisé : {e}")
        return None

@lru_cache(maxsize=1)
def get_enriched_schema() -> Dict[str, Any]:
    client = _bq_client()
    live_schema: Dict[str, Any] = {"tables": []}
    descriptions = _load_static_descriptions()
    table_descriptions = {t["name"]: t for t in descriptions.get("tables", [])}
    if client and DATASET:
        try:
            query = f"SELECT table_name, column_name, data_type FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS` ORDER BY table_name, ordinal_position;"
            rows = client.query(query).result()
            tables_temp: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"fields": []})
            for row in rows:
                tables_temp[row.table_name]["name"] = row.table_name
                tables_temp[row.table_name]["fields"].append({"name": row.column_name, "type": row.data_type})
            for name, table_data in tables_temp.items():
                enriched_table = table_descriptions.get(name, {})
                field_descriptions = {f["name"]: f for f in enriched_table.get("fields", [])}
                merged_fields = [{**live_field, **field_descriptions.get(live_field["name"], {})} for live_field in table_data["fields"]]
                live_schema["tables"].append({"name": name, "description": enriched_table.get("description", ""), "fields": merged_fields})
            logging.info(f"Schéma chargé et enrichi pour {len(live_schema['tables'])} tables.")
        except GoogleAPICallError as e:
            logging.warning(f"Échec de la récupération du schéma live : {e}. Utilisation du schéma statique.")
            return descriptions
    return live_schema

def _load_static_descriptions() -> Dict[str, Any]:
    try:
        if os.path.exists(DESCRIPTION_PATH):
            with open(DESCRIPTION_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.warning(f"Impossible de charger le fichier de description du schéma : {e}")
    return {"tables": []}

@lru_cache(maxsize=1)
def _load_sql_examples() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(EXAMPLES_PATH):
            with open(EXAMPLES_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.warning(f"Impossible de charger les exemples SQL : {e}")
    return []

# --- Moteur de Recherche Sémantique ---
class EmbeddingClient:
    def __init__(self, model_name: str = "text-embedding-004"):
        if not VERTEXAI_AVAILABLE: raise RuntimeError("Le SDK Vertex AI n'est pas disponible.")
        self.model = TextEmbeddingModel.from_pretrained(model_name)
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([])
        try:
            return np.array([e.values for e in self.model.get_embeddings(texts)])
        except Exception as e:
            logging.error(f"Échec de l'obtention des embeddings : {e}")
            return np.zeros((len(texts), 768))

class SemanticIndex:
    def __init__(self, metadata: List[Dict[str, Any]], vectors: np.ndarray, client: EmbeddingClient):
        self.metadata = metadata
        self.vectors = vectors
        self.client = client
        self._normalize_vectors()
    def _normalize_vectors(self):
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.normalized_vectors = np.divide(self.vectors, norms, out=np.zeros_like(self.vectors), where=norms!=0)
    @staticmethod
    def create(schema: Dict[str, Any], client: EmbeddingClient) -> "SemanticIndex":
        logging.info("Construction de l'index sémantique...")
        docs, metadata = [], []
        for table in schema.get("tables", []):
            for field in table.get("fields", []):
                doc = f"Table: {table.get('name', '')}. Colonne: {field.get('name', '')}. Description: {field.get('description', '')}."
                docs.append(doc)
                metadata.append({"table": table.get('name', ''), "field": field.get('name', '')})
        vectors = client.get_embeddings(docs)
        logging.info(f"Index sémantique construit avec {len(metadata)} entrées.")
        return SemanticIndex(metadata, vectors, client)
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.client.get_embeddings([query])
        query_norm = query_vec / np.linalg.norm(query_vec)
        sims = np.dot(self.normalized_vectors, query_norm.T).flatten()
        indices = np.argpartition(sims, -top_k)[-top_k:]
        sorted_indices = indices[np.argsort(-sims[indices])]
        return [{**self.metadata[i], "score": sims[i]} for i in sorted_indices]

_SEMANTIC_INDEX: Optional[SemanticIndex] = None
def _initialize_components():
    global _SEMANTIC_INDEX
    if VERTEXAI_AVAILABLE and not _SEMANTIC_INDEX:
        client = EmbeddingClient()
        schema = get_enriched_schema()
        _SEMANTIC_INDEX = SemanticIndex.create(schema, client)

# --- Outils pour les Agents ---
def find_relevant_schema(question: str) -> Dict[str, Any]:
    """Trouve les tables et colonnes les plus pertinentes pour une question."""
    if not _SEMANTIC_INDEX: return {"error": "L'index sémantique n'est pas initialisé."}
    
    search_results = _SEMANTIC_INDEX.search(question, top_k=15)
    
    table_context = defaultdict(list)
    for res in search_results:
        if res['score'] > 0.6: # Augmenter le seuil de pertinence
            table_context[res['table']].append(res['field'])
            
    # Retourner un objet JSON propre, pas une chaîne formatée
    return {"relevant_tables": dict(table_context)}

def rag_sql_examples(question: str) -> Dict[str, Any]:
    """Récupère des exemples de questions-SQL similaires à la question de l'utilisateur."""
    if not _SEMANTIC_INDEX: return {"examples": "L'index sémantique n'est pas initialisé."}
    
    examples = _load_sql_examples()
    if not examples: return {"examples": "Aucun exemple SQL disponible."}
    
    example_questions = [ex['question'] for ex in examples]
    all_texts = [question] + example_questions
    all_embeddings = _SEMANTIC_INDEX.client.get_embeddings(all_texts)
    
    q_vec = all_embeddings[0]
    ex_vecs = all_embeddings[1:]
    
    sims = np.dot(ex_vecs, q_vec.T) / (np.linalg.norm(ex_vecs, axis=1) * np.linalg.norm(q_vec))
    
    top_indices = np.argsort(-sims)[:3]
    
    # Retourner une liste d'objets JSON, pas une chaîne formatée
    top_examples = [examples[i] for i in top_indices]
    return {"examples": top_examples}

def run_sql(query: str) -> Dict[str, Any]:
    """Exécute une requête SQL SELECT-only sur BigQuery."""
    if not query: return {"error": "Requête vide reçue."}
    
    # Nettoyage simple des ```sql
    if match := re.compile(r"^\s*```(?:sql)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE).match(query): query = match.group(1)
    final_query = query.strip()

    if not re.compile(r"^\s*SELECT\b", re.IGNORECASE).search(final_query):
        return {"error": "Seules les requêtes SELECT sont autorisées."}

    client = _bq_client()
    if not client: return {"error": "Client BigQuery non disponible."}
    
    try:
        job = client.query(final_query)
        rows = list(job.result(max_results=1000))
        if not rows: return {"result": "La requête a fonctionné mais n'a retourné aucune ligne."}
        
        cols = [sf.name for sf in job.schema]
        data = [{col: row[i] for i, col in enumerate(cols)} for row in rows]
        return {"result": json.dumps(data, indent=2, default=str)}
    except Exception as e:
        logging.error(f"Échec du job SQL : {e} sur la requête : {final_query}")
        return {"error": f"Erreur lors de l'exécution de la requête : {str(e)}"}


def chart_spec(data_json: str, user_intent: str) -> Dict[str, Any]:
    """Génère une spécification Vega-Lite basique."""
    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not data:
            return {"spec": "Données invalides ou vides."}
    except json.JSONDecodeError:
        return {"spec": "Erreur de formatage des données d'entrée (JSON invalide)."}

    cols = list(data[0].keys())
    if len(cols) < 2: return {"spec": "Pas assez de colonnes pour un graphique."}
    
    x_field, y_field = None, None
    for col in cols:
        val = data[0][col]
        if isinstance(val, (int, float)) and not y_field:
            y_field = col
        elif isinstance(val, str) and not x_field:
            x_field = col
    
    if not x_field or not y_field:
        x_field, y_field = cols[0], cols[1]

    mark = "line" if any(kw in user_intent for kw in ["évolution", "tendance", "courbe"]) else "bar"
    
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data},
        "mark": {"type": mark, "tooltip": True},
        "encoding": {
            "x": {"field": x_field, "type": "nominal", "axis": {"labelAngle": -45}},
            "y": {"field": y_field, "type": "quantitative"},
        },
    }
    return {"vega_lite_spec": json.dumps(spec, indent=2)}

# --- Définitions des Agents ---

ux_agent = LlmAgent(
    name="ux_agent", model=MODEL,
    description="Clarifie les questions vagues des utilisateurs.",
    instruction="Ta tâche est de reformuler la question de l'utilisateur ou de poser une question de clarification si elle est trop vague. Sois concis. Ne réponds qu'avec la clarification.",
)

metadata_agent = LlmAgent(
    name="metadata_agent", model=MODEL,
    description="Trouve les tables et colonnes pertinentes pour une question.",
    instruction="Tu es un expert en schémas de données. Utilise l'outil `find_relevant_schema` pour identifier le contexte de données nécessaire pour répondre à la question de l'utilisateur. Retourne uniquement le résultat de l'outil.",
    tools=[find_relevant_schema],
)

sql_agent = LlmAgent(
    name="sql_agent", model=MODEL,
    description="Génère et exécute une requête SQL pour répondre à une question, en se basant sur un contexte de schéma.",
    instruction=(
        "Tu es un expert SQL pour BigQuery. Ta mission est de répondre à la question de l'utilisateur en générant et exécutant une seule requête.\n"
        "1. Tu recevras la question, un objet JSON `relevant_tables` et un objet JSON `examples`.\n"
        "2. **Logique Métier :** Avant d'écrire la requête, vérifie si la question de l'utilisateur correspond à un concept métier (une 'metric' ou une 'dimension'). Si c'est le cas, utilise l'expression SQL fournie dans la description du concept.\n"
        "3. Inspire-toi des exemples fournis pour la syntaxe correcte.\n"
        "4. **RÈGLE ABSOLUE :** Ta requête DOIT qualifier chaque nom de table avec `avisia-training.reine_des_maracas.`. Par exemple : `FROM `avisia-training.reine_des_maracas.ticket_caisse` AS t`.\n"
        "5. Écris une requête SQL `SELECT` qui utilise **uniquement** les tables et colonnes de `relevant_tables`.\n"
        "6. Finalement, exécute cette requête avec l'outil `run_sql`.\n"
        "7. Ta réponse finale doit être le résultat de l'outil `run_sql`."
    ),
    tools=[rag_sql_examples, run_sql],
)


viz_agent = LlmAgent(
    name="viz_agent", model=MODEL,
    description="Crée une spécification de graphique à partir de données JSON.",
    instruction="Tu recevras des données au format JSON et l'intention de l'utilisateur. Utilise l'outil `chart_spec` pour générer une spécification Vega-Lite. Retourne uniquement le résultat de l'outil.",
    tools=[chart_spec],
)

root_agent = LlmAgent(
    name="root_agent_reine_des_maracas", model=MODEL,
    description="Orchestre les agents spécialisés pour répondre aux questions analytiques.",
    instruction=(
        "Tu es l'orchestrateur principal. Ton but est de répondre à la question de l'utilisateur en coordonnant les agents spécialisés. Voici ton plan d'action :\n"
        "1. **Étape 1 : Obtenir le Contexte.** Appelle `metadata_agent` avec la question de l'utilisateur pour savoir quelles tables et colonnes sont pertinentes.\n"
        "2. **Étape 2 : Obtenir les Données.** Prends la question originale de l'utilisateur ET le contexte de schéma de l'étape 1, et passe les deux à `sql_agent` pour qu'il génère et exécute la requête SQL.\n"
        "3. **Étape 3 : Présenter le Résultat.** La sortie de `sql_agent` est la réponse finale. Présente-la clairement à l'utilisateur.\n"
        "4. **Gestion de l'ambiguïté (si nécessaire) :** Si, à n'importe quelle étape, un agent retourne une erreur ou si la question semble vraiment trop vague, tu peux appeler `ux_agent` pour demander une clarification.\n"
        "5. **Visualisation (Optionnel) :** Si la question initiale demande un 'graphique' ou une 'visualisation' et que tu as obtenu des données de `sql_agent`, passe ces données et la question à `viz_agent`."
    ),
    sub_agents=[ux_agent, metadata_agent, sql_agent, viz_agent],
)

# --- Initialisation au démarrage ---
_initialize_components()