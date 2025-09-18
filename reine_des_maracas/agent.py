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

# --- Vérification et importation des dépendances ---
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

# --- Configuration ---
MODEL = os.getenv("DATA_MODEL", "gemini-2.5-flash")
if os.getenv("VERTEX_PROJECT") and not os.getenv("VERTEXAI_PROJECT"):
    os.environ["VERTEXAI_PROJECT"] = os.environ["VERTEX_PROJECT"]
if os.getenv("VERTEX_LOCATION") and not os.getenv("VERTEXAI_LOCATION"):
    os.environ["VERTEXAI_LOCATION"] = os.environ["VERTEX_LOCATION"]
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
        if query_vec.size == 0 or not np.any(query_vec):
            return []
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            return []
        query_norm = query_vec / qnorm
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
    """
    Retourne:
      - relevant_tables: {table: [colonnes pertinentes]}
      - schema_details:  tables -> champs (name, type, mode, description, allowed_values, format), joins filtrés
      - business_concepts: metrics/dimensions depuis table_description.json
    """
    if not _SEMANTIC_INDEX:
        return {"error": "L'index sémantique n'est pas initialisé."}

    # 1) Recherche sémantique
    search_results = _SEMANTIC_INDEX.search(question, top_k=20)

    table_context = defaultdict(list)
    for res in search_results:
        if res["score"] > 0.45:
            table_context[res["table"]].append(res["field"])

    # 2) Safety nets (ventes/ville)
    ql = question.lower()
    if ("ticket_caisse" in table_context) or any(k in ql for k in ["vente", "ventes", "ca", "chiffre d'affaires"]):
        must_tc = ["DATE_TICKET", "PRIX_AP_REMISE", "QUANTITE", "CODE_BOUTIQUE", "ANNULATION", "ANNULATION_IMMEDIATE"]
        table_context["ticket_caisse"] = sorted(set(table_context["ticket_caisse"] + must_tc))
    if any(city in ql for city in ["paris", "lyon", "marseille", "toulouse", "lille"]):
        must_m = ["VILLE", "CODE_BOUTIQUE"]
        table_context["magasin"] = sorted(set(table_context["magasin"] + must_m))

    # Fallback si vide
    if not table_context:
        table_context = {
            "ticket_caisse": ["DATE_TICKET", "PRIX_AP_REMISE", "QUANTITE", "CODE_BOUTIQUE", "ANNULATION", "ANNULATION_IMMEDIATE"],
            "magasin": ["VILLE", "CODE_BOUTIQUE"],
        }

    # 3) Enrichissement: types, descriptions, allowed_values, format
    enriched = get_enriched_schema()          # merge live + statique
    static_desc = _load_static_descriptions() # brut du JSON (pour joins/concepts)

    # index rapides
    tbl_map = {t["name"]: t for t in enriched.get("tables", [])}
    static_tbl_map = {t["name"]: t for t in static_desc.get("tables", [])}

    schema_details = {"tables": {}}

    # ne garder que les tables pertinentes
    kept_tables = set(table_context.keys())

    for tname in kept_tables:
        t_live = tbl_map.get(tname, {"fields": []})
        t_static = static_tbl_map.get(tname, {"fields": []})
        # map pour retrouver allowed_values/format du JSON statique
        stat_fields = {f["name"]: f for f in t_static.get("fields", [])}

        fields_out = []
        for f in t_live.get("fields", []):
            fname = f.get("name")
            if fname in set(table_context[tname]):  # ne sortir que les colonnes pertinentes
                stat = stat_fields.get(fname, {})
                fields_out.append({
                    "name": fname,
                    "type": f.get("type"),
                    "mode": f.get("mode"),
                    "description": f.get("description") or stat.get("description"),
                    "allowed_values": stat.get("allowed_values"),
                    "format": stat.get("format"),
                    "key": stat.get("key"),
                })

        pk = [ff["name"] for ff in t_static.get("fields", []) if str(ff.get("key","")).startswith("PRIMARY_KEY")]
        fk = [ff["name"] for ff in t_static.get("fields", []) if "FOREIGN_KEY" in str(ff.get("key",""))]
        schema_details["tables"][tname] = {
            "description": t_live.get("description"),
            "fields": fields_out,
            "joins": [],
            "keys": {"primary": pk, "foreign": fk},
        }

    # 4) Relations (joins) filtrées aux tables retenues
    joins = static_desc.get("relations", [])
    filtered_joins = []
    for j in joins:
        left_tbl = j.get("left", "").split(".")[0]
        right_tbl = j.get("right", "").split(".")[0]
        if left_tbl in kept_tables or right_tbl in kept_tables:
            filtered_joins.append(j)
    # distribuer par table
    for j in filtered_joins:
        for side in ["left", "right"]:
            t_side = j.get(side, "").split(".")[0]
            if t_side in schema_details["tables"]:
                schema_details["tables"][t_side]["joins"].append(j)

    # 5) Règles métier (concepts)
    business_concepts = static_desc.get("concepts", {})

    return {
        "relevant_tables": dict(table_context),
        "schema_details": schema_details,
        "business_concepts": business_concepts,
    }


def rag_sql_examples(question: str) -> Dict[str, Any]:
    """Récupère des exemples de questions-SQL similaires à la question de l'utilisateur."""
    if not _SEMANTIC_INDEX: return {"examples": "L'index sémantique n'est pas initialisé."}
    
    examples = _load_sql_examples()
    if not examples: return {"examples": []}
    
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
    """Exécute une requête SQL SELECT-only sur BigQuery en qualifiant intelligemment les tables."""
    if not query: return {"error": "Requête vide reçue."}
    
    # Nettoyage initial
    if match := re.compile(r"^\s*```(?:sql)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE).match(query): query = match.group(1)
    final_query = query.strip()

    if not re.compile(r"^\s*SELECT\b", re.IGNORECASE).search(final_query):
        return {"error": "Seules les requêtes SELECT sont autorisées."}

    # Remplacement intelligent : n'ajoute le préfixe que si la table n'est pas déjà qualifiée.
    def qualify_table_names(q: str) -> str:
        # FROM/JOIN <name>   où <name> ∈ {table | dataset.table | project.dataset.table}
        # et pas un appel de fonction (pas de '(' juste après le token)
        pattern = r"\b(FROM|JOIN)\s+`?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,2})`?(?!\s*\()"
        def repl(m):
            kw = m.group(1)
            full = m.group(2).strip("`")
            parts = full.split(".")
            if len(parts) == 3:
                # déjà project.dataset.table → inchangé
                return f"{kw} `{full}`"
            if len(parts) == 2:
                ds, tbl = parts
                return f"{kw} `{PROJECT}.{ds}.{tbl}`"
            # len==1
            return f"{kw} `{PROJECT}.{DATASET}.{parts[0]}`"
        return re.sub(pattern, repl, q, flags=re.IGNORECASE)


    # Appliquer la qualification uniquement si l'agent a "oublié" le nom complet
    # On vérifie la présence du nom du projet dans la requête pour décider.

    final_query = qualify_table_names(final_query)
    
    client = _bq_client()
    if not client: return {"error": "Client BigQuery non disponible."}
    
    try:
        job = client.query(final_query)
        rows_iter = job.result(max_results=1000)
        
        cols = [sf.name for sf in rows_iter.schema]
        rows = [list(row.values()) for row in rows_iter]

        if not rows: return {"result": "La requête a fonctionné mais n'a retourné aucune ligne."}
        
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

# --- Correctif pour l'Automatic Function Calling (AFC) ---
# Matérialise les annotations de chaînes en types réels pour éviter les erreurs AFC
from typing import get_type_hints

def _materialize_annotations(func):
    try:
        func.__annotations__ = get_type_hints(func)
    except Exception:
        pass
    return func

# Matérialiser les annotations des fonctions exposées comme tools
_materialize_annotations(find_relevant_schema)
_materialize_annotations(rag_sql_examples)
_materialize_annotations(run_sql)
_materialize_annotations(chart_spec)

# --- Définitions des Agents ---

ux_agent = LlmAgent(
    name="ux_agent",
    model=MODEL,
    description="Clarifie les questions vagues des utilisateurs.",
    instruction=(
        "Ta tâche est de reformuler la question de l'utilisateur ou de poser une question de clarification si elle est trop vague. "
        "Sois concis. Ne réponds qu'avec la clarification."
    ),
)

metadata_agent = LlmAgent(
    name="metadata_agent",
    model=MODEL,
    description="Trouve les tables et colonnes pertinentes pour une question.",
    instruction=(

        "Utilise STRICTEMENT l’outil `find_relevant_schema(question=<question>)`.\n"
        "Après la réponse de l’outil, renvoie EXACTEMENT l’objet JSON retourné, sans autre texte."
    ),
    tools=[find_relevant_schema],
)


sql_agent = LlmAgent(
    name="sql_agent", model=MODEL,
    description="Génère et exécute une requête SQL pour répondre à une question, en se basant sur un contexte de schéma.",
    instruction=(
        "Tu es un expert SQL BigQuery.\n"
        "Tu recevras :\n"
        "- `question`\n"
        "- `find_relevant_schema_response` (avec `relevant_tables`, `schema_details`=tables/fields/joins et `business_concepts`)\n"
        "- `examples`\n"
        "**RÈGLE ABSOLUE :** Ta requête DOIT qualifier chaque nom de table avec `avisia-training.reine_des_maracas.`. Par exemple : `FROM `avisia-training.reine_des_maracas.ticket_caisse` AS t`.\n"
        "Règles strictes :\n"
        "• Utilise les *relations* de `schema_details` pour choisir les JOINs et les clés.\n"
        "• N’emploie que les colonnes présentes dans `schema_details.tables[*].fields`.\n"
        "• Si la question correspond à un concept métier, utilise `business_concepts.metrics/dimensions` (expression SQL).\n"
        "• `DATE_TICKET` est STRING au format DD/MM/YYYY → `SAFE.PARSE_DATE('%d/%m/%Y', t.DATE_TICKET)` pour tout EXTRACT/filtre.\n"
        "• `ANNULATION` et `ANNULATION_IMMEDIATE` sont BOOL → filtre avec `NOT t.ANNULATION AND NOT t.ANNULATION_IMMEDIATE`.\n"
        "• Pour filtrer les villes, fais une comparaison insensible à la casse :UPPER(m.VILLE) = UPPER('<valeur utilisateur>').\n"
        "• Toutes les tables doivent être **entièrement qualifiées** `avisia-training.reine_des_maracas.*`.\n"
        "• Mesure principale alias `ca` et enveloppée par `COALESCE(...,0)`.\n"
        "Ensuite, exécute via `run_sql` et renvoie son résultat."
    ),
    tools=[rag_sql_examples, run_sql],
)


viz_agent = LlmAgent(
    name="viz_agent",
    model=MODEL,
    description="Crée une spécification de graphique à partir de données JSON.",
    instruction=(
        "Tu recevras des données au format JSON et l'intention de l'utilisateur. "
        "Utilise l'outil `chart_spec` pour générer une spécification Vega-Lite."
    ),
    tools=[chart_spec],
)

root_agent = LlmAgent(
    name="root_agent_reine_des_maracas", model=MODEL,
    description=(
        "Orchestre les agents spécialisés pour répondre aux questions analytiques. "
        "Après `metadata_agent`, transfert obligatoire vers `sql_agent` via `transfer_to_agent` avec la question, la réponse de schéma et la consigne."
    ),
    instruction=(
        "Tu es l'orchestrateur principal. Ton but est de répondre à la question de l'utilisateur en coordonnant les agents spécialisés. Voici ton plan d'action :\n"
        "1. **Étape 1 : Obtenir le Contexte.** Appelle `metadata_agent` avec la question de l'utilisateur pour savoir quelles tables et colonnes sont pertinentes.\n"
        "2. **Étape 2 : Obtenir les Données.** Après avoir reçu la réponse JSON de `metadata_agent`, tu DOIS immédiatement appeler `transfer_to_agent` vers `sql_agent` en lui envoyant un message EXACTEMENT structuré ainsi :\n"
        "   - question: \"<question originale>\"\n"
        "   - find_relevant_schema_response: <le JSON retourné par metadata_agent, inchangé>\n"
        "   - consigne: \"Génère une seule requête SELECT BigQuery, entièrement qualifiée avisia-training.reine_des_maracas.*, puis exécute-la via run_sql.\"\n"
        "   Ne modifie pas le JSON de schéma.\n"
        "   Important : Si tu as appelé `metadata_agent`, ne réponds JAMAIS directement à l'utilisateur tant que `sql_agent` n’a pas renvoyé le résultat.\n"
        "   Une fois le transfert effectué, attends la sortie de `sql_agent`.\n"
        "3. **Étape 3 : Présenter le Résultat.** La sortie de `sql_agent` est la réponse finale. Présente-la clairement à l'utilisateur.\n"
        "4. **Gestion de l'ambiguïté (si nécessaire) :** Si, à n'importe quelle étape, un agent retourne une erreur ou si la question semble vraiment trop vague, tu peux appeler `ux_agent` pour demander une clarification.\n"
        "5. **Visualisation (Optionnel) :** Si la question initiale demande un 'graphique' ou une 'visualisation' et que tu as obtenu des données de `sql_agent`, passe ces données et la question à `viz_agent`."
    ),
    sub_agents=[ux_agent, metadata_agent, sql_agent, viz_agent],
)

# --- Initialisation au démarrage ---
_initialize_components()