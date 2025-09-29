# -*- coding: utf-8 -*-
"""
Data Orchestrator Agents (v3) for Reine des Maracas.
This version implements a flexible, multi-agent architecture where a root
orchestrator routes tasks to specialized sub-agents (UX, Metadata, SQL, Viz).
"""

from __future__ import annotations

import os
from pathlib import Path
from importlib import resources
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
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
import google.genai.types as genai_types

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
BASE_DIR = os.path.dirname(__file__)
DESCRIPTION_PATH = os.getenv(
    "SQL_SCHEMA_PATH",
    os.path.join(BASE_DIR, "table_description.json")
)
EXAMPLES_PATH = os.getenv(
    "SQL_EXAMPLES_PATH",
    os.path.join(BASE_DIR, "sql_examples.json")
)

logging.basicConfig(level=logging.INFO) 
logging.info(f"Chemin des exemples SQL utilisé : {EXAMPLES_PATH}")
logging.info(f"Le fichier existe-t-il ? {'Oui' if os.path.exists(EXAMPLES_PATH) else 'Non'}")

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
            job = client.query(
                query,
                location=os.getenv("BQ_LOCATION"),
                job_config=bigquery.QueryJobConfig()   
            )
            # TIMEOUT DUR ET COURT (ex. 8s)
            rows = job.result(timeout=8)
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
    """
    Recherche dans l'ordre :
      1) Chemin explicite via $SQL_SCHEMA_PATH
      2) Fichier embarqué dans le package `reine_des_maracas/table_description.json`
      3) Fichier à la racine de l'app (./table_description.json)
    """
    candidates: List[Path] = []
    # 1) ENV
    if DESCRIPTION_PATH:
        candidates.append(Path(DESCRIPTION_PATH))
    # 2) Package resource
    try:
        pkg_file = resources.files(__package__).joinpath("table_description.json")  # type: ignore[arg-type]
        candidates.append(Path(str(pkg_file)))
    except Exception:
        pass
    # 3) Repo root fallback
    candidates.append(Path(__file__).resolve().parent.parent / "table_description.json")

    for p in candidates:
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logging.info(f"Chargé table_description.json depuis: {p}")
                return data
        except (IOError, json.JSONDecodeError) as e:
            logging.warning(f"Échec lecture {p}: {e}")
    logging.warning("Aucune description statique trouvée. Joins/concepts indisponibles.")
    return {"tables": [], "relations": [], "concepts": {}}

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
_SQL_EXAMPLES_INDEX: Dict[str, Any] = {"examples": [], "vectors": np.array([])}

def _initialize_sql_examples_index():
    """Charge les exemples SQL et pré-calcule leurs embeddings."""
    global _SQL_EXAMPLES_INDEX
    if not VERTEXAI_AVAILABLE:
        logging.warning("L'index des exemples SQL ne peut être initialisé car Vertex AI n'est pas dispo.")
        return

    examples = _load_sql_examples()
    if not examples:
        logging.info("Aucun exemple SQL à indexer.")
        return

    # On ne calcule les embeddings que pour les questions des exemples
    example_questions = [ex['question'] for ex in examples]
    client = EmbeddingClient() # Réutilise le même client que pour le schéma
    vectors = client.get_embeddings(example_questions)

    # Normaliser les vecteurs une seule fois pour des calculs de similarité plus rapides
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
    
    _SQL_EXAMPLES_INDEX["examples"] = examples
    _SQL_EXAMPLES_INDEX["vectors"] = normalized_vectors # On stocke les vecteurs normalisés
    logging.info(f"{len(examples)} exemples SQL ont été chargés et vectorisés.")

def _initialize_components():
    global _SEMANTIC_INDEX
    if VERTEXAI_AVAILABLE and not _SEMANTIC_INDEX:
        client = EmbeddingClient()
        schema = get_enriched_schema()
        _SEMANTIC_INDEX = SemanticIndex.create(schema, client)

# --- Helpers SQL ---

def _extract_cte_names(sql: str) -> set[str]:
    """
    Retourne l'ensemble des noms de CTE définis dans un WITH ... .
    On lit à partir de WITH, on collecte tous les 'nom AS (' jusqu'au SELECT racine
    (après que la parenthésation des CTE soit revenue à zéro).
    """
    s = sql
    m_with = re.search(r"\bWITH\b", s, flags=re.IGNORECASE)
    if not m_with:
        return set()

    i = m_with.end()          # début après WITH
    depth = 0
    names: set[str] = set()

    # petit helper pour lire un ident simple (nom de CTE)
    ident_re = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*AS\s*\(", re.IGNORECASE)

    while i < len(s):
        # quand depth == 0 et on voit 'SELECT', on considère que la liste des CTE est finie
        m_sel = re.match(r"\s*SELECT\b", s[i:], flags=re.IGNORECASE)
        if depth == 0 and m_sel:
            break

        # essaie de capturer "cte_name AS ("
        m_cte = ident_re.match(s, i)
        if m_cte:
            names.add(m_cte.group(1).lower())
            i = m_cte.end()  # on est positionné juste après l'ouverture '(' du CTE
            depth = 1        # on est entré dans le bloc de ce CTE
            # avance jusqu'à refermer entièrement ce CTE (compte () imbriqués)
            while i < len(s) and depth > 0:
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                i += 1
            # après la fermeture, on s'attend à une virgule (autre CTE) ou SELECT (racine)
            # la boucle continue pour potentiellement capturer d'autres CTE
            continue

        # sinon, avance d'un caractère en gérant la profondeur si on tombe sur ( ou )
        c = s[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth = max(0, depth - 1)
        i += 1

    return names

# --- Sanitize SQL Functions ---
def _sanitize_functions(q: str) -> str:
    """
    Nettoie les dérives de génération liées aux fonctions BigQuery :
      - PARSE_DAT`E  → PARSE_DATE
      - <project>.SAFE.PARSE_DATE → SAFE.PARSE_DATE (on retire toute qualification projet/dataset)
      - Backticks résiduels autour des fonctions (ex: `EXTRACT`(...) → EXTRACT(...))
    """
    # 1) Répare PARSE_DAT`E
    q = re.sub(r"PARSE_DAT`?E", "PARSE_DATE", q, flags=re.IGNORECASE)

    # 2) Retire une qualification projet.* (avec ou sans backticks) placée devant des fonctions connues
    funcs = [
        r"SAFE\.PARSE_DATE", r"SAFE\.PARSE_DATETIME", r"SAFE\.PARSE_TIME",
        r"SAFE\.CAST", r"EXTRACT", r"DATE", r"DATETIME", r"TIMESTAMP",
    ]
    proj = re.escape(PROJECT)
    # patterns possibles (ordre important : le plus spécifique d'abord)
    # - `project.func`  → func
    # - project.func    → func
    for f in funcs:
        q = re.sub(rf"`?{proj}\.\`?({f})`?", r"\1", q, flags=re.IGNORECASE)

    # 3) Backticks orphelins sur les noms de fonctions
    q = re.sub(r"`(EXTRACT|DATE|DATETIME|TIMESTAMP|SAFE)`\s*\(", r"\1(", q, flags=re.IGNORECASE)

    return q

# --- Outils pour les Agents ---
#def find_relevant_schema(question: str) -> Dict[str, Any]:
    #"""
    #Retourne:
      #- relevant_tables: {table: [colonnes pertinentes]}
      #- schema_details:  tables -> champs (name, type, mode, description, allowed_values, format), joins filtrés
      #- business_concepts: metrics/dimensions depuis table_description.json
    #"""
    #if not _SEMANTIC_INDEX:
        #return {"error": "L'index sémantique n'est pas initialisé."}
#
    ## 1) Recherche sémantique
    #search_results = _SEMANTIC_INDEX.search(question, top_k=20)
#
    #table_context = defaultdict(list)
    #for res in search_results:
        #if res["score"] > 0.45:
            #table_context[res["table"]].append(res["field"])
#
    ## 2) Safety nets (ventes/ville)
    #ql = question.lower()
    ## élargit le déclencheur "ventes" pour couvrir "vendu", "best sellers", etc.
    #ventes_trigs = [
        #"vente", "ventes", "vendu", "vendus",
        #"meilleures ventes", "meilleurs ventes", "best sellers", "top ventes",
        #"chiffre d'affaires", "ca"
    #]
    #if ("ticket_caisse" in table_context) or any(k in ql for k in ventes_trigs):
        #must_tc = ["DATE_TICKET", "PRIX_AP_REMISE", "QUANTITE", "CODE_BOUTIQUE", "ANNULATION", "ANNULATION_IMMEDIATE"]
        #table_context["ticket_caisse"] = sorted(set(table_context["ticket_caisse"] + must_tc))
    #if any(city in ql for city in ["paris", "lyon", "marseille", "toulouse", "lille"]):
        #must_m = ["VILLE", "CODE_BOUTIQUE"]
        #table_context["magasin"] = sorted(set(table_context["magasin"] + must_m))
#
    ## Fallback si vide
    #if not table_context:
        #table_context = {
            #"ticket_caisse": ["DATE_TICKET", "PRIX_AP_REMISE", "QUANTITE", "CODE_BOUTIQUE", "ANNULATION", "ANNULATION_IMMEDIATE"],
            #"magasin": ["VILLE", "CODE_BOUTIQUE"],
        #}
#
    ## 3) Enrichissement: types, descriptions, allowed_values, format
    #enriched = get_enriched_schema()          # merge live + statique
    #static_desc = _load_static_descriptions() # brut du JSON (pour joins/concepts)
#
    ## index rapides
    #tbl_map = {t["name"]: t for t in enriched.get("tables", [])}
    #static_tbl_map = {t["name"]: t for t in static_desc.get("tables", [])}
#
    #schema_details = {"tables": {}}
#
    ## ne garder que les tables pertinentes
    #kept_tables = set(table_context.keys())
#
    #for tname in kept_tables:
        #t_live = tbl_map.get(tname, {"fields": []})
        #t_static = static_tbl_map.get(tname, {"fields": []})
        ## map pour retrouver allowed_values/format du JSON statique
        #stat_fields = {f["name"]: f for f in t_static.get("fields", [])}
#
        #fields_out = []
        #for f in t_live.get("fields", []):
            #fname = f.get("name")
            #if fname in set(table_context[tname]):  # ne sortir que les colonnes pertinentes
                #stat = stat_fields.get(fname, {})
                #fields_out.append({
                    #"name": fname,
                    #"type": f.get("type"),
                    #"mode": f.get("mode"),
                    #"description": f.get("description") or stat.get("description"),
                    #"allowed_values": stat.get("allowed_values"),
                    #"format": stat.get("format"),
                    #"key": stat.get("key"),
                #})
#
        #pk = [ff["name"] for ff in t_static.get("fields", []) if str(ff.get("key","")).startswith("PRIMARY_KEY")]
        #fk = [ff["name"] for ff in t_static.get("fields", []) if "FOREIGN_KEY" in str(ff.get("key",""))]
        #schema_details["tables"][tname] = {
            #"description": t_live.get("description"),
            #"fields": fields_out,
            #"joins": [],
            #"keys": {"primary": pk, "foreign": fk},
        #}
#
    ## 4) Relations (joins) filtrées aux tables retenues
    #joins = static_desc.get("relations", [])
    #filtered_joins = []
    #for j in joins:
        #left_tbl = j.get("left", "").split(".")[0]
        #right_tbl = j.get("right", "").split(".")[0]
        #if left_tbl in kept_tables or right_tbl in kept_tables:
            #filtered_joins.append(j)
    ## distribuer par table
    #for j in filtered_joins:
        #for side in ["left", "right"]:
            #t_side = j.get(side, "").split(".")[0]
            #if t_side in schema_details["tables"]:
                #schema_details["tables"][t_side]["joins"].append(j)
#
    ## 5) Règles métier (concepts)
    #business_concepts = static_desc.get("concepts", {})
#
    #return {
        #"relevant_tables": dict(table_context),
        #"schema_details": schema_details,
        #"business_concepts": business_concepts,
    #}

def get_full_schema_context() -> Dict[str, Any]:
    """
    Charge et retourne le contexte complet du schéma de la base de données.
    Cette fonction remplace `find_relevant_schema` pour fournir toutes les tables,
    colonnes, relations et concepts métier à l'agent SQL.
    """
    enriched_schema = get_enriched_schema()
    static_desc = _load_static_descriptions()

    schema_details = {"tables": {}}
    relevant_tables = {}
    
    # Indexer les descriptions statiques pour un accès rapide
    static_table_map = {t["name"]: t for t in static_desc.get("tables", [])}

    # 1. Construire les détails de chaque table et champ
    for table_data in enriched_schema.get("tables", []):
        table_name = table_data["name"]
        
        # Obtenir les infos statiques pour cette table
        static_table = static_table_map.get(table_name, {"fields": []})
        static_fields_map = {f["name"]: f for f in static_table.get("fields", [])}

        fields_out = []
        all_field_names = [f["name"] for f in table_data.get("fields", [])]
        
        for f in table_data.get("fields", []):
            fname = f.get("name")
            stat = static_fields_map.get(fname, {})
            fields_out.append({
                "name": fname,
                "type": f.get("type"),
                "mode": f.get("mode"),
                "description": f.get("description") or stat.get("description"),
                "allowed_values": stat.get("allowed_values"),
                "format": stat.get("format"),
                "key": stat.get("key"),
            })
        
        # Remplir la liste des "tables pertinentes" avec toutes les colonnes
        relevant_tables[table_name] = all_field_names

        # Extraire les clés primaires et étrangères
        pk = [ff["name"] for ff in static_table.get("fields", []) if str(ff.get("key","")).startswith("PRIMARY_KEY")]
        fk = [ff["name"] for ff in static_table.get("fields", []) if "FOREIGN_KEY" in str(ff.get("key",""))]

        schema_details["tables"][table_name] = {
            "description": table_data.get("description"),
            "fields": fields_out,
            "joins": [], # Sera rempli à l'étape suivante
            "keys": {"primary": pk, "foreign": fk},
        }

    # 2. Ajouter toutes les relations (joins)
    all_joins = static_desc.get("relations", [])
    for j in all_joins:
        for side in ["left", "right"]:
            t_side = j.get(side, "").split(".")[0]
            if t_side in schema_details["tables"]:
                schema_details["tables"][t_side]["joins"].append(j)

    # 3. Ajouter tous les concepts métier
    business_concepts = static_desc.get("concepts", {})

    return {
        "relevant_tables": relevant_tables,
        "schema_details": schema_details,
        "business_concepts": business_concepts,
    }

def rag_sql_examples(question: str, top_k: int = 3, similarity_threshold: float = 0.65) -> Dict[str, Any]:
    """
    Récupère des exemples de questions-SQL similaires à la question de l'utilisateur
    en utilisant des embeddings pré-calculés.
    """
    if not _SQL_EXAMPLES_INDEX["examples"]:
        return {"examples": []}

    # 1. Calculer l'embedding UNIQUEMENT pour la nouvelle question
    if not VERTEXAI_AVAILABLE:
        return {"examples": []}
    
    client = EmbeddingClient()
    query_vec = client.get_embeddings([question])
    
    # Gérer le cas où l'embedding échoue
    if query_vec.size == 0:
        return {"examples": []}

    # 2. Normaliser le vecteur de la question
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return {"examples": []}
    query_norm_vec = query_vec / q_norm

    # 3. Calculer la similarité avec les vecteurs pré-calculés (très rapide)
    # C'est un simple produit matriciel car tous les vecteurs sont déjà normalisés
    sims = np.dot(_SQL_EXAMPLES_INDEX["vectors"], query_norm_vec.T).flatten()

    # 4. Trier et filtrer par seuil
    # On crée des paires (index, score) pour pouvoir filtrer
    indexed_sims = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
    
    top_examples = []
    for index, score in indexed_sims:
        if score >= similarity_threshold and len(top_examples) < top_k:
            # On ajoute le score à l'exemple pour un éventuel débuggage
            example = _SQL_EXAMPLES_INDEX["examples"][index].copy()
            example["score"] = float(score)  # Conversion pour la sérialisation JSON
            top_examples.append(example)

    return {"examples": top_examples}

def get_full_context_for_sql(question: str) -> Dict[str, Any]:
    """
    Rassemble TOUT le contexte nécessaire pour l'agent SQL : le schéma et les exemples RAG.
    """
    logging.info("Début de la collecte de contexte complet (schéma + exemples)...")
    
    # Appel 1: Obtenir le contexte du schéma
    schema_context = get_full_schema_context()
    
    # Appel 2: Obtenir les exemples pertinents
    examples_context = rag_sql_examples(question=question)
    
    # Fusionner les deux dictionnaires en un seul objet de contexte
    full_context = {**schema_context, **examples_context}
    
    logging.info("Contexte complet collecté.")
    return full_context

def pick_example_sql(question: str, min_score: float = 0.80) -> Optional[str]:
    """
    Recherche un exemple SQL avec un score élevé pour la question donnée.
    Si trouvé, retourne directement la requête SQL de l'exemple.
    """
    ex = rag_sql_examples(question, top_k=1, similarity_threshold=min_score).get("examples", [])
    if not ex:
        return None
    sql = ex[0].get("sql") or ex[0].get("query") or ex[0].get("sql_query")
    if not sql:
        return None
    logging.info(f"Exemple SQL trouvé avec score {ex[0].get('score', 0):.3f} : utilisation directe")
    return sql

def run_sql_with_examples_first(question: str) -> Dict[str, Any]:
    """
    1) Essaie l'exemple (si score élevé) -> run_sql(sql_exemple)
    2) Sinon, retourne une erreur spéciale pour indiquer qu'il faut passer par le LLM
    """
    sql = pick_example_sql(question)
    if sql:
        result = run_sql(sql)
        if "error" not in result:
            # Ajouter des métadonnées pour indiquer qu'on a utilisé un exemple
            result["used_example"] = True
            result["sql_used"] = sql
        return result
    return {"error": "NO_EXAMPLE"}  # Signal pour utiliser sql_agent

def run_sql(query: str) -> Dict[str, Any]:
    """Exécute une requête SQL SELECT-only sur BigQuery en qualifiant intelligemment les tables."""
    if not query: return {"error": "Requête vide reçue."}
    
    # Nettoyage initial
    if match := re.compile(r"^\s*```(?:sql)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE).match(query): query = match.group(1)
    final_query = query.strip()

    if not re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE).search(final_query):
        return {"error": "Seules les requêtes commençant par SELECT ou WITH sont autorisées."}

    # Remplacement intelligent : n'ajoute le préfixe que si la table n'est pas déjà qualifiée.
    def qualify_table_names(q: str, cte_names: set[str]) -> str:
        """
        Qualifie uniquement les tables physiques après FROM/JOIN.
        Cas gérés :
          - table
          - dataset.table
          - project-with-dash.dataset.table
          - déjà backtické : `project.dataset.table`
        Ne touche pas aux fonctions ni sous-requêtes.
        Ignore :
          - les sous-requêtes (token suivi de '(')
          - les CTE (détectés via _extract_cte_names)
        """
        pattern = r"\b(FROM|JOIN)\s+(?:\n|\r|\s)*(`[^`]+`|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_]+){0,2})(?!\s*\()"

        def normalize(name: str) -> str:
            if name.startswith("`") and name.endswith("`"):
                name = name[1:-1]
            return name.strip()

        def repl(m: re.Match) -> str:
            kw = m.group(1)
            token = normalize(m.group(2))
            # Si le token correspond à un CTE → ne pas qualifier
            # (on compare en lower pour être tolérant à la casse)
            if token.lower() in cte_names:
                return f"{kw} {token}"

            parts = token.split(".")
            # Si on a déjà project.dataset.table → on ne touche pas
            if len(parts) == 3:
                return f"{kw} `{token}`"
            # dataset.table → préfixer avec PROJECT
            if len(parts) == 2:
                ds, tbl = parts
                return f"{kw} `{PROJECT}.{ds}.{tbl}`"
            # table → préfixer avec PROJECT.DATASET
            if len(parts) == 1:
                tbl = parts[0]
                return f"{kw} `{PROJECT}.{DATASET}.{tbl}`"
            # fallback
            return f"{kw} `{token}`"

        return re.sub(pattern, repl, q, flags=re.IGNORECASE)

    # Si le LLM a déjà écrit project.dataset.table SANS backticks (ex: avisia-training.xxx.yyy),
    # on ajoute les backticks autour des 3 parties pour éviter l'erreur sur le '-'.
    def backtick_3part_with_project(q: str) -> str:
        proj = re.escape(PROJECT)  # gère le tiret
        # autorise ds et table non quotés (letters/digits/_), pas d’espaces
        pat = rf"(?<!`)({proj}\.[A-Za-z_]\w*\.[A-Za-z_]\w*)(?!`)"
        return re.sub(pat, r"`\1`", q)


    # Appliquer la qualification uniquement si l'agent a "oublié" le nom complet
    # On vérifie la présence du nom du projet dans la requête pour décider.

    cte_names = _extract_cte_names(final_query)       
    final_query = qualify_table_names(final_query, cte_names)
    final_query = backtick_3part_with_project(final_query)
    final_query = _sanitize_functions(final_query)

    # Filet de sécurité : si, malgré tout, un CTE a été backtické/qualifié, on le remet nu.
    if cte_names:
        def _dequalify_cte(m: re.Match) -> str:
            kw, token = m.group(1), m.group(2)
            tok_norm = token.strip("`")
            if tok_norm.split(".")[0].lower() in cte_names or tok_norm.lower() in cte_names:
                return f"{kw} {tok_norm.split('.')[-1]}"
            return m.group(0)

        final_query = re.sub(
            r"\b(FROM|JOIN)\s+(`[^`]+`|[A-Za-z0-9_.-]+)",
            _dequalify_cte,
            final_query,
            flags=re.IGNORECASE
        )
    
    client = _bq_client()
    if not client: return {"error": "Client BigQuery non disponible."}
    
    try:
        logging.info(f"Final SQL:\n{final_query}")
        job = client.query(final_query, location=os.getenv("BQ_LOCATION"))
        rows_iter = job.result(max_results=1000)
        
        cols = [sf.name for sf in rows_iter.schema]
        rows = [list(row.values()) for row in rows_iter]

        if not rows: return {"result": "La requête a fonctionné mais n'a retourné aucune ligne."}
        
        data = [{col: row[i] for i, col in enumerate(cols)} for row in rows]
        return {"result": json.dumps(data, indent=2, default=str)}

    except Exception as e:
        logging.error(f"Échec du job SQL : {e} sur la requête : {final_query}")
        return {"error": f"Erreur lors de l'exécution de la requête : {str(e)}"}
    
async def chart_spec_tool(data_json: str, user_intent: str, *_, **kwargs) -> Dict[str, Any]:
    """
    Génère une spec Vega-Lite à partir de données et enregistre l'image + la spec
    comme artifacts ADK via ToolContext. Retourne un petit texte.
    """
    import json
    # 1) Parse data
    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not data:
            return {"text": "Données vides ou invalides pour le graphique."}
    except json.JSONDecodeError:
        return {"text": "JSON invalide."}

    sample = data[0]
    cols = list(sample.keys())
    intent = (user_intent or "").lower()

    # Candidats label (x) par priorité
    x_priority = [
        "libelle_modele", "libellé_modèle", "ville", "ville_magasin",
        "famille", "ligne", "id_modele", "id_modèle", "magasin", "boutique",
    ]

    # Candidats métriques (y) par priorité / alias
    y_alias_groups = [
        ("ca", "chiffre_affaires", "chiffre", "revenue", "sales"),
        ("quantite_vendue", "quantité", "quantite", "qty", "qte"),
        ("panier_moyen", "basket", "avg_basket"),
    ]

    # Si l'intention mentionne clairement une métrique, forcer l’ordre
    if any(k in intent for k in ["chiffre", "ca", "revenue", "vente", "ventes"]):
        y_alias_groups = [
            ("ca", "chiffre_affaires", "chiffre", "revenue", "sales"),
            ("quantite_vendue", "quantité", "quantite", "qty", "qte"),
            ("panier_moyen",),
        ]
    elif any(k in intent for k in ["quantité", "quantite", "qty", "qte"]):
        y_alias_groups = [
            ("quantite_vendue", "quantité", "quantite", "qty", "qte"),
            ("ca", "chiffre_affaires", "chiffre", "revenue", "sales"),
            ("panier_moyen",),
        ]

    # Détection types
    numeric_cols = [c for c in cols_original if isinstance(sample[c], (int, float))]
    text_cols    = [c for c in cols_original if isinstance(sample[c], str)]

    # Exclure les IDs/codes des candidats métriques
    def looks_like_id(name: str) -> bool:
        n = name.lower()
        return (
            n.startswith("id_") or
            n.startswith("code_") or
            n.startswith("num_") or
            re.fullmatch(r"(id|code|num(?:ero)?)", n or "") is not None
        )
    metric_candidates = [c for c in numeric_cols if not looks_like_id(c)]

    # Choix du y_field via alias, sinon 1er numérique non-ID
    y_field = None
    for group in y_alias_groups:
        col = has_col(*group)
        if col and col in metric_candidates:
            y_field = col
            break
    if not y_field:
        # fallback : s'il y a 'ca' présent mais filtré par l'heuristique, autoriser
        ca_like = has_col("ca", "chiffre_affaires")
        if ca_like and ca_like in numeric_cols:
            y_field = ca_like
        elif metric_candidates:
            y_field = metric_candidates[0]
        elif numeric_cols:
            y_field = numeric_cols[-1]  # dernier recours

    # Choix du x_field via priorité, sinon 1ère string
    x_field = None
    for pref in x_priority:
        col = has_col(pref)
        if col and col in text_cols:
            x_field = col
            break
    if not x_field:
        x_field = text_cols[0] if text_cols else cols_original[0]

    # Type de mark
    mark_type = "line" if any(k in intent for k in ["évolution", "evolution", "tendance", "over time"]) else "bar"

    # Spec Vega-Lite
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data},
        "transform": [
            {"calculate": f"isValid(datum['{x_field}']) ? datum['{x_field}'] : 'Non renseigné'", "as": f"{x_field}_label"}
        ],
        "mark": {"type": mark_type, "tooltip": True},
        "encoding": {
            "x": {"field": f"{x_field}_label", "type": "nominal", "axis": {"labelAngle": -45}},
            "y": {"field": y_field, "type": "quantitative"},
        },
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}},
    }

    if mark_type == "bar":
        spec["encoding"]["x"]["sort"] = f"-{y_field}"
        spec["width"] = 900
        spec["height"] = 450

    # Render to PNG (fallback to SVG)
    png_bytes = None
    svg_text = None
    try:
        from vl_convert import vl_convert as vlc
        try:
            png_bytes = vlc.vegalite_to_png(spec)
        except Exception:
            svg_text = vlc.vegalite_to_svg(spec)
    except Exception:
        try:
            from vl_convert import VegaLite
            try:
                png_bytes = VegaLite(spec).png()
            except Exception:
                svg_text = VegaLite(spec).svg()
        except Exception:
            pass

    # Save artifacts via ToolContext (Part); retrieve injected context
    context = kwargs.get("context") or kwargs.get("tool_context") or kwargs.get("_context")
    if context is not None:
        if png_bytes:
            img_part = genai_types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            await context.save_artifact(filename="graphique.png", artifact=img_part)
        elif svg_text:
            svg_part = genai_types.Part.from_bytes(data=svg_text.encode("utf-8"), mime_type="image/svg+xml")
            await context.save_artifact(filename="graphique.svg", artifact=svg_part)

        spec_part = genai_types.Part.from_bytes(
            data=json.dumps(spec, ensure_ascii=False, indent=2).encode("utf-8"),
            mime_type="application/json"
        )
        await context.save_artifact(filename="graphique.spec.vega-lite.json", artifact=spec_part)

    return {"text": "Voici le graphique."}



# --- Correctif pour l'Automatic Function Calling (AFC) ---
# Matérialise les annotations de chaînes en types réels pour éviter les erreurs AFC
from typing import get_type_hints

def _materialize_annotations(func):
    try:
        func.__annotations__ = get_type_hints(func)
    except Exception:
        pass
    return func

# --- Outil de rendu Vega-Lite ---
def render_vega_block(viz_out: Dict[str, Any]) -> Dict[str, str]:
    """Construit un bloc de code vega-lite à partir d'une sortie de viz_agent ou d'une spec directe.
    Accepte soit un objet contenant la clé 'vega_lite_spec', soit directement la spec (dict/str).
    Retourne {"text": "```vega-lite\n...\n```"}
    """
    try:
        spec = viz_out.get("vega_lite_spec") if isinstance(viz_out, dict) else viz_out
    except Exception:
        spec = viz_out

    # Tolérer l'ancienne version qui renvoyait une chaîne JSON
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            # Si ce n'est pas du JSON valide, on l'enveloppe dans un objet minimal
            spec = {"_raw": spec}

    text = "```vega-lite\n" + json.dumps(spec, ensure_ascii=False, indent=2) + "\n```"
    return {"text": text}

# Matérialiser les annotations des fonctions exposées comme tools
_materialize_annotations(get_full_schema_context) 
_materialize_annotations(rag_sql_examples)
_materialize_annotations(get_full_context_for_sql)
_materialize_annotations(pick_example_sql)
_materialize_annotations(run_sql_with_examples_first)
_materialize_annotations(run_sql)
_materialize_annotations(chart_spec_tool)
_materialize_annotations(render_vega_block)

# --- Définitions des Agents ---

def _wants_chart(text: str) -> bool:
    if not text: return False
    t = text.lower()
    triggers = [
        "graph", "graphique", "courbe", "bar chart", "barre", "camembert",
        "line chart", "visualisation", "viz", "plot", "évolution", "tendance"
    ]
    return any(k in t for k in triggers)

ux_agent = LlmAgent(
    name="ux_agent",
    model=MODEL,
    description="Clarifie les questions vagues des utilisateurs.",
    instruction=(
        "Ta tâche est de reformuler la question de l'utilisateur ou de poser une question de clarification si elle est trop vague. "
        "Sois concis. Ne réponds au root_agent qu'avec la clarification."
    ),
)

metadata_agent = LlmAgent(
    name="metadata_agent",
    model=MODEL,
    description="Spécialiste du schéma de la base de données. Fournit la structure des tables, les colonnes et les relations.",
    instruction=(
        "Ta sortie sert UNIQUEMENT de contexte pour d'autres agents.\n"
        "1) Utilise STRICTEMENT l’outil `get_full_context_for_sql(question=<question>)`.\n"
        "2) Renvoie EXACTEMENT l’objet JSON retourné par l’outil à root_agent via transfer_to_agent, SANS aucun autre texte.\n"
        "3) Tu ne t’adresses JAMAIS à l’utilisateur."
    ),
    tools=[get_full_context_for_sql],
)


sql_agent = LlmAgent(
    name="sql_agent",
    model=MODEL,
    description="Génère et exécute une requête SQL en utilisant un contexte fourni.",
    instruction=(
        """Tu es un worker SQL technique.
        - Ne t’adresse JAMAIS à l’utilisateur.
        - Ta mission : générer UNE requête SELECT BigQuery, l’exécuter via l’outil `run_sql`, puis transmettre le résultat à l’agent racine.

        ENTRÉES FOURNIES
        - `question` (texte en langage naturel)
        - `get_full_context_for_sql_response` contenant :
        - `schema_details` (tables, colonnes, relations, business_concepts)
        - `examples` (requêtes SQL précédemment validées)

        PROCESSUS OBLIGATOIRE
        1) Analyse la `question` et le contexte.
        2) Génère UNE requête SQL BigQuery en appliquant la HIERARCHIE DES RÈGLES.
        3) Exécute la requête via l’outil `run_sql`.
        4) Appelle `transfer_to_agent` vers `root_agent_reine_des_maracas` avec un UNIQUE message JSON STRICT au format :
        { "agent":"sql_agent", "question": <question>, "sql": <ta_requête>, "run_sql_response": <sortie_de_run_sql> }
        5) Ne renvoie AUCUN autre texte, AUCUNE explication.


        HIERARCHIE DES RÈGLES
        PRIORITÉ 1 — EXEMPLES (RÈGLE D’OR)
        - S’il existe un `example` pertinent dans `examples`, commence par COPIER-COLLER sa requête telle quelle.
        - Ensuite SEULEMENT, adapte-la aux besoins de la question (filtres, dates, agrégations) en respectant TOUTES les règles techniques ci-dessous.
        - Ne réinvente jamais une logique complexe si un exemple est fourni.

        PRIORITÉ 2 — RÈGLES TECHNIQUES & MÉTIER
        - Qualification des tables physiques : toujours utiliser le projet/dataset complet :
        `avisia-training.reine_des_maracas.<table>` AS alias
        (Ne qualifie jamais les alias de CTE.)
        - N’emploie que les colonnes présentes dans `schema_details.tables[*].fields`.
        - Relations & clés : déduis les JOINs à partir de `schema_details.tables[*].joins`.
        - Business concepts : si la question correspond à un concept, utilise l’expression SQL fournie dans
        `business_concepts.metrics` et `business_concepts.dimensions`.



        RÈGLES DE ROBUSTESSE SQL (OBLIGATOIRES)
        1) Qualification complète :
        - TOUTE table doit être entièrement qualifiée : `avisia-training.reine_des_maracas.*`, sauf les tables alias (CTE)
        2) Jointures inclusives :
        - Utilise systématiquement `LEFT JOIN` en partant de la table de transactions (`ticket_caisse`) pour ne perdre aucune vente
        3) Filtres texte (insensibles à la casse/espaces) :
        - Utilise `UPPER(TRIM(colonne)) = 'VALEUR'`
        4) Filtres booléens (annulations) :
        - Utilise `COALESCE(t.ANNULATION, FALSE) = FALSE`
        - Et, si nécessaire, `COALESCE(t.ANNULATION_IMMEDIATE, FALSE) = FALSE`
        5) Agrégations sûres :
        - Enveloppe TOUTES les agrégations dans `COALESCE(..., 0)` pour éviter les NULL
        - Exemple : `COALESCE(SUM(t.QUANTITE), 0)` ; `COALESCE(AVG(x), 0)`
        6) Gestion des dates :
        - `DATE_TICKET` est une STRING au format `DD/MM/YYYY`
        - Toujours parser : `SAFE.PARSE_DATE('%d/%m/%Y', t.DATE_TICKET)`
        - Toute comparaison/EXTRACT sur date doit passer par ce parse
        7) Mesure principale :
        - Si tu calcules un chiffre d’affaires, alias `ca`, enveloppé par `COALESCE(..., 0)`

        CONTRAINTES DE SORTIE
        - Exécute la requête avec l’outil `run_sql`.
        - Puis appelle `transfer_to_agent` vers `root_agent_reine_des_maracas` avec EXACTEMENT :
        { "agent":"sql_agent", "question": <question>, "sql": <ta_requête>, "run_sql_response": <sortie_de_run_sql> }
        - Ne produis AUCUN autre contenu.
        """
    ),
    tools=[run_sql],
)


viz_agent = LlmAgent(
    name="viz_agent",
    model=MODEL,
    description="Génère une spec Vega-Lite à partir de données JSON.",
    instruction=(
        "Ne t'adresse JAMAIS à l'utilisateur.\n"
        "Tu reçois { data_json, user_intent }.\n"
        "1) Appelle l'outil `chart_spec(data_json=<data_json>, user_intent=<user_intent>)`.\n"
        "2) Le tool sauvegarde l'image en artifact et renvoie un court texte. Renvoie ce texte tel quel."
    ),
    tools=[chart_spec_tool],
)

root_agent = LlmAgent(
    name="root_agent_reine_des_maracas",
    model=MODEL,
    description="Orchestre les agents spécialisés pour répondre aux questions analytiques.",
    
    # --- AJOUT DU PLANNER ---
    planner=BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(
            # On active la réflexion interne du modèle pour suivre le plan
            include_thoughts=False, 
        )
    ),
    # --- FIN DE L'AJOUT ---
    
instruction=(
        "Tu es l'orchestrateur principal et le SEUL à communiquer avec l'utilisateur.\n"
        "Ton travail consiste à suivre un plan séquentiel en appelant des agents spécialisés. Tu ne dois JAMAIS afficher leurs réponses brutes, qui sont tes notes de travail internes.\n\n"
        "**PLAN D'ACTION SÉQUENTIEL OBLIGATOIRE :**\n"
        "1.  **ÉTAPE 1 : Collecte du Schéma et des Exemples SQL.**\n"
        "    - Appelle `metadata_agent` avec la question initiale pour obtenir le contexte complet.\n"
        "2.  **ÉTAPE 2 : Génération SQL.**\n"
        "    - Appelle `sql_agent` en lui passant la question initiale et ce `contexte` combinés.\n"
        "4.  **ÉTAPE 4 : Synthèse de la Réponse Finale.**\n"
        "    - Le résultat de l'étape 3 est un objet JSON. Extrais la valeur de la clé `sql_result` pour l'analyse.\n"
        "    - Si le résultat contient une erreur, explique-la simplement à l'utilisateur.\n"
        "    - Si l'utilisateur a demandé un graphique et que les données sont valides, appelle `viz_agent` avec les données de `sql_result`.\n"
        "    - Sinon, formule une réponse finale claire en langage naturel pour l'utilisateur en te basant sur les données de `sql_result`.\n\n"
        "**Cas particulier :** Si la question initiale est ambiguë, appelle `ux_agent` avant de commencer le plan."
    ),
    tools=[render_vega_block],
    sub_agents=[ux_agent, metadata_agent, sql_agent, viz_agent],
)

# --- Initialisation au démarrage ---
_initialize_components()
_initialize_sql_examples_index()