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
from typing import Any, Dict, List, Optional
from functools import lru_cache
from collections import defaultdict

import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError, BadRequest, Forbidden
from google.adk.agents.llm_agent import LlmAgent
import google.genai.types as genai_types

# --- Vérification et importation des dépendances ---
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

# --- Configuration ---
MODEL = os.getenv("DATA_MODEL", "gemini-2.5-pro")
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
EXAMPLE_MIN_SCORE = float(os.getenv("EXAMPLE_MIN_SCORE", "0.85"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.info(f"Chemin des exemples SQL utilisé : {EXAMPLES_PATH}")
logger.info(f"Le fichier existe-t-il ? {'Oui' if os.path.exists(EXAMPLES_PATH) else 'Non'}")
logger.info(f"BQ_LOCATION={os.getenv('BQ_LOCATION')!r} — DATASET={PROJECT}.{DATASET}")
# Réduire le bruit des warnings aiohttp/asyncio si des librairies externes ne ferment pas proprement
logging.getLogger("aiohttp.client").setLevel(logging.ERROR)
logging.getLogger("aiohttp.connector").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

if VERTEXAI_AVAILABLE:
    try:
        if PROJECT:
            vertexai.init(project=PROJECT, location=LOCATION)
            logger.info(f"Vertex AI SDK initialized for project '{PROJECT}' in '{LOCATION}'.")
    except Exception as e:
        logger.warning(f"Could not initialize Vertex AI SDK. Error: {e}")
        VERTEXAI_AVAILABLE = False
else:
    logger.warning("Vertex AI SDK not found. Install 'google-cloud-aiplatform' to enable semantic search.")

# --- Gestion du Schéma ---
def _bq_client() -> Optional[bigquery.Client]:
    try:
        if not PROJECT: return None
        return bigquery.Client(project=PROJECT, location=os.getenv("BQ_LOCATION"))
    except Exception as e:
        logger.warning(f"Le client BigQuery n'a pas pu être initialisé : {e}")
        return None

@lru_cache(maxsize=1)
def get_enriched_schema() -> Dict[str, Any]:
    # Option d'invalidation du cache à la volée
    if os.getenv("DISABLE_SCHEMA_CACHE") == "1":
        try:
            get_enriched_schema.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
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
            logger.info(f"Schéma chargé et enrichi pour {len(live_schema['tables'])} tables.")
        except (GoogleAPICallError, BadRequest) as e:
            logger.warning(f"Échec de la récupération du schéma live (Google API): {e}. Utilisation du schéma statique.")
            return descriptions
        except Exception as e:
            logger.warning(f"Échec de la récupération du schéma live (Exception): {e}. Utilisation du schéma statique.")
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
                logger.info(f"Chargé table_description.json depuis: {p}")
                return data
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Échec lecture {p}: {e}")
    logger.warning("Aucune description statique trouvée. Joins/concepts indisponibles.")
    return {"tables": [], "relations": [], "concepts": {}}

@lru_cache(maxsize=1)
def _load_sql_examples() -> List[Dict[str, Any]]:
    # Invalidation du cache à la volée si demandé
    if os.getenv("DISABLE_EXAMPLES_CACHE") == "1":
        try:
            _load_sql_examples.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        if os.path.exists(EXAMPLES_PATH):
            with open(EXAMPLES_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Impossible de charger les exemples SQL : {e}")
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
            logger.error(f"Échec de l'obtention des embeddings : {e}")
            return np.zeros((len(texts), 768))

# Client global unique pour éviter les "Unclosed client session"
_EMBED_CLIENT: Optional[EmbeddingClient] = None

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
        logger.info("Construction de l'index sémantique...")
        docs, metadata = [], []
        for table in schema.get("tables", []):
            for field in table.get("fields", []):
                doc = f"Table: {table.get('name', '')}. Colonne: {field.get('name', '')}. Description: {field.get('description', '')}."
                docs.append(doc)
                metadata.append({"table": table.get('name', ''), "field": field.get('name', '')})
        vectors = client.get_embeddings(docs)
        logger.info(f"Index sémantique construit avec {len(metadata)} entrées.")
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
    global _SQL_EXAMPLES_INDEX, _EMBED_CLIENT
    if not VERTEXAI_AVAILABLE:
        logger.warning("L'index des exemples SQL ne peut être initialisé car Vertex AI n'est pas dispo.")
        return

    examples = _load_sql_examples()
    if not examples:
        logger.info("Aucun exemple SQL à indexer.")
        return

    if _EMBED_CLIENT is None:
        _EMBED_CLIENT = EmbeddingClient()

    # On ne calcule les embeddings que pour les questions des exemples
    example_questions = [ex['question'] for ex in examples]
    vectors = _EMBED_CLIENT.get_embeddings(example_questions)

    # Normaliser les vecteurs une seule fois pour des calculs de similarité plus rapides
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
    
    _SQL_EXAMPLES_INDEX["examples"] = examples
    _SQL_EXAMPLES_INDEX["vectors"] = normalized_vectors # On stocke les vecteurs normalisés
    logger.info(f"{len(examples)} exemples SQL ont été chargés et vectorisés.")

def _initialize_components():
    global _SEMANTIC_INDEX, _EMBED_CLIENT
    if VERTEXAI_AVAILABLE and _EMBED_CLIENT is None:
        _EMBED_CLIENT = EmbeddingClient()
    if VERTEXAI_AVAILABLE and not _SEMANTIC_INDEX:
        schema = get_enriched_schema()
        _SEMANTIC_INDEX = SemanticIndex.create(schema, _EMBED_CLIENT)

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

    # 1bis) Forcer SAFE.PARSE_DATE si PARSE_DATE nu
    q = re.sub(r"(?<!SAFE\.)\bPARSE_DATE\s*\(", "SAFE.PARSE_DATE(", q, flags=re.IGNORECASE)

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

    # 2bis) Retire une qualification project.dataset.func( … ) si elle s'est glissée (sécurité supplémentaire)
    q = re.sub(
        rf"`?{re.escape(PROJECT)}\.`?{re.escape(DATASET)}\.`?(SAFE\.PARSE_DATE|SAFE\.PARSE_DATETIME|SAFE\.PARSE_TIME)`?\(",
        r"\1(",
        q,
        flags=re.IGNORECASE,
    )

    # 3) Backticks orphelins sur les noms de fonctions
    q = re.sub(r"`(EXTRACT|DATE|DATETIME|TIMESTAMP|SAFE)`\s*\(", r"\1(", q, flags=re.IGNORECASE)

    return q

def _wrap_date_ticket_best_effort(q: str) -> str:
    """Enveloppe DATE_TICKET par SAFE.PARSE_DATE uniquement dans des contextes de comparaison usuels.
    Évite de toucher aux alias, chaînes, ou autres occurrences non pertinentes.
    """
    # Cas 1: DATE_TICKET <op> ...  où <op> ∈ {<=, >=, !=, =, <, >}
    q = re.sub(
        r"(\bDATE_TICKET\b)\s*(<=|>=|!=|=|<|>)",
        lambda m: f"SAFE.PARSE_DATE('%d/%m/%Y', DATE_TICKET){m.group(2)}",
        q,
        flags=re.IGNORECASE,
    )

    # Cas 2: ... <op> DATE_TICKET
    q = re.sub(
        r"(<=|>=|!=|=|<|>)\s*(\bDATE_TICKET\b)",
        lambda m: f"{m.group(1)}SAFE.PARSE_DATE('%d/%m/%Y', DATE_TICKET)",
        q,
        flags=re.IGNORECASE,
    )

    # Cas 3: BETWEEN x AND DATE_TICKET
    q = re.sub(
        r"\bBETWEEN\s+([^\s]+)\s+AND\s+(\bDATE_TICKET\b)",
        lambda m: f"BETWEEN {m.group(1)} AND SAFE.PARSE_DATE('%d/%m/%Y', DATE_TICKET)",
        q,
        flags=re.IGNORECASE,
    )

    # Cas 4: DATE_TICKET BETWEEN x AND y
    q = re.sub(
        r"(\bDATE_TICKET\b)\s+BETWEEN\s+([^\s]+)\s+AND\s+([^\s]+)",
        lambda m: f"SAFE.PARSE_DATE('%d/%m/%Y', DATE_TICKET) BETWEEN {m.group(2)} AND {m.group(3)}",
        q,
        flags=re.IGNORECASE,
    )

    return q

def _retarget_project_dataset(sql: str) -> str:
    """
    Remplace les références 3-parties par le triplet actuel `PROJECT.DATASET.table`.
    - Respecte les identifiants backtickés.
    - Pour les non-backtickés, NE modifie pas l'intérieur des chaînes (simples/doubles)
      et évite les appels de fonctions de type proj.dataset.func( grâce à un lookahead simple.
    """
    # 1) Backtickés
    sql = re.sub(
        r"`([A-Za-z0-9_-]+)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)`",
        rf"`{PROJECT}.{DATASET}.\3`",
        sql,
    )
    # 2) Non backtickés, hors quotes et hors appels de fonctions proj.ds.func(
    parts, out, in_s, in_d = [], [], False, False
    i = 0
    while i < len(sql):
        c = sql[i]
        if c == "'" and not in_d:
            in_s = not in_s
            out.append(c)
            i += 1
            continue
        if c == '"' and not in_s:
            in_d = not in_d
            out.append(c)
            i += 1
            continue
        if in_s or in_d:
            out.append(c)
            i += 1
            continue
        m = re.match(r"\b([A-Za-z0-9_-]+)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b(?!\s*\()", sql[i:])
        if m:
            out.append(f"{PROJECT}.{DATASET}.{m.group(3)}")
            i += m.end()
        else:
            out.append(c)
            i += 1
    return "".join(out)

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
    if not _SQL_EXAMPLES_INDEX["examples"] or not VERTEXAI_AVAILABLE or _EMBED_CLIENT is None:
        return {"examples": []}

    # 1. Calculer l'embedding UNIQUEMENT pour la nouvelle question
    query_vec = _EMBED_CLIENT.get_embeddings([question])
    
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
            # Log quand un exemple est trouvé avec un score supérieur au seuil
            logger.info(f"Exemple trouvé avec score {score:.3f} (seuil: {similarity_threshold}) - Question: '{example.get('question', 'N/A')}, query: {example.get('sql', example.get('query', example.get('sql_query', 'N/A')))}'")
        elif score >= similarity_threshold:
            # Log également les exemples qui dépassent le seuil mais ne sont pas retenus (limite top_k atteinte)
            logger.debug(f"Exemple avec score {score:.3f} ignoré (limite top_k={top_k} atteinte)")
            break  # Comme la liste est triée, les suivants auront des scores plus faibles

    return {"examples": top_examples}

def get_full_context_for_sql(question: str) -> Dict[str, Any]:
    """
    Rassemble TOUT le contexte nécessaire pour l'agent SQL : le schéma, les exemples RAG,
    et OPTIONNELLEMENT un exemple prioritaire retargeté si le score est très élevé.
    """
    logger.info("Début de la collecte de contexte complet (schéma + exemples)...")
    
    # Appel 1: Obtenir le contexte du schéma
    schema_context = get_full_schema_context()
    
    # Appel 2: Obtenir les exemples pertinents
    examples_context = rag_sql_examples(question=question)
    
    # Appel 3: Vérifier s'il y a un exemple très pertinent à privilégier
    priority_example = None
    if examples_context.get("examples"):
        top_example = examples_context["examples"][0]
        if top_example.get("score", 0) >= EXAMPLE_MIN_SCORE:
            # Retargeter l'exemple vers le bon projet/dataset
            original_sql = top_example.get("sql") or top_example.get("query") or top_example.get("sql_query")
            if original_sql:
                retargeted_sql = _retarget_project_dataset(original_sql)
                cleaned_sql = _sanitize_functions(retargeted_sql)
                priority_example = {
                    **top_example,
                    "sql_retargeted": cleaned_sql,
                    "is_priority": True
                }
                logger.info(f"Exemple prioritaire retenu (score {top_example.get('score', 0):.3f} >= {EXAMPLE_MIN_SCORE:.2f}) et retargeté.")
    
    # Fusionner les dictionnaires en un seul objet de contexte
    full_context = {**schema_context, **examples_context}
    if priority_example:
        full_context["priority_example"] = priority_example
    
    logger.info("Contexte complet collecté.")
    return full_context


def run_sql(query: str) -> Dict[str, Any]:
    """Exécute une requête SQL SELECT-only sur BigQuery en qualifiant intelligemment les tables."""
    if not query: return {"error": "Requête vide reçue."}

    # ✅ NEW: retarget tôt toute écriture 3-parties (avec/sans backticks)
    query = _retarget_project_dataset(query)
    
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
        pat = rf"(?<!`)\b({proj}\.[A-Za-z_]\w*\.[A-Za-z_]\w*)\b(?!\s*\(|\.)"
        return re.sub(pat, r"`\1`", q)


    # Appliquer la qualification uniquement si l'agent a "oublié" le nom complet
    # On vérifie la présence du nom du projet dans la requête pour décider.

    cte_names = _extract_cte_names(final_query)       
    final_query = qualify_table_names(final_query, cte_names)
    final_query = backtick_3part_with_project(final_query)
    final_query = _wrap_date_ticket_best_effort(final_query)
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
            r"\b(FROM|JOIN)\s+(`[^`]+`|[A-Za-z0-9_.-]+)(?!\s*\()",
            _dequalify_cte,
            final_query,
            flags=re.IGNORECASE
        )
    
    client = _bq_client()
    if not client: return {"error": "Client BigQuery non disponible."}
    
    try:
        logger.info(f"Final SQL:\n{final_query}")
        job = client.query(final_query, location=os.getenv("BQ_LOCATION"))
        rows_iter = job.result(timeout=120)          # attend la fin avec un timeout
        rows       = list(rows_iter)[:1000]          # coupe côté client
        logger.info(f"Rows fetched (capped): {len(rows)}")
        cols       = [sf.name for sf in rows_iter.schema]

        if not rows: return {"result": "La requête a fonctionné mais n'a retourné aucune ligne."}
            
        # Mappe chaque ligne en dict en s'appuyant sur l'ordre des colonnes
        data = [dict(zip(cols, list(row.values()))) for row in rows]
        return {"result": json.dumps(data, indent=2, default=str)}

    except BadRequest as e:
        return {"error": f"Requête invalide BigQuery: {getattr(e, 'message', str(e))}"}
    except Forbidden as e:
        bq_loc = os.getenv("BQ_LOCATION")
        logger.error(f"Forbidden BigQuery on {PROJECT}.{DATASET} (BQ_LOCATION={bq_loc!r}): {e}")
        return {"error": f"Accès BigQuery refusé sur {PROJECT}.{DATASET} (BQ_LOCATION={bq_loc!r}). Vérifier permissions/ACLs et la région du job."}
    except Exception as e:
        logger.error(f"Échec du job SQL : {e} sur la requête : {final_query}")
        return {"error": f"Erreur lors de l'exécution de la requête : {str(e)}"}
    
async def chart_spec_tool(data_json: str, user_intent: str) -> Dict[str, Any]:
    """
    Génère UNIQUEMENT une spec Vega-Lite à partir de données JSON.
    Ne génère pas d'image et ne sauvegarde pas d'artefact.
    """
    import json
    import unicodedata

    logger.info("[viz] chart_spec_tool (spec only) called.")

    # -- Parse & garde-fous --
    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not data:
            return {"error": "Données vides pour le graphique."}
    except json.JSONDecodeError:
        return {"error": "JSON invalide."}

    sample = data[0]
    cols = list(sample.keys())
    intent = (user_intent or "").lower()

    # -- Normalisation colonnes --
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()

    col_norm_map = {_norm(c): c for c in cols}

    def has_col(*candidates: str) -> Optional[str]:
        for cand in candidates:
            real = col_norm_map.get(_norm(cand))
            if real is not None:
                return real
        return None

    def looks_like_id(name: str) -> bool:
        n = _norm(name)
        return n.startswith(("id_", "code_", "num_")) or n in {"id", "code", "num", "numero", "code_boutique"}

    numeric_cols = [c for c in cols if isinstance(sample.get(c), (int, float))]
    text_cols = [c for c in cols if isinstance(sample.get(c), str)]

    x_priority = [
        "libelle_modele", "libellé_modèle", "ville", "ville_magasin",
        "famille", "ligne", "id_modele", "id_modèle", "magasin", "boutique",
    ]

    y_alias_groups = [
        ("ca", "chiffre_affaires", "chiffre", "revenue", "sales"),
        ("quantite_vendue", "quantité", "quantite", "qty", "qte"),
        ("panier_moyen", "basket", "avg_basket"),
    ]
    if any(k in intent for k in ["chiffre", "ca", "revenue", "vente", "ventes"]):
        y_alias_groups = [y_alias_groups[0], y_alias_groups[1], y_alias_groups[2]]
    elif any(k in intent for k in ["quantité", "quantite", "qty", "qte"]):
        y_alias_groups = [y_alias_groups[1], y_alias_groups[0], y_alias_groups[2]]

    metric_candidates = [c for c in numeric_cols if not looks_like_id(c)]

    y_field = None
    for group in y_alias_groups:
        col = has_col(*group)
        if col and col in metric_candidates:
            y_field = col
            break
    if not y_field:
        ca_like = has_col("ca", "chiffre_affaires")
        if ca_like and ca_like in numeric_cols:
            y_field = ca_like
        elif metric_candidates:
            y_field = metric_candidates[0]
        elif numeric_cols:
            y_field = numeric_cols[0]
        else:
            return {"error": "Aucune métrique numérique détectée pour un graphique."}

    x_field = None
    for pref in x_priority:
        col = has_col(pref)
        if col and col in text_cols:
            x_field = col
            break
    if not x_field:
        x_field = text_cols[0] if text_cols else cols[0]

    mark_type = "line" if any(k in intent for k in ["évolution", "evolution", "tendance", "over time"]) else "bar"

    def _looks_like_date(v: str) -> bool:
        return bool(re.match(r"\d{4}-\d{2}-\d{2}$", v)) or bool(re.match(r"\d{2}/\d{2}/\d{4}$", v))

    x_type = "temporal" if (text_cols and isinstance(sample.get(x_field), str) and _looks_like_date(sample.get(x_field, ""))) else "nominal"

    calculate_expression = f"isValid(datum[\"{x_field}\"]) ? datum[\"{x_field}\"] : \"Non renseigné\""

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data},
        "transform": [
            {"calculate": calculate_expression, "as": f"{x_field}_label"}
        ],
        "mark": {"type": mark_type, "tooltip": True},
        "encoding": {
            "x": {"field": f"{x_field}_label", "type": x_type, "axis": {"labelAngle": -45}},
            "y": {"field": y_field, "type": "quantitative"},
        },
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}},
    }
    if mark_type == "bar":
        spec["encoding"]["x"]["sort"] = f"-{y_field}"
        spec["width"] = 900
        spec["height"] = 450

    # La fonction retourne maintenant UNIQUEMENT la spec.
    return {"vega_lite_spec": spec}



async def persist_viz_artifacts(spec_json: str) -> Dict[str, str]:
    """
    Prend une spec Vega-Lite, génère une image PNG, et sauvegarde la spec et l'image comme artefacts.
    """
    # Import paresseux pour éviter les imports circulaires avec services.py
    try:
        from importlib import import_module
        _services = import_module('reine_des_maracas.services')
        runner = getattr(_services, 'runner', None)
        APP_NAME = getattr(_services, 'APP_NAME', os.getenv("ADK_APP_NAME", "reine_des_maracas"))
        session_service_stateful = getattr(_services, 'session_service_stateful', None)
    except Exception as e:
        logger.error(f"Impossible d'importer les services (lazy): {e}")
        return {"text": f"Services indisponibles: {e}"}

    # On vérifie si le runner et son service d'artefacts sont bien disponibles
    if runner is None or getattr(runner, 'artifact_service', None) is None:
        msg = "Le runner global ou son service d'artefacts ne sont pas initialisés."
        logger.error(msg)
        return {"text": msg}

    try:
        import json
        # Étape 1 : Générer l'image PNG à partir de la spec
        try:
            from vl_convert import vl_convert as vlc
            spec = json.loads(spec_json)
            png_bytes = vlc.vegalite_to_png(spec)
            logger.info("[root] Image PNG générée avec succès à partir de la spec.")
        except Exception as e:
            logger.error(f"[root] Échec de la génération de l'image PNG : {e}")
            return {"text": f"Erreur lors de la génération de l'image : {e}"}

        # On récupère le service directement depuis notre instance de runner
        artifact_service = runner.artifact_service
        
        # On a besoin des IDs de la session actuelle pour sauvegarder au bon endroit.
        # On les récupère depuis le service de session.
        # Note : ceci suppose une seule session active dans l'environnement de test local.
        # C'est le comportement par défaut de adk web.
        # Étape 2 : Sauvegarder les artefacts (CORRIGÉ)
        artifact_service = runner.artifact_service
        
        # list_sessions renvoie un objet, pas une liste.
        list_sessions_response = await session_service_stateful.list_sessions(app_name=APP_NAME, user_id="user")
        
        # On accède à l'attribut .sessions pour obtenir la VRAIE liste des IDs.
        if not list_sessions_response or not list_sessions_response.sessions:
            raise RuntimeError("Aucune session active trouvée pour sauvegarder l'artefact.")
        
        # On prend le premier ID de la liste .sessions
        session_id = list_sessions_response.sessions[0]

        # Création de la Part pour la spec JSON
        spec_part = genai_types.Part.from_bytes(
            data=spec_json.encode("utf-8"),
            mime_type="application/json",
        )
        await artifact_service.save_artifact(
            app_name=APP_NAME, user_id="adk-user", session_id=session_id,
            filename="graphique.spec.vega-lite.json", artifact=spec_part
        )
        # Création et sauvegarde de la Part pour l'image PNG
        if png_bytes:
            img_part = genai_types.Part.from_bytes(
                data=png_bytes,
                mime_type="image/png",
            )
            await artifact_service.save_artifact(
                app_name=APP_NAME, user_id="adk-user", session_id=session_id,
                filename="graphique.png", artifact=img_part
            )
        
        logger.info("Artefacts sauvegardés avec succès via le service direct du runner.")
        return {"text": "Artifacts sauvegardés."}

    except Exception as e:
        logger.error(f"Échec de sauvegarde des artifacts via le service direct : {e}")
        return {"text": f"Échec de sauvegarde des artifacts: {e}"}

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
def render_vega_block(viz_out: Any) -> Dict[str, str]:
    """Construit un bloc de code Vega-Lite à partir d'une spec directe ou d'une sortie d'agent.

    Accepte:
    - un dict contenant la clé "vega_lite_spec"
    - une spec Vega-Lite directement (dict ou str JSON)
    """
    try:
        spec = viz_out.get("vega_lite_spec") if isinstance(viz_out, dict) else viz_out
        # Si None (dict sans la clé), considère le dict entier comme étant la spec
        if spec is None and isinstance(viz_out, dict):
            spec = viz_out
    except Exception:
        spec = viz_out

    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception:
            spec = {"_raw": spec}

    spec_json = json.dumps(spec, ensure_ascii=False, indent=2)
    text = f"```vega-lite\n{spec_json}\n```"

    return {"text": text}

# Matérialiser les annotations des fonctions exposées comme tools
_materialize_annotations(get_full_schema_context) 
_materialize_annotations(rag_sql_examples)
_materialize_annotations(get_full_context_for_sql)
_materialize_annotations(run_sql)
_materialize_annotations(render_vega_block)
_materialize_annotations(chart_spec_tool)
_materialize_annotations(persist_viz_artifacts)

# --- Définitions des Agents ---

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
    description="Génère et exécute une requête SQL en utilisant un contexte fourni et optionnellement un exemple de référence.",
    instruction=(
        """Tu es un worker SQL technique.
        - Ne t’adresse JAMAIS à l’utilisateur.
        - Ta mission : générer UNE requête SELECT BigQuery, l’exécuter via l’outil `run_sql`, puis transmettre le résultat à l’agent racine.

        ENTRÉES FOURNIES
        - `question` (texte en langage naturel)
        - `get_full_context_for_sql_response` contenant :
          - `schema_details` (tables, colonnes, relations, business_concepts)
          - `examples` (requêtes SQL précédemment validées)
          - OPTIONNEL : `priority_example` avec `sql_retargeted` (exemple prioritaire déjà retargeté)

        PROCESSUS OBLIGATOIRE
        1) Analyse la `question` et le contexte.
        2) Génère UNE requête SQL BigQuery en appliquant la HIERARCHIE DES RÈGLES.
        3) Exécute la requête via l’outil `run_sql`.
        4) Appelle `transfer_to_agent` vers `root_agent_reine_des_maracas` avec la syntaxe correcte :
        transfer_to_agent(agent_name='root_agent_reine_des_maracas', message='{"agent":"sql_agent", "question": <question>, "sql": <ta_requête>, "run_sql_response": <sortie_de_run_sql>, "used_priority_example": <true/false>}')
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
        - Préférence de jointure transaction ↔ référentiel : `tc.EAN = ref.EAN` (sauf contexte contraire explicite dans le schéma ou la question).
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

        ALIGNEMENT STRICT SUR EXEMPLE (si `priority_example` présent)
        - Démarre par copier-coller EXACTEMENT `priority_example.sql_retargeted` (CTE, alias, JOINs).
        - N'ajoute PAS de nouvelles tables / CTE / colonnes, SAUF:
          - filtres/conditions sur dates, texte, booléens demandé(e)s par la question ;
          - projections supplémentaires (alias) hors agrégats critiques, si et seulement si nécessaire pour répondre.
        - Conserve les gardes-fous (SAFE.PARSE_DATE, COALESCE, LEFT JOIN, qualification complète).

        CONTRAINTES DE SORTIE
        - Exécute la requête avec l’outil `run_sql`.
        - Puis appelle `transfer_to_agent` avec EXACTEMENT cette syntaxe :
          transfer_to_agent(agent_name='root_agent_reine_des_maracas', message='{"agent":"sql_agent", "question": "<question>", "sql": "<ta_requête>", "run_sql_response": <sortie_de_run_sql>, "used_priority_example": <true/false>}')
        - Le paramètre `message` doit être une CHAÎNE JSON valide (entre guillemets simples).
        - ATTENTION : N'utilise JAMAIS de noms de variables inventés ou de syntaxe Python incorrecte.
        - Ne produis AUCUN autre contenu.
        """
    ),
    tools=[run_sql],
)


root_agent = LlmAgent(
    name="root_agent_reine_des_maracas",
    model=MODEL,
    description="Orchestre les agents spécialisés pour répondre aux questions analytiques.",
    instruction=(
        "Tu es l'orchestrateur principal et le SEUL à communiquer avec l'utilisateur.\n"
        "Ton travail consiste à suivre un plan séquentiel en appelant des agents spécialisés ou tes propres outils. Tu ne dois JAMAIS afficher leurs réponses brutes.\n\n"
        "**PLAN D'ACTION SÉQUENTIEL OBLIGATOIRE :**\n"
        "1.  **ÉTAPE 1 : Collecte du Contexte Complet.**\n"
        "    - Appelle le sous-agent `metadata_agent` avec la question initiale pour obtenir le contexte complet.\n"
        "2.  **ÉTAPE 2 : Génération SQL.**\n"
        "    - Appelle le sous-agent `sql_agent` en lui passant la question initiale et le contexte de l'étape 1.\n"
        "3.  **ÉTAPE 3 : Traitement du Résultat et Visualisation.**\n"
        "    - Extrais la valeur de `run_sql_response` de la sortie de l'agent précédent.\n"
        "    - Si c'est une erreur, explique-la à l'utilisateur et arrête-toi là.\n"
        "    - Si une visualisation est demandée et que les données sont valides, **exécute directement le plan de visualisation suivant :**\n"
        "    - **3a. GÉNÉRATION DE LA SPEC : Appelle ton propre outil `chart_spec_tool` avec les données JSON de la requête SQL et l'intention de l'utilisateur pour obtenir la `vega_lite_spec`.**\n"
        "    - **3b. SAUVEGARDE DE L'ARTEFACT : Prends la `vega_lite_spec` obtenue à l'étape précédente, transforme-la en chaîne JSON, et appelle ton outil `persist_viz_artifacts` avec cette chaîne.**\n"
        "    - **3c. PRÉPARATION DE L'AFFICHAGE : Appelle ton outil `render_vega_block` avec la `vega_lite_spec` pour préparer le bloc de code à afficher dans le chat.**\n"
        "    - **3d. RÉPONSE FINALE : Combine le résultat de `render_vega_block` (le graphe) avec une explication textuelle des résultats pour formuler ta réponse finale à l'utilisateur.**\n"
        "    - Si aucune visualisation n'était demandée, formule une réponse textuelle à partir des données.\n\n"
        "**Cas particulier :** Si la question initiale est ambiguë, appelle le sous-agent `ux_agent` avant de commencer le plan."
    ),
    # Le root_agent a désormais tous les outils de visualisation nécessaires
    tools=[chart_spec_tool, render_vega_block, persist_viz_artifacts],
    # viz_agent supprimé de la liste des sous-agents
    sub_agents=[ux_agent, metadata_agent, sql_agent],
)

# --- Initialisation au démarrage ---
_initialize_components()
_initialize_sql_examples_index()
