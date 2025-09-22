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
def _initialize_components():
    global _SEMANTIC_INDEX
    if VERTEXAI_AVAILABLE and not _SEMANTIC_INDEX:
        client = EmbeddingClient()
        schema = get_enriched_schema()
        _SEMANTIC_INDEX = SemanticIndex.create(schema, client)

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
        """
        Qualifie uniquement les identifiants de table après FROM/JOIN.
        Cas gérés :
          - table
          - dataset.table
          - project-with-dash.dataset.table
          - déjà backtické : `project.dataset.table`
        Ne touche pas aux fonctions ni sous-requêtes.
        """
        # Capture :
        # 1) `...` (déjà backtiqué)
        # 2) project/dataset/table alphanum + '_' + '-' (pour le project)
        #    avec 0 à 2 segments '.'
        pattern = r"\b(FROM|JOIN)\s+(?:\n|\r|\s)*(`[^`]+`|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_]+){0,2})(?!\s*\()"

        def normalize(name: str) -> str:
            if name.startswith("`") and name.endswith("`"):
                name = name[1:-1]
            return name.strip()

        def repl(m):
            kw = m.group(1)
            token = normalize(m.group(2))
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
            # fallback parano
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

    final_query = qualify_table_names(final_query)
    final_query = backtick_3part_with_project(final_query)
    final_query = _sanitize_functions(final_query)
    
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
    
def chart_spec(data_json: str, user_intent: str) -> Dict[str, Any]:
    """
    Génére une spec Vega-Lite à partir de data JSON (list[dict]) et de l'intention utilisateur.
    - Détecte x/y automatiquement (numérique vs texte/date)
    - Met 'temporal' si la colonne ressemble à une date
    - 'line' si l'intent suggère une évolution, sinon 'bar'
    Retour: { "vega_lite_spec": "<json pretty>" } ou { "spec": "<message d'erreur>" }
    """    
    try:
        data = json.loads(data_json)
        if not isinstance(data, list) or not data:
            return {"error": "Données invalides ou vides."}
    except json.JSONDecodeError:
        return {"error": "JSON invalide."}

    # Heuristiques pour nommer X/Y
    sample = data[0]
    cols = list(sample.keys())

    # Si la métrique attendue est claire dans la question
    intent = user_intent.lower()
    y_pref = None
    if "quantité" in intent:
        y_pref = "quantite_vendue"
    elif "chiffre" in intent or "ca" in intent:
        y_pref = "chiffre_affaires"

    # Choisir y: métrique numérique
    numeric_cols = [c for c in cols if isinstance(sample[c], (int, float))]
    if y_pref and y_pref in cols:
        y_field = y_pref
    else:
        y_field = numeric_cols[0] if numeric_cols else cols[1]

    # Choisir x: label catégoriel
    label_cols = [c for c in cols if isinstance(sample[c], str)]
    # quelques colonnes courantes en priorité
    for cand in ["LIBELLE_MODELE", "VILLE", "FAMILLE", "ID_MODELE"]:
        if cand in cols:
            x_field = cand
            break
    else:
        x_field = label_cols[0] if label_cols else cols[0]

    mark = "line" if any(k in intent for k in ["évolution","tendance","courbe","over time"]) else "bar"

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data},
        "transform": [
            {"calculate": f"isValid(datum['{x_field}']) ? datum['{x_field}'] : 'Non renseigné'", "as": f"{x_field}_label"}
        ],
        "mark": {"type": mark, "tooltip": True},
        "encoding": {
            "x": {"field": f"{x_field}_label", "type": "nominal", "axis": {"labelAngle": -45}},
            "y": {"field": y_field, "type": "quantitative"},
        },
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}},
    }

    # Si bar chart catégoriel, trier par la métrique
    if mark == "bar":
        spec["encoding"]["x"]["sort"] = f"-{y_field}"
        spec["width"] = 900  # confortable
        spec["height"] = 450

    return {"vega_lite_spec": spec}



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
#_materialize_annotations(find_relevant_schema)
_materialize_annotations(get_full_schema_context) 
_materialize_annotations(rag_sql_examples)
_materialize_annotations(run_sql)
_materialize_annotations(chart_spec)
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
        "Sois concis. Ne réponds qu'avec la clarification."
    ),
)

metadata_agent = LlmAgent(
    name="metadata_agent",
    model=MODEL,
    description="Trouve les tables et colonnes pertinentes pour une question.",
    instruction=(
        "Ta sortie sert UNIQUEMENT de contexte pour d'autres agents.\n"
        "1) Utilise STRICTEMENT l’outil `find_relevant_schema(question=<question>)`.\n"
        "2) Renvoie EXACTEMENT l’objet JSON retourné par l’outil, SANS aucun autre texte.\n"
        "3) Tu ne t’adresses JAMAIS à l’utilisateur."
    ),
    #tools=[find_relevant_schema],
    tools=[get_full_schema_context],
)


sql_agent = LlmAgent(
    name="sql_agent", model=MODEL,
    description="Génère et exécute une requête SQL pour répondre à une question, en se basant sur un contexte de schéma.",
    instruction=(
         "Tu es un worker technique.\n"
        "• NE T'ADRESSE JAMAIS à l'utilisateur.\n"
        "• Génère UNE requête SELECT BigQuery (règles ci-dessous), exécute-la via l'outil `run_sql`.\n"
        "• Quand tu as le résultat (ou une erreur), APPELLE `transfer_to_agent` vers "
        "`root_agent_reine_des_maracas` avec un message JSON strictement au format :\n"
        "{ \"agent\":\"sql_agent\", \"question\": <question>, \"sql\": <ta_requête>, \"run_sql_response\": <sortie_de_run_sql> }\n"
        "• Ne renvoie AUCUN autre texte.\n"
        "\n"
        "Tu es un expert SQL BigQuery.\n"
        "Tu recevras :\n"
        "- `question`\n"
        "- `get_full_schema_context_response` (avec `relevant_tables`, `schema_details`=tables/fields/joins et `business_concepts`)\n"
        "- `examples`\n"
        "**RÈGLE ABSOLUE :** Ta requête DOIT qualifier chaque nom de table avec `avisia-training.reine_des_maracas.`. Par exemple : `FROM `avisia-training.reine_des_maracas.ticket_caisse` AS t`.\n"
        "Règles strictes :\n"
        "• Utilise les *relations* de `schema_details` pour choisir les JOINs et les clés.\n"
        "• N’emploie que les colonnes présentes dans `schema_details.tables[*].fields`.\n"
        "• Si la question correspond à un concept métier, utilise `business_concepts.metrics/dimensions` (expression SQL).\n"
        "• `DATE_TICKET` est STRING au format DD/MM/YYYY → `SAFE.PARSE_DATE('%d/%m/%Y', t.DATE_TICKET)` pour tout EXTRACT/filtre.\n"
        "• Toutes les tables doivent être **entièrement qualifiées** `avisia-training.reine_des_maracas.*`.\n"
        "• Mesure principale alias `ca` et enveloppée par `COALESCE(...,0)`.\n"
        "**RÈGLES DE ROBUSTESSE SQL (OBLIGATOIRES) :**\n"
        "✅ **Qualif. complète :** Toutes les tables DOIVENT être qualifiées : `avisia-training.reine_des_maracas.nom_table`.\n"
        "✅ **Jointures Inclusives :** Utilise systématiquement `LEFT JOIN` en partant de la table de transactions (`ticket_caisse`) pour ne perdre aucune vente, même si les informations associées sont manquantes.\n"
        "✅ **Filtres sur Texte :** Pour toute comparaison de chaînes de caractères (villes, régions, etc.), rends-la insensible à la casse et aux espaces. Utilise le format `UPPER(TRIM(colonne)) = 'VALEUR_EN_MAJUSCULES'`. Exemple : `WHERE UPPER(TRIM(m.REGIONS)) = 'PARIS'`.\n"
        "✅ **Filtres sur Booléens (Annulations) :** Pour filtrer les ventes valides, gère les `NULL` potentiels sur les colonnes `ANNULATION` et `ANNULATION_IMMEDIATE`. Utilise `COALESCE(t.ANNULATION, FALSE) = FALSE` au lieu de `NOT t.ANNULATION`.\n"
        "✅ **Agrégations Sûres :** Enveloppe TOUTES les fonctions d'agrégation (`SUM`, `AVG`, etc.) avec `COALESCE(..., 0)` pour garantir un résultat numérique (`0`) au lieu de `NULL` en l'absence de données. Exemple : `COALESCE(SUM(t.QUANTITE), 0)`.\n"
        "✅ **Gestion des Dates :** `DATE_TICKET` est une STRING `DD/MM/YYYY`. Utilise toujours `SAFE.PARSE_DATE('%d/%m/%Y', t.DATE_TICKET)` pour les filtres ou les extractions de date.\n"
        "✅ **Concepts Métier :** Si la question de l'utilisateur correspond à un `business_concepts`, utilise l'expression SQL fournie."
        "Ensuite, exécute via `run_sql` et renvoie son résultat."
    ),
    tools=[rag_sql_examples, run_sql],
)


viz_agent = LlmAgent(
    name="viz_agent",
    model=MODEL,
    description="Génère une spec Vega-Lite à partir de données JSON.",
    instruction=(
        "Ne t'adresse JAMAIS à l'utilisateur.\n"
        "Tu reçois { data_json, user_intent }.\n"
        "1) Appelle l'outil `chart_spec(data_json=<data_json>, user_intent=<user_intent>)`.\n"
        "2) Retourne EXCLUSIVEMENT l'objet JSON de sortie de `chart_spec` (sans texte autour)."
    ),
    tools=[chart_spec],
)

root_agent = LlmAgent(
    name="root_agent_reine_des_maracas",
    model=MODEL,
    description=(
        "Orchestre les agents spécialisés pour répondre aux questions analytiques. "
        "Après `metadata_agent`, transfert obligatoire vers `sql_agent` via `transfer_to_agent` avec la question, la réponse de schéma et la consigne."
    ),
    instruction=(
        "Tu es le SEUL agent autorisé à parler à l'utilisateur.\n"
        "\n"
        "Flux :\n"
        "1) Appelle `metadata_agent` pour le contexte.\n"
        "2) Transfère à `sql_agent` (via `transfer_to_agent`) avec {question, find_relevant_schema_response}.\n"
        "   Important : Après `metadata_agent`, tu n’adresses JAMAIS de réponse à l’utilisateur avant le retour de `sql_agent`.\n"
        "   Une fois le transfert effectué, attends la sortie de `sql_agent`.\n"
        "3) À la réception de la réponse de `sql_agent` (via `transfer_to_agent`), si `run_sql_response.result` contient des lignes :\n"
        "   - Si la question de l'utilisateur demande un graphique (détecte des mots comme 'graphique', 'courbe', 'barre', 'visualisation', 'évolution', 'tendance'),\n"
        "     alors transfère à `viz_agent` avec { data_json: run_sql_response.result, user_intent: <question originale> }.\n"
        "   - Sinon, formate directement la réponse tabulaire pour l'utilisateur.\n"
        "4) À la réception de `viz_agent` (via `transfer_to_agent`), APPELLE l'outil `render_vega_block(viz_out=<réponse de viz_agent>)` et RENVOIE STRICTEMENT la clé `text` en sortie (sans autre texte).\n"
        "5) NE JAMAIS afficher les JSON intermédiaires (`metadata_agent`, `viz_agent` brut, etc.).\n"
        "6) En cas d'erreur dans `run_sql_response` ou `chart`, explique clairement l'erreur.\n"
        "7) En cas d'ambiguïté, tu peux appeler `ux_agent` pour clarifier la question."
    ),
    tools=[render_vega_block],
    sub_agents=[ux_agent, metadata_agent, sql_agent, viz_agent],
)

# --- Initialisation au démarrage ---
_initialize_components()