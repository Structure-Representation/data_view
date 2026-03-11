import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from multi_ab import (
    _build_subordinate_parent_lookup,
    _load_llm_json,
    _node_text,
    _normalize_cluster_list,
    _read_json,
    _render_prompt,
    _safe_int,
    _split_abstraction_text,
    _write_json,
    annotate_abstractions,
    build_level_entries,
    identify_subordinate_subcategory_clusters,
    annotate_subordinate_nodes_by_basic_cluster,
    process_annotated_data,
)
from tree_builder import Edge, Node, Tree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini_function import embed_texts  # type: ignore


DEFAULT_ORIGINAL_FILE_PATH = "multi_ab/gemini_abstract/0_ab_with_labels.json"
DEFAULT_DESIGN_PROBLEM = "clean laundry with less water"
DEFAULT_VARIANT_OUTPUT_BASE = "multi_ab/gemini_abstract/GPT/tree_pipeline_variants"


COARSE_BASIC_GLOBAL_PROMPT = """
## Task
Identify a small set of coarse first-layer clusters over mechanisms for solving the given problem.

## Context
Design problem: {design_problem}

## Requirements
- The given mechanisms are functional components of solutions for the given problem.
- Make the first-layer categories as coarse as possible while still separating clearly different mechanism families.
- Avoid implementation-specific clusters.
- Treat support/control/infrastructure patterns as separate only if they form a meaningful recurring family.

## Input
Basic node profiles:
{profiles}

## Output
<json>
{{
  "level_name": "level1_basic",
  "clusters": [
    {{
      "id": 0,
      "cluster_name": "cluster name",
      "description": "brief description(less than 20 words)"
    }}
  ]
}}
</json>
"""


BASIC_SUPER_GUIDED_PROMPT = """
## Task
Identify a small set of coarse first-layer clusters over BASIC-level mechanism for solving the given problem.

## Context
Design problem: {design_problem}

## Requirements
- The given mechanisms are functional components of solutions for the design problem. The superordinate nodes are the broad mechanism families that the basic-level mechanisms belong to.
- Focus primarily on BASIC-level mechanism similarity. Use linked SUPERORDINATE context just for reference.
- Make the clusters intentionally coarse.
- Do not let superordinate labels override clear differences in the basic mechanisms themselves.

## Input
Basic node profiles:
{profiles}

## Output
<json>
{{
  "level_name": "level1_basic_super_guided",
  "clusters": [
    {{
      "id": 0,
      "cluster_name": "cluster name",
      "description": "brief description(less than 20 words)"
    }}
  ]
}}
</json>
"""


BASIC_LEVEL2_WITH_SUB_PROMPT = """
## Task
Sub-cluster BASIC-level mechanisms that already belong to the same coarse parent cluster. The mechanisms are functional components of solutions for the given problem.

## Context
Design problem: {design_problem}
Parent cluster: {parent_cluster_name}: {parent_cluster_description}

## Requirements
- Produce a second-layer split over BASIC nodes.
- Use subordinate evidence to sharpen the split.
- Keep clusters mechanism-centric, not purely on specific implementation details.
- Check the cluster again after the first pass to ensure the clusters are not too fine-grained.

## Input
Profiles:
{profiles}

## Output
<json>
{{
  "parent_cluster_name": "{parent_cluster_name}",
  "sub_clusters": [
    {{
      "id": 0,
      "cluster_name": "sub-cluster name",
      "description": "brief description"
    }}
  ]
}}
</json>
"""


BASIC_LEVEL2_BASIC_ONLY_PROMPT = """
## Task
Sub-cluster BASIC-level mechanisms that already belong to the same coarse parent cluster. The mechanisms are functional components of solutions for the given problem.

## Context
Design problem: {design_problem}
Parent cluster: {parent_cluster_name}: {parent_cluster_description}

## Requirements
- Use BASIC-level abstraction text only.
- Keep this layer mechanism-centric and cleaner than the first coarse split.
- Check the cluster again after the first pass to ensure the clusters are not too fine-grained.

## Input
Profiles:
{profiles}

## Output
<json>
{{
  "parent_cluster_name": "{parent_cluster_name}",
  "sub_clusters": [
    {{
      "id": 0,
      "cluster_name": "sub-cluster name",
      "description": "brief description"
    }}
  ]
}}
</json>
"""


CLUSTER_SUMMARY_PROMPT = """
## Task
Summarize the shared mechanism theme of this cluster. The mechanisms are functional components of solutions for the given problem.

## Context
Design problem: {design_problem}

## Input
Cluster member profiles:
{profiles}

## Requirements
- Focus on the structural function instead of specific implementation details.
- The themes should be different from each other and also has a proper abstraction level.

## Output
<json>
{{
  "cluster_name": "cluster theme",
  "description": "brief summary of the shared mechanism"
}}
</json>
"""


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _basic_entry_for_prompt(entry: Dict[str, Any], use_super: bool, use_sub: bool) -> Dict[str, Any]:
    profile = {
        "idea_id": entry["idea_id"],
        "abstraction_id": entry["abstraction_id"],
        "abstraction": entry["abstraction"],
        "is_core": entry.get("is_core", False),
    }
    if use_super:
        profile["parent_superordinate_node"] = entry.get("parent_superordinate_node", "")
    if use_sub:
        profile["child_subordinate_evidence"] = entry.get("child_subordinate_evidence", [])
    return profile


def build_basic_entries_for_prompt(
    original_file_path: str,
    use_super: bool = False,
    use_sub: bool = False,
) -> List[Dict[str, Any]]:
    entries = build_level_entries(original_file_path, level="basic")
    return [_basic_entry_for_prompt(entry, use_super=use_super, use_sub=use_sub) for entry in entries]


def identify_global_clusters_from_profiles(
    profiles: List[Dict[str, Any]],
    output_path: str,
    prompt_template: str,
    level_name: str,
    design_problem: str,
) -> Dict[str, Any]:
    if os.path.exists(output_path):
        existing = _read_json(output_path)
        if isinstance(existing, dict):
            return existing

    prompt = _render_prompt(
        prompt_template,
        design_problem=design_problem,
        profiles=json.dumps(profiles, ensure_ascii=False, indent=2),
    )
    result = _load_llm_json(prompt)
    clusters = result.get("clusters", []) if isinstance(result, dict) else []
    normalized = []
    for idx, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            continue
        item = dict(cluster)
        item.setdefault("id", idx)
        normalized.append(item)

    payload = {"level_name": level_name, "clusters": normalized}
    _write_json(payload, output_path)
    return payload


def _cluster_lookup_from_payload(cluster_payload: Any) -> Dict[str, Dict[str, Any]]:
    items = _normalize_cluster_list(cluster_payload)
    return {str(item.get("id")): item for item in items}


def _load_assignment_map(processed_path: str) -> Dict[Tuple[int, str], str]:
    data = _read_json(processed_path)
    assignment_map: Dict[Tuple[int, str], str] = {}
    for row in data:
        iid = _safe_int(row.get("idea_id"))
        abstraction_id = row.get("abstraction_id")
        assigned = row.get("assigned_cluster")
        if iid is None or abstraction_id is None or assigned is None:
            continue
        assignment_map[(iid, str(abstraction_id))] = str(assigned)
    return assignment_map


def group_basic_profiles_by_parent_assignment(
    original_file_path: str,
    assignment_processed_path: str,
    parent_cluster_payload: Any,
    use_super: bool = False,
    use_sub: bool = False,
) -> List[Dict[str, Any]]:
    parent_lookup = _cluster_lookup_from_payload(parent_cluster_payload)
    assignment_map = _load_assignment_map(assignment_processed_path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for entry in build_level_entries(original_file_path, level="basic"):
        key = (_safe_int(entry.get("idea_id")), str(entry.get("abstraction_id")))
        if key[0] is None:
            continue
        parent_cluster_id = assignment_map.get((key[0], key[1]))
        if parent_cluster_id is None:
            continue
        groups[parent_cluster_id].append(
            _basic_entry_for_prompt(entry, use_super=use_super, use_sub=use_sub)
        )

    results = []
    for parent_cluster_id, profiles in groups.items():
        parent = parent_lookup.get(parent_cluster_id, {})
        results.append(
            {
                "parent_cluster_id": parent_cluster_id,
                "parent_cluster_name": parent.get("cluster_name", str(parent_cluster_id)),
                "parent_cluster_description": parent.get("description", ""),
                "profiles": profiles,
            }
        )
    return results


def identify_grouped_basic_subclusters(
    groups: List[Dict[str, Any]],
    output_path: str,
    prompt_template: str,
    design_problem: str,
    id_prefix_template: str,
) -> Dict[str, Any]:
    if os.path.exists(output_path):
        existing = _read_json(output_path)
        if isinstance(existing, dict):
            return existing

    output_groups = []
    for group in groups:
        profiles = group["profiles"]
        if not profiles:
            continue

        if len(profiles) == 1:
            only_profile = profiles[0]
            name, description = _split_abstraction_text(only_profile.get("abstraction", ""))
            sub_clusters = [
                {
                    "id": id_prefix_template.format(parent_cluster_id=group["parent_cluster_id"], local_id="singleton"),
                    "local_id": "singleton",
                    "cluster_name": name or "singleton_cluster",
                    "description": description,
                }
            ]
        else:
            prompt = _render_prompt(
                prompt_template,
                design_problem=design_problem,
                parent_cluster_name=group["parent_cluster_name"],
                parent_cluster_description=group["parent_cluster_description"],
                profiles=json.dumps(profiles, ensure_ascii=False, indent=2),
            )
            result = _load_llm_json(prompt)
            raw_sub_clusters = result.get("sub_clusters", []) if isinstance(result, dict) else []
            sub_clusters = []
            for idx, cluster in enumerate(raw_sub_clusters):
                if not isinstance(cluster, dict):
                    continue
                local_id = cluster.get("id", idx)
                sub_clusters.append(
                    {
                        "id": id_prefix_template.format(
                            parent_cluster_id=group["parent_cluster_id"],
                            local_id=local_id,
                        ),
                        "local_id": local_id,
                        "cluster_name": cluster.get("cluster_name", ""),
                        "description": cluster.get("description", ""),
                    }
                )

        output_groups.append(
            {
                "level_name": "grouped_basic",
                "parent_cluster_id": group["parent_cluster_id"],
                "parent_cluster_name": group["parent_cluster_name"],
                "parent_cluster_description": group["parent_cluster_description"],
                "sub_clusters": sub_clusters,
            }
        )

    payload = {"level_name": "grouped_basic", "groups": output_groups}
    _write_json(payload, output_path)
    return payload


def annotate_basic_entries_with_cluster_file(
    original_file_path: str,
    entries: List[Dict[str, Any]],
    cluster_file_path: str,
    raw_output_path: str,
    processed_output_path: str,
    design_problem: str,
    definition_of_abstraction: str,
    batch_size: int = 20,
) -> List[Dict[str, Any]]:
    annotate_abstractions(
        original_file_path=original_file_path,
        output_path=raw_output_path,
        design_problem=design_problem,
        level="basic",
        definition_of_abstraction=definition_of_abstraction,
        batch_size=batch_size,
        cluster_file_path=cluster_file_path,
        entries=entries,
    )
    return process_annotated_data(
        original_file_path=original_file_path,
        annotated_file_path=raw_output_path,
        output_path=processed_output_path,
        level="basic",
        cluster_file_path=cluster_file_path,
    )


def run_llm_subordinate_stage(
    original_file_path: str,
    level2_cluster_file_path: str,
    level2_processed_path: str,
    level3_cluster_file_path: str,
    level3_raw_output_path: str,
    level3_processed_output_path: str,
    design_problem: str,
    batch_size: int = 20,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    identify_subordinate_subcategory_clusters(
        original_file_path=original_file_path,
        basic_subcluster_file_path=level2_cluster_file_path,
        basic_annotated_processed_path=level2_processed_path,
        output_path=level3_cluster_file_path,
        design_problem=design_problem,
    )
    processed = annotate_subordinate_nodes_by_basic_cluster(
        original_file_path=original_file_path,
        basic_annotated_processed_path=level2_processed_path,
        subordinate_subcluster_file_path=level3_cluster_file_path,
        output_path=level3_raw_output_path,
        processed_output_path=level3_processed_output_path,
        design_problem=design_problem,
        batch_size=batch_size,
    )
    return _read_json(level3_cluster_file_path), processed


def compute_text_embeddings(
    texts: Sequence[str],
    embedding_backend: str = "tfidf",
    embedding_model_name: str = "gemini",
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=float)

    if embedding_backend == "gemini":
        return np.array(embed_texts(list(texts), model_name=embedding_model_name))

    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    n_components = max(2, min(128, matrix.shape[0] - 1, matrix.shape[1] - 1))
    if n_components >= 2:
        dense = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(matrix)
        dense = Normalizer(copy=False).fit_transform(dense)
        return dense
    return matrix.toarray()


def choose_k_by_elbow(embeddings: np.ndarray, min_k: int = 2, max_k: int = 12) -> int:
    n = len(embeddings)
    if n <= 2:
        return 1

    min_k = max(2, min_k)
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        return 1

    candidate_ks = list(range(min_k, max_k + 1))
    inertias = []
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(embeddings)
        inertias.append(float(model.inertia_))

    x1, y1 = candidate_ks[0], inertias[0]
    x2, y2 = candidate_ks[-1], inertias[-1]
    denom = math.hypot(x2 - x1, y2 - y1)
    if denom == 0:
        return candidate_ks[0]

    best_k = candidate_ks[0]
    best_dist = -1.0
    for k, inertia in zip(candidate_ks, inertias):
        dist = abs((y2 - y1) * k - (x2 - x1) * inertia + x2 * y1 - y2 * x1) / denom
        if dist > best_dist:
            best_dist = dist
            best_k = k
    return best_k


def cluster_embeddings(embeddings: np.ndarray, k: int) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1 or k <= 1:
        return np.zeros(len(embeddings), dtype=int)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(embeddings)


def choose_k_by_agglomerative(embeddings: np.ndarray, min_k: int = 2, max_k: int = 12) -> int:
    n = len(embeddings)
    if n <= 2:
        return 1

    min_k = max(2, min_k)
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        return 1

    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(embeddings, labels))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_embeddings_agglomerative(embeddings: np.ndarray, k: int) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) == 1 or k <= 1:
        return np.zeros(len(embeddings), dtype=int)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(embeddings)


def choose_cluster_count(
    embeddings: np.ndarray,
    clustering_method: str,
    min_k: int = 2,
    max_k: int = 12,
) -> int:
    if clustering_method == "agglomerative":
        return choose_k_by_agglomerative(embeddings, min_k=min_k, max_k=max_k)
    return choose_k_by_elbow(embeddings, min_k=min_k, max_k=max_k)


def cluster_with_method(
    embeddings: np.ndarray,
    k: int,
    clustering_method: str,
) -> np.ndarray:
    if clustering_method == "agglomerative":
        return cluster_embeddings_agglomerative(embeddings, k=k)
    return cluster_embeddings(embeddings, k=k)


def summarize_embedding_clusters_with_llm(
    clusters_to_profiles: Dict[int, List[Dict[str, Any]]],
    design_problem: str,
) -> Dict[int, Dict[str, str]]:
    summaries: Dict[int, Dict[str, str]] = {}
    for cluster_id, profiles in clusters_to_profiles.items():
        prompt = _render_prompt(
            CLUSTER_SUMMARY_PROMPT,
            design_problem=design_problem,
            profiles=json.dumps(profiles[:20], ensure_ascii=False, indent=2),
        )
        result = _load_llm_json(prompt)
        summaries[cluster_id] = {
            "cluster_name": result.get("cluster_name", f"cluster_{cluster_id}") if isinstance(result, dict) else f"cluster_{cluster_id}",
            "description": result.get("description", "") if isinstance(result, dict) else "",
        }
    return summaries


def write_direct_assignments(
    assignments: List[Dict[str, Any]],
    raw_output_path: str,
) -> None:
    _write_json(assignments, raw_output_path)


def process_direct_assignments(
    original_file_path: str,
    raw_output_path: str,
    processed_output_path: str,
    level: str,
    cluster_file_path: str,
) -> List[Dict[str, Any]]:
    return process_annotated_data(
        original_file_path=original_file_path,
        annotated_file_path=raw_output_path,
        output_path=processed_output_path,
        level=level,
        cluster_file_path=cluster_file_path,
    )


def build_embedding_level1_payload(
    profiles: List[Dict[str, Any]],
    labels: np.ndarray,
    output_path: str,
    design_problem: str,
    summarize_with_llm: bool,
) -> Dict[str, Any]:
    if os.path.exists(output_path):
        existing = _read_json(output_path)
        if isinstance(existing, dict):
            return existing

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for profile, label in zip(profiles, labels):
        grouped[int(label)].append(profile)

    if summarize_with_llm:
        summaries = summarize_embedding_clusters_with_llm(grouped, design_problem=design_problem)
    else:
        summaries = {
            cid: {
                "cluster_name": f"embedding_cluster_{cid}",
                "description": f"Embedding-derived cluster {cid}",
            }
            for cid in grouped
        }

    clusters = []
    for cid in sorted(grouped):
        summary = summaries[cid]
        clusters.append(
            {
                "id": cid,
                "cluster_name": summary["cluster_name"],
                "description": summary["description"],
            }
        )

    payload = {"level_name": "embedding_level1", "clusters": clusters}
    _write_json(payload, output_path)
    return payload


def collect_subordinate_entries_by_basic_assignment(
    original_file_path: str,
    basic_assignment_processed_path: str,
) -> List[Dict[str, Any]]:
    data = _read_json(original_file_path)
    basic_assignment = _load_assignment_map(basic_assignment_processed_path)
    entries: List[Dict[str, Any]] = []

    for item in data:
        idea_id = _safe_int(item.get("id"))
        if idea_id is None:
            continue
        abstraction = item.get("abstraction", {})
        basic_lookup = {str(b.get("id")): b for b in abstraction.get("basic", [])}
        subordinate_parent_lookup = _build_subordinate_parent_lookup(abstraction)

        for subordinate_node in abstraction.get("subordinate", []):
            sub_id = str(subordinate_node.get("id"))
            parent_basic_id = subordinate_parent_lookup.get(sub_id)
            if parent_basic_id is None:
                continue
            parent_cluster = basic_assignment.get((idea_id, str(parent_basic_id)))
            if parent_cluster is None:
                continue

            entries.append(
                {
                    "idea_id": idea_id,
                    "abstraction_id": sub_id,
                    "abstraction": _node_text(subordinate_node),
                    "parent_basic_node": _node_text(basic_lookup.get(str(parent_basic_id), {})),
                    "parent_basic_abstraction_id": str(parent_basic_id),
                    "parent_cluster_id": str(parent_cluster),
                    "level": "subordinate",
                    "idea": item.get("idea", ""),
                }
            )
    return entries


def identify_embedding_subclusters_from_subordinate_entries(
    entries: List[Dict[str, Any]],
    parent_clusters_payload: Any,
    output_path: str,
    embedding_backend: str = "tfidf",
    embedding_model_name: str = "gemini",
    clustering_method: str = "kmeans",
) -> Dict[str, Any]:
    if os.path.exists(output_path):
        existing = _read_json(output_path)
        if isinstance(existing, dict):
            return existing

    parent_lookup = _cluster_lookup_from_payload(parent_clusters_payload)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["parent_cluster_id"])].append(entry)

    output_groups = []
    for parent_cluster_id, group_entries in grouped.items():
        texts = [f"{e['abstraction']} || parent_basic={e.get('parent_basic_node', '')}" for e in group_entries]
        embeddings = compute_text_embeddings(
            texts,
            embedding_backend=embedding_backend,
            embedding_model_name=embedding_model_name,
        )
        k = choose_cluster_count(
            embeddings,
            clustering_method=clustering_method,
            min_k=2,
            max_k=min(8, len(group_entries) - 1),
        )
        labels = cluster_with_method(embeddings, k=k, clustering_method=clustering_method)

        sub_clusters = []
        for cid in sorted(set(int(x) for x in labels.tolist())):
            sub_clusters.append(
                {
                    "id": f"{parent_cluster_id}::subordinate::{cid}",
                    "local_id": cid,
                    "cluster_name": f"embedding_subordinate_{cid}",
                    "description": "Embedding-derived subordinate cluster",
                }
            )

        parent = parent_lookup.get(parent_cluster_id, {})
        output_groups.append(
            {
                "level_name": "embedding_subordinate",
                "parent_cluster_id": parent_cluster_id,
                "parent_cluster_name": parent.get("cluster_name", str(parent_cluster_id)),
                "parent_cluster_description": parent.get("description", ""),
                "sub_clusters": sub_clusters,
            }
        )

    payload = {"level_name": "embedding_subordinate", "groups": output_groups}
    _write_json(payload, output_path)
    return payload


def write_subordinate_embedding_assignments(
    entries: List[Dict[str, Any]],
    subordinate_cluster_file_path: str,
    raw_output_path: str,
    processed_output_path: str,
    original_file_path: str,
    embedding_backend: str = "tfidf",
    embedding_model_name: str = "gemini",
    clustering_method: str = "kmeans",
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["parent_cluster_id"])].append(entry)

    assignments = []
    for parent_cluster_id, group_entries in grouped.items():
        texts = [f"{e['abstraction']} || parent_basic={e.get('parent_basic_node', '')}" for e in group_entries]
        embeddings = compute_text_embeddings(
            texts,
            embedding_backend=embedding_backend,
            embedding_model_name=embedding_model_name,
        )
        k = choose_cluster_count(
            embeddings,
            clustering_method=clustering_method,
            min_k=2,
            max_k=min(8, len(group_entries) - 1),
        )
        labels = cluster_with_method(embeddings, k=k, clustering_method=clustering_method)
        for entry, label in zip(group_entries, labels):
            assignments.append(
                {
                    "idea_id": entry["idea_id"],
                    "abstraction_id": entry["abstraction_id"],
                    "assigned_cluster": f"{parent_cluster_id}::subordinate::{int(label)}",
                    "explanation": "Assigned by embedding-based subordinate clustering.",
                }
            )

    write_direct_assignments(assignments, raw_output_path)
    return process_direct_assignments(
        original_file_path=original_file_path,
        raw_output_path=raw_output_path,
        processed_output_path=processed_output_path,
        level="subordinate",
        cluster_file_path=subordinate_cluster_file_path,
    )


def _collect_idea_refs(items: List[Dict[str, Any]], idea_lookup: Dict[Any, str]) -> Tuple[List[Any], List[str], List[Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        iid = item.get("idea_id")
        if iid is None:
            continue
        key = str(iid)
        if key not in by_id:
            by_id[key] = {"idea_id": iid, "idea_text": idea_lookup.get(iid, "")}

    refs = sorted(by_id.values(), key=lambda x: str(x["idea_id"]))
    return [x["idea_id"] for x in refs], [x["idea_text"] for x in refs], refs


def _member_record(item: Dict[str, Any], idea_lookup: Dict[Any, str]) -> Dict[str, Any]:
    label, description = _split_abstraction_text(item.get("abstraction", ""))
    return {
        "idea_id": item.get("idea_id"),
        "idea_text": idea_lookup.get(item.get("idea_id"), ""),
        "abstraction_id": item.get("abstraction_id"),
        "abstraction": item.get("abstraction", ""),
        "label": label,
        "description": description,
        "level": item.get("level", ""),
        "is_core": item.get("is_core", False),
        "explanation": item.get("explanation", ""),
        "parent": item.get("parent"),
        "children": item.get("children", []),
    }


def build_variant_tree(
    original_file_path: str,
    output_path: str,
    level1_cluster_payload: Dict[str, Any],
    level1_processed: List[Dict[str, Any]],
    level2_cluster_payload: Optional[Dict[str, Any]] = None,
    level2_processed: Optional[List[Dict[str, Any]]] = None,
    level3_cluster_payload: Optional[Dict[str, Any]] = None,
    level3_processed: Optional[List[Dict[str, Any]]] = None,
    root_name: str = "Water-Efficient Laundry Cleaning",
    root_description: str = DEFAULT_DESIGN_PROBLEM,
) -> Tree:
    original_data = _read_json(original_file_path)
    idea_lookup = {item.get("id"): item.get("idea", "") for item in original_data}

    tree = Tree()
    next_node_id = 0

    all_idea_ids = sorted(idea_lookup.keys(), key=lambda x: str(x))
    all_idea_texts = [idea_lookup[i] for i in all_idea_ids]
    root_id = next_node_id
    tree.nodes.append(
        Node(
            tree_node_id=root_id,
            name=root_name,
            description=root_description,
            type="root",
            idea_id=all_idea_ids,
            idea_text=all_idea_texts,
            idea_ids=all_idea_ids,
            idea_texts=all_idea_texts,
            property={
                "level": "root",
                "associated_ideas": [{"idea_id": i, "idea_text": idea_lookup[i]} for i in all_idea_ids],
            },
        )
    )
    next_node_id += 1

    level1_nodes: Dict[str, int] = {}
    for cluster in level1_cluster_payload.get("clusters", []):
        cid = str(cluster.get("id"))
        members = [row for row in level1_processed if str(row.get("assigned_cluster")) == cid]
        idea_ids, idea_texts, refs = _collect_idea_refs(members, idea_lookup)
        included = [_member_record(row, idea_lookup) for row in members]
        node_id = next_node_id
        next_node_id += 1
        level1_nodes[cid] = node_id
        tree.nodes.append(
            Node(
                tree_node_id=node_id,
                name=cluster.get("cluster_name", cid),
                description=cluster.get("description", ""),
                type="level1_cluster",
                idea_id=idea_ids,
                idea_text=idea_texts,
                idea_ids=idea_ids,
                idea_texts=idea_texts,
                property={
                    "level": "level1_cluster",
                    "original_id": cid,
                    "associated_ideas": refs,
                    "included_members": included,
                },
            )
        )
        tree.edges.append(Edge(from_node_id=root_id, to_node_id=node_id))

    level2_nodes: Dict[str, int] = {}
    if level2_cluster_payload and level2_processed is not None:
        for group in level2_cluster_payload.get("groups", []):
            parent_cluster_id = str(group.get("parent_cluster_id"))
            parent_node_id = level1_nodes.get(parent_cluster_id)
            if parent_node_id is None:
                continue
            for cluster in group.get("sub_clusters", []):
                cid = str(cluster.get("id"))
                members = [row for row in level2_processed if str(row.get("assigned_cluster")) == cid]
                idea_ids, idea_texts, refs = _collect_idea_refs(members, idea_lookup)
                included = [_member_record(row, idea_lookup) for row in members]
                node_id = next_node_id
                next_node_id += 1
                level2_nodes[cid] = node_id
                tree.nodes.append(
                    Node(
                        tree_node_id=node_id,
                        name=cluster.get("cluster_name", cid),
                        description=cluster.get("description", ""),
                        type="level2_cluster",
                        idea_id=idea_ids,
                        idea_text=idea_texts,
                        idea_ids=idea_ids,
                        idea_texts=idea_texts,
                        property={
                            "level": "level2_cluster",
                            "original_id": cid,
                            "parent_cluster_id": parent_cluster_id,
                            "associated_ideas": refs,
                            "included_members": included,
                        },
                    )
                )
                tree.edges.append(Edge(from_node_id=parent_node_id, to_node_id=node_id))

    if level3_cluster_payload and level3_processed is not None:
        for group in level3_cluster_payload.get("groups", []):
            parent_cluster_id = str(group.get("parent_cluster_id"))
            parent_node_id = level2_nodes.get(parent_cluster_id, level1_nodes.get(parent_cluster_id))
            if parent_node_id is None:
                continue
            for cluster in group.get("sub_clusters", []):
                cid = str(cluster.get("id"))
                members = [row for row in level3_processed if str(row.get("assigned_cluster")) == cid]
                idea_ids, idea_texts, refs = _collect_idea_refs(members, idea_lookup)
                included = [_member_record(row, idea_lookup) for row in members]
                node_id = next_node_id
                next_node_id += 1
                tree.nodes.append(
                    Node(
                        tree_node_id=node_id,
                        name=cluster.get("cluster_name", cid),
                        description=cluster.get("description", ""),
                        type="level3_cluster",
                        idea_id=idea_ids,
                        idea_text=idea_texts,
                        idea_ids=idea_ids,
                        idea_texts=idea_texts,
                        property={
                            "level": "level3_cluster",
                            "original_id": cid,
                            "parent_cluster_id": parent_cluster_id,
                            "associated_ideas": refs,
                            "included_members": included,
                        },
                    )
                )
                tree.edges.append(Edge(from_node_id=parent_node_id, to_node_id=node_id))

    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)
    return tree


def build_embedding_level1_assignments(
    profiles: List[Dict[str, Any]],
    raw_output_path: str,
    cluster_file_path: str,
    processed_output_path: str,
    original_file_path: str,
    design_problem: str,
    summarize_with_llm: bool = False,
    embedding_backend: str = "tfidf",
    embedding_model_name: str = "gemini",
    clustering_method: str = "kmeans",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    texts = [profile["abstraction"] for profile in profiles]
    embeddings = compute_text_embeddings(
        texts,
        embedding_backend=embedding_backend,
        embedding_model_name=embedding_model_name,
    )
    k = choose_cluster_count(
        embeddings,
        clustering_method=clustering_method,
        min_k=2,
        max_k=min(12, len(profiles) - 1),
    )
    labels = cluster_with_method(embeddings, k=k, clustering_method=clustering_method)
    cluster_payload = build_embedding_level1_payload(
        profiles=profiles,
        labels=labels,
        output_path=cluster_file_path,
        design_problem=design_problem,
        summarize_with_llm=summarize_with_llm,
    )

    assignments = []
    for profile, label in zip(profiles, labels):
        assignments.append(
            {
                "idea_id": profile["idea_id"],
                "abstraction_id": profile["abstraction_id"],
                "assigned_cluster": int(label),
                "explanation": "Assigned by embedding-based first-layer clustering.",
            }
        )

    write_direct_assignments(assignments, raw_output_path)
    processed = process_direct_assignments(
        original_file_path=original_file_path,
        raw_output_path=raw_output_path,
        processed_output_path=processed_output_path,
        level="basic",
        cluster_file_path=cluster_file_path,
    )
    return cluster_payload, processed
