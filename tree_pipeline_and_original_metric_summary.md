# Tree Building Pipeline And Original Metric Summary

## Scope

This repo contains two related but distinct pieces of logic:

1. The **original tree-building pipeline** that builds the canonical mechanism tree with the schema `root -> superordinate_cluster -> basic_cluster -> subordinate_cluster`.
2. A newer family of **tree-pipeline variants** that rebuild the hierarchy with alternative first-layer clustering strategies.
3. The **original routed benchmark metric** that scores idea similarity on top of the original tree schema.

The main source files are:

- `mechanism_tree_pipeline.py` for the original pipeline.
- `tree_pipeline_variant_*.py` plus `tree_pipeline_variant_utils.py` for the variant pipelines.
- `tree_builder.py` for the original tree data structure and baseline tree distances.
- `core_single_purpose_basic_routed_emd_greedy_sweep.py` for the original routed metric used in the benchmark sweep.

## 1. Original Tree-Building Pipeline

The original pipeline is implemented in `run_mechanism_tree_pipeline()` in `mechanism_tree_pipeline.py`.

### Input

It starts from the original abstraction JSON, where each idea already has abstraction nodes at multiple levels: superordinate, basic, and subordinate.

### Stages

The pipeline is a staged LM-assisted clustering/assignment process:

1. **Identify superordinate clusters**
   - Group superordinate abstractions into top-level mechanism families.
2. **Annotate superordinate nodes**
   - Assign each superordinate abstraction to one of those clusters.
   - Save both raw LM output and processed assignments.
3. **Identify basic subclusters inside each superordinate cluster**
   - For each superordinate family, split its basic abstractions into finer clusters.
4. **Annotate basic nodes**
   - Assign each basic abstraction to a basic cluster under its parent superordinate family.
5. **Identify subordinate subclusters inside each basic cluster**
   - For each basic cluster, split subordinate abstractions into subordinate-level groups.
6. **Annotate subordinate nodes**
   - Assign each subordinate abstraction to a subordinate cluster under its parent basic cluster.
7. **Build the final tree**
   - Convert the processed cluster assignments into a `Tree` object and write JSON output.

That exact sequence is visible in `mechanism_tree_pipeline.py` and the final construction is done by `build_mechanism_tree_from_annotations()` in `tree_builder.py`.

### Output Tree Schema

The original tree schema is explicitly defined in `tree_builder.py`:

```text
root
└── superordinate_cluster
    └── basic_cluster
        └── suboridinate_cluster
```

Each cluster node stores aggregated metadata:

- member abstractions assigned to that cluster
- associated idea ids and idea texts
- original cluster ids and parent-cluster ids

This means the tree is **cluster-only**. It does not store one tree node per original abstraction; instead, each tree node is a cluster containing many abstraction members.

## 2. Tree-Pipeline Variants

The variant family is orchestrated by `tree_pipeline_variants_main.py`, which dispatches to six pipelines.

### Shared Variant Pattern

All six variants follow the same high-level structure:

1. Build first-layer groups over **basic abstractions**.
2. Convert those groups into processed assignments.
3. Optionally build a second LM-refined layer over basic abstractions.
4. Build subordinate-level clusters, either by LM or embeddings.
5. Materialize a variant tree with `build_variant_tree()`.

`build_variant_tree()` writes a tree with the schema:

```text
root
└── level1_cluster
    └── level2_cluster   (optional)
        └── level3_cluster
```

If a pipeline skips level 2, `level3_cluster` hangs directly under `level1_cluster`.

### What Changes Across Variants

- **Variant 1: `basic_only`**
  - Level 1 from LM clustering of basic abstractions only.
  - Level 2 from LM subclustering using subordinate evidence.
  - Level 3 from LM subordinate clustering.

- **Variant 2: `basic_super_guided`**
  - Level 1 from LM clustering of basic abstractions, lightly guided by superordinate context.
  - Level 2 from LM subclustering using only the basic text.
  - Level 3 from LM subordinate clustering.

- **Variant 3: `embedding_lm_hybrid`**
  - Level 1 from embedding clustering of basic abstractions.
  - Cluster labels are optionally summarized with the LM.
  - Level 2 and level 3 are still LM-driven.

- **Variant 4: `embedding_only`**
  - Level 1 from embedding clustering of basic abstractions.
  - Subordinate layer from embedding clustering of subordinate entries grouped by basic assignment.
  - No LM-built level 2.

- **Variant 5: `agglomerative_lm_hybrid`**
  - Same structure as variant 3, but embedding clustering uses agglomerative clustering instead of k-means.

- **Variant 6: `agglomerative_only`**
  - Same structure as variant 4, but uses agglomerative clustering instead of k-means.

### Embedding-Based Steps

The embedding-based pipelines use:

- `compute_text_embeddings()`
- `choose_cluster_count()`
- `cluster_with_method()`

The default backend is TF-IDF, but the code can also call Gemini embeddings.

## 3. Original Tree Distances In `tree_builder.py`

Before the routed metric, the original tree object already supported two simpler distances:

- **`calculate_idea_core_abstract_distance()`**
  - path length between the two ideas' core basic nodes
  - this is the simplest "core path" baseline

- **`calculate_distance()`**
  - an older basic-cluster EMD
  - each idea puts mass on all of its basic clusters
  - the core basic cluster gets weight `0.5`
  - the remaining `0.5` is spread evenly across support basic clusters
  - distance is the sum of subtree mass differences over tree edges

Those are useful to know because some benchmark docs call them the older or baseline metrics.

## 4. The Original Routed Metric

The benchmarked "original metric" is implemented by `_build_original_metric_context()` and `score_pair()` in `core_single_purpose_basic_routed_emd_greedy_sweep.py`.

### Intuition

This metric does not compare ideas only by raw tree position. It adds a purpose-routing step:

1. pick one **core purpose** for each idea
2. build a **weighted basic-cluster distribution** for each idea
3. compare those distributions only inside the subtree allowed by the shared purpose
4. blend that mechanism distance with a purpose-mismatch penalty

So it is a **purpose-routed basic-level EMD** built on top of the **original tree schema**.

### Step A: Build Per-Idea Basic-Cluster Info

For each `basic_cluster` node in the original tree, the metric collects:

- whether the idea is present in that cluster
- whether the abstraction is marked `is_core`
- which purpose labels are linked to that abstraction
- which superordinate parent the basic cluster belongs to

### Step B: Choose One Core Purpose Per Idea

For each idea:

- if core basic abstractions have purpose labels, choose the most frequent purpose among them
- otherwise, fall back to the most frequent purpose across all of the idea's basic abstractions

This produces a one-hot distribution:

```text
core_single_purpose_dist[idea] = {core_purpose: 1.0}
```

### Step C: Build The Weighted Basic Distribution

For each basic cluster touched by the idea, assign one of four weights:

- `core_weight` if the basic cluster is core
- `same_purpose_weight` if it is not core but its local purpose matches the idea's core purpose
- `same_super_weight` if it is not in the same purpose bucket but it shares the same superordinate branch as a core node
- `other_weight` otherwise

The resulting weights are normalized into `basic_core_dist`.

The default sweep configuration is:

- `lambda = 0.8`
- `core_weight = 0.78`
- `same_purpose_weight = 0.17`
- `same_super_weight = 0.10`
- `other_weight = 0.05`

The code also checks that the intended ordering holds:

```text
core_weight > same_purpose_weight > same_super_weight > other_weight
```

### Step D: Restrict Comparison To The Purpose-Allowed Subtree

For each purpose, the metric precomputes which original tree nodes are allowed:

- find superordinate clusters linked to that purpose
- include those superordinate nodes plus all descendants

This creates `allowed_nodes_by_purpose`.

### Step E: Compute Routed EMD

For a pair of ideas:

1. look at the shared purpose support
2. project each idea's `basic_core_dist` onto the allowed subtree for that purpose
3. compute tree EMD on that projected mass
4. combine all shared-purpose EMD terms
5. add a purpose-overlap penalty

At the tree level, EMD is the usual edge-flow form used throughout the repo:

```text
distance = sum over edges of |mass_in_child_subtree(A) - mass_in_child_subtree(B)|
```

The final routed distance is:

```text
distance = lambda * D_mech + (1 - lambda) * D_purpose
```

where:

- `D_mech` is the routed tree EMD
- `D_purpose = 1 - purpose_overlap`

### Fallback Behavior

If the two ideas do not share usable routed purpose mass, the metric falls back to a full-tree comparison plus a full purpose penalty:

```text
distance = lambda * fallback_mech + (1 - lambda) * 1.0
```

That behavior is also described in `benchmark_calculation_methods.md` under `core_single_purpose_basic_routed_emd`.

## 5. Short Summary

- The **original tree-building pipeline** is a 3-level LM-assisted clustering pipeline:
  superordinate -> basic -> subordinate.
- The **variant pipelines** keep the same overall goal but change how the first and lower layers are clustered, especially with embedding-based methods.
- The **original routed metric** is not just path distance and not just plain EMD.
  It is a **core-purpose-routed basic-level EMD** over the original tree, using four manually weighted buckets for each idea's basic clusters.
- The repo still contains simpler older baselines:
  `core_path` and the older `basic_cluster_emd`.
