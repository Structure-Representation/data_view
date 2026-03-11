# Benchmark Calculation Methods Explained

This document explains the calculation methods used in
[distance_method_benchmark.py](/Users/yaqing_personal/Documents/cmu/structured_representation/structure_tree_build_test/distance_method_benchmark.py).

The benchmark compares three ideas inside one triple. Each method returns a
`score` for each pair:

- larger score = the pair is considered more similar
- for true distance methods such as EMD or path length, the benchmark stores
`score = -distance`, so a smaller distance still becomes a larger score

## 1. Shared Toy Example

All methods below use the same small toy mechanism tree and the same two ideas.

### 1.1 Toy mechanism tree

```text
Root
├─ S1: Remove contaminants          (purpose = Clean)
│  ├─ B1: Dissolve / solvate dirt
│  │  ├─ D1: solvent spray
│  │  └─ D2: supercritical CO2
│  └─ B2: physically lift particles
│     └─ D3: air jet
└─ S2: Refresh / deodorize          (purpose = Refresh)
   └─ B3: neutralize odor molecules
      └─ D4: cold plasma
```

### 1.2 Two toy ideas

- Idea A
  - core basic node: `B1`
  - support basic node: `B2`
  - subordinate evidence: `D1` under `B1`, `D3` under `B2`
  - purpose: mostly `Clean`
- Idea B
  - core basic node: `B3`
  - support basic node: `B1`
  - subordinate evidence: `D4` under `B3`, `D2` under `B1`
  - purposes: mixed `Clean` + `Refresh`, but core purpose is `Refresh`

### 1.3 Toy feature vectors derived from the tree

These are the small example vectors used below.

#### Purpose features

- `A purpose set = {Clean}`
- `B purpose set = {Clean, Refresh}`
- `A purpose uniform = {Clean: 1.00}`
- `B purpose uniform = {Clean: 0.50, Refresh: 0.50}`
- `A purpose weighted = {Clean: 1.00}`
- `B purpose weighted = {Clean: 0.35, Refresh: 0.65}`
- `A purpose tfidf = {Clean: 1.00}`
- `B purpose tfidf = {Clean: 0.412, Refresh: 0.588}`

#### Tree membership sets

- `A super set = {S1}`
- `B super set = {S1, S2}`
- `A basic set = {B1, B2}`
- `B basic set = {B1, B3}`
- `A subordinate set = {D1, D3}`
- `B subordinate set = {D2, D4}`

#### Weighted basic vectors

- `A core-weighted basic = {B1: 0.667, B2: 0.333}`
- `B core-weighted basic = {B1: 0.333, B3: 0.667}`

#### Weighted subordinate distributions

Using the benchmark's default rule `core parent = 1.0`, `support parent = 0.35`:

- raw `A subordinate weights = {D1: 1.0, D3: 0.35}`
- raw `B subordinate weights = {D2: 0.35, D4: 1.0}`
- normalized `A subordinate dist = {D1: 0.741, D3: 0.259}`
- normalized `B subordinate dist = {D2: 0.259, D4: 0.741}`

#### Core-support feature vectors

- `A basic_core_purpose_vec = {B1: 0.667, B2: 0.333}`
- `B basic_core_purpose_vec = {B1: 0.121, B3: 0.879}`
- `A basic_core_decay_vec = {B1: 0.702, B2: 0.298}`
- `B basic_core_decay_vec = {B1: 0.167, B3: 0.833}`
- `A super_core_purpose_vec = {S1: 1.000}`
- `B super_core_purpose_vec = {S1: 0.179, S2: 0.821}`
- `A basic_core_dist = {B1: 0.821, B2: 0.179}`
- `B basic_core_dist = {B1: 0.060, B3: 0.940}`
- `A super_core_dist = {S1: 1.000}`
- `B super_core_dist = {S1: 0.046, S2: 0.954}`
- `A super_basic_hybrid_vec = {B1: 0.489, B2: 0.244, S1: 0.267}`
- `B super_basic_hybrid_vec = {B1: 0.076, B3: 0.554, S1: 0.066, S2: 0.303}`

#### Core-only purpose rows

- `A core single purpose = {Clean: 1.0}`
- `B core single purpose = {Refresh: 1.0}`

## 2. Helper formulas used many times

### 2.1 Jaccard

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### 2.2 Overlap for weighted distributions

```text
Overlap(P, Q) = Σ_k min(P_k, Q_k)
```

### 2.3 Weighted Jaccard

```text
WeightedJaccard(P, Q) = Σ_k min(P_k, Q_k) / Σ_k max(P_k, Q_k)
```

### 2.4 Cosine

```text
Cosine(x, y) = (x · y) / (||x|| ||y||)
```

### 2.5 Tree EMD used in this benchmark

The benchmark computes tree EMD as:

```text
distance = Σ_edges |mass_A_in_child_subtree - mass_B_in_child_subtree|
score = -distance
```

So larger score still means "closer".

## 3. Method-by-Method Explanation

### 3.1 `purpose_set_jaccard`

Uses only the set of purpose nodes.

- A set = `{Clean}`
- B set = `{Clean, Refresh}`
- score = `1 / 2 = 0.5`

Interpretation: the ideas share one purpose family, but B has one extra purpose.

### 3.2 `purpose_overlap_uniform`

Uses uniform purpose distributions.

- A = `{Clean: 1.0}`
- B = `{Clean: 0.5, Refresh: 0.5}`
- score = `min(1.0, 0.5) = 0.5`

Interpretation: half of B's uniform purpose mass matches A.

### 3.3 `purpose_overlap_weighted`

Uses weighted purpose distributions, so B's core-purpose bias matters.

- A = `{Clean: 1.0}`
- B = `{Clean: 0.35, Refresh: 0.65}`
- score = `min(1.0, 0.35) = 0.35`

Interpretation: once core weighting is applied, the shared purpose becomes smaller.

### 3.4 `purpose_tfidf_cosine`

Uses purpose TF-IDF vectors.

- A = `{Clean: 1.0}`
- B = `{Clean: 0.412, Refresh: 0.588}`
- score ≈ `0.574`

Interpretation: the two ideas still point in a similar purpose direction, but not perfectly.

### 3.5 `super_set_jaccard`

Uses only which super-level mechanism clusters each idea touches.

- A = `{S1}`
- B = `{S1, S2}`
- score = `1 / 2 = 0.5`

Interpretation: they share the cleaning branch, but B also uses the refresh branch.

### 3.6 `basic_set_jaccard`

Uses only which basic-level clusters each idea touches.

- A = `{B1, B2}`
- B = `{B1, B3}`
- score = `1 / 3 ≈ 0.333`

Interpretation: they share one basic mechanism family out of three total families present.

### 3.7 `basic_core_weighted_jaccard`

Like `basic_set_jaccard`, but core basic nodes get higher weight.

- A = `{B1: 0.667, B2: 0.333}`
- B = `{B1: 0.333, B3: 0.667}`
- score = `0.333 / (0.667 + 0.333 + 0.667) = 0.200`

Interpretation: they share only one basic node, and that shared node is not core in both ideas.

### 3.8 `subordinate_set_jaccard`

Uses subordinate node sets only.

- A = `{D1, D3}`
- B = `{D2, D4}`
- score = `0`

Interpretation: no subordinate evidence is shared exactly.

### 3.9 `subordinate_weighted_jaccard`

Uses subordinate weighted distributions instead of sets.

- A = `{D1: 0.741, D3: 0.259}`
- B = `{D2: 0.259, D4: 0.741}`
- score = `0`

Interpretation: still zero, because the subordinate supports are fully disjoint.

### 3.10 `hierarchical_overlap`

Combines overlap on three layers:

```text
0.20 * super_overlap + 0.35 * basic_overlap + 0.45 * subordinate_overlap
```

Toy values:

- super overlap = `0.5`
- basic overlap = `0.5`
- subordinate overlap = `0.0`
- score = `0.20*0.5 + 0.35*0.5 + 0.45*0 = 0.275`

Interpretation: higher layers agree somewhat, but the detailed mechanisms do not.

### 3.11 `fusion_linear_purpose_mech`

Linear fusion of purpose TF-IDF cosine and subordinate weighted Jaccard:

```text
0.45 * purpose_tfidf_cosine + 0.55 * subordinate_weighted_jaccard
```

Toy example:

- purpose cosine ≈ `0.574`
- subordinate weighted jaccard = `0`
- score ≈ `0.45*0.574 = 0.258`

Interpretation: the match comes only from purpose, not from detailed mechanism evidence.

### 3.12 `fusion_gated`

Uses purpose overlap to decide how much to trust mechanism overlap.

```text
if purpose_overlap_weighted < 0.25:
    0.75*p + 0.25*m
else:
    0.35*p + 0.65*m
```

Toy example:

- `p = 0.35`
- `m = 0.275`
- since `0.35 >= 0.25`, score = `0.35*0.35 + 0.65*0.275 ≈ 0.301`

Interpretation: both purpose and hierarchy matter because purpose agreement is not too low.

### 3.13 `fusion_multiplicative`

Mechanism similarity is scaled by purpose agreement:

```text
score = s_m * (0.5 + 0.5*s_p)
```

Toy example:

- `s_m = 0`
- `s_p = 0.35`
- score = `0`

Interpretation: if there is no subordinate agreement at all, the final score stays zero.

### 3.14 `core_purpose_aligned_basic_jaccard`

Builds a purpose-aware basic-level vector. Core basics get strong weight. Support basics
get extra weight if they align with the idea's core purpose.

Toy vectors:

- A = `{B1: 0.667, B2: 0.333}`
- B = `{B1: 0.121, B3: 0.879}`
- score ≈ `0.121 / (0.667 + 0.333 + 0.879) = 0.064`

Interpretation: the shared basic node exists, but it is weak once core-purpose alignment is considered.

### 3.15 `core_support_decay_basic_jaccard`

Similar to the previous method, but support basics decay according to whether they are
under the same super node as the core.

Toy vectors:

- A = `{B1: 0.702, B2: 0.298}`
- B = `{B1: 0.167, B3: 0.833}`
- score ≈ `0.167 / (0.702 + 0.298 + 0.833) = 0.091`

Interpretation: still a weak match, but slightly higher than the stricter purpose-aligned version.

### 3.16 `core_purpose_super_jaccard`

Compares super-level vectors centered on the core purpose.

Toy vectors:

- A = `{S1: 1.000}`
- B = `{S1: 0.179, S2: 0.821}`
- score ≈ `0.179 / (1.000 + 0.821) = 0.098`

Interpretation: B mainly lives in a different super branch.

### 3.17 `core_super_basic_hybrid_cosine`

Creates one mixed vector over both super and basic nodes and takes cosine similarity.

Toy vectors:

- A = `{B1: 0.489, B2: 0.244, S1: 0.267}`
- B = `{B1: 0.076, B3: 0.554, S1: 0.066, S2: 0.303}`
- score ≈ `0.141`

Interpretation: there is some directional similarity because both touch `B1` and `S1`, but not much.

### 3.18 `core_weighted_basic_tree_emd`

Runs tree EMD on the benchmark's core-support basic distribution.

Toy distributions:

- A = `{B1: 0.821, B2: 0.179}`
- B = `{B1: 0.060, B3: 0.940}`

Subtree difference sum on the toy tree gives:

- distance ≈ `3.760`
- score = `-3.760`

Interpretation: much of A's mass sits in `S1`, while B's mass has moved to the `S2 -> B3` branch.

### 3.19 `core_weighted_super_tree_emd`

Runs tree EMD on the super-level core-support distribution.

Toy distributions:

- A = `{S1: 1.000}`
- B = `{S1: 0.046, S2: 0.954}`

Edge differences:

- `Root -> S1`: `|1.000 - 0.046| = 0.954`
- `Root -> S2`: `|0.000 - 0.954| = 0.954`
- distance = `1.908`
- score = `-1.908`

Interpretation: almost all of B's super-level mass moved away from A's branch.

### 3.20 `core_single_purpose_basic_routed_emd`

Uses one core purpose per idea, then routes a basic-level core-support distribution through the
subtree allowed by the shared purpose.

Toy example:

- A core purpose = `Clean`
- B core purpose = `Refresh`
- there is no shared purpose

So the method falls back to:

```text
0.8 * full_basic_tree_emd + 0.2 * 1.0
```

Using the toy `basic_core_dist` example:

- fallback basic-tree distance ≈ `3.760`
- final distance ≈ `0.8*3.760 + 0.2*1 = 3.208`
- score = `-3.208`

Interpretation: no shared routed purpose means a full penalty on purpose mismatch.

### 3.21 `core_single_purpose_super_routed_emd`

Same idea as the previous method, but uses the super-level distribution.

Toy example:

- no shared core purpose
- fallback super-tree distance ≈ `1.908`
- final distance ≈ `0.8*1.908 + 0.2*1 = 1.726`
- score = `-1.726`

Interpretation: the super-level routed version is less harsh because it compares at a coarser level.

### 3.22 `baseline_core_subordinate_emd`

Uses only subordinate nodes that hang under the core basic abstraction.

Toy core subordinate distributions:

- A = `{D1: 1.0}`
- B = `{D4: 1.0}`

All mass must move from the `S1 -> B1 -> D1` branch to the `S2 -> B3 -> D4` branch:

- distance = `6`
- score = `-6`

Interpretation: the core detailed mechanisms are completely different.

### 3.23 `baseline_all_subordinate_emd`

Uses all subordinate nodes under all basic abstractions, with support nodes down-weighted.

Toy distributions:

- A = `{D1: 0.741, D3: 0.259}`
- B = `{D2: 0.259, D4: 0.741}`

Subtree edge-difference sum on the toy tree gives:

- distance ≈ `4.800`
- score = `-4.800`

Interpretation: once support evidence is included, the ideas are still far apart, but less extreme than the pure core-only case.

### 3.24 `baseline_purpose_routed_emd`

Routes the subordinate distributions through shared purpose subtrees, then blends routed
mechanism distance with purpose-overlap penalty.

Toy example:

- A weighted purpose = `{Clean: 1.00}`
- B weighted purpose = `{Clean: 0.35, Refresh: 0.65}`
- shared purpose = `Clean`
- purpose overlap = `0.35`
- purpose penalty = `1 - 0.35 = 0.65`

Project both ideas onto the `Clean` subtree:

- A projected subordinate dist = `{D1: 0.741, D3: 0.259}`
- B projected subordinate dist = `{D2: 1.000}`

Toy routed subtree EMD ≈ `2.600`

Final distance:

```text
0.8 * 2.600 + 0.2 * 0.650 = 2.210
```

- score = `-2.210`

Interpretation: the method says "they are different, but not as different as a full-tree comparison would suggest, because they still share the Clean purpose branch."

### 3.25 `baseline_basic_cluster_emd`

Uses the original basic-cluster tree EMD with one core node and the remaining support mass.

Toy basic distributions:

- A = `{B1: 0.5, B2: 0.5}`
- B = `{B1: 0.5, B3: 0.5}`

Subtree edge-difference sum:

- `Root -> S1`: `|1.0 - 0.5| = 0.5`
- `Root -> S2`: `|0.0 - 0.5| = 0.5`
- `S1 -> B1`: `|0.5 - 0.5| = 0`
- `S1 -> B2`: `|0.5 - 0.0| = 0.5`
- `S2 -> B3`: `|0.0 - 0.5| = 0.5`
- distance = `2.0`
- score = `-2.0`

Interpretation: the ideas share one basic node but disagree on where the core action sits.

### 3.26 `baseline_core_path`

Compares only the path distance between the two ideas' core basic nodes.

Toy example:

- A core basic = `B1`
- B core basic = `B3`
- path = `B1 -> S1 -> Root -> S2 -> B3`
- distance = `4`
- score = `-4`

Interpretation: this is the simplest structural baseline: only core-node separation matters.

### 3.27 `core_basic_single_purpose_routed_emd`

This baseline keeps only one purpose per idea, then runs the purpose-routed EMD over the
all-subordinate mechanism distribution.

Toy example:

- A core purpose = `Clean`
- B core purpose = `Refresh`
- no shared purpose

So the method again falls back to:

```text
0.8 * full_subordinate_emd + 0.2 * 1.0
```

Using the toy all-subordinate distance:

- full subordinate distance ≈ `4.800`
- final distance ≈ `0.8*4.800 + 0.2*1 = 4.040`
- score = `-4.040`

Interpretation: this method is harsher than `baseline_purpose_routed_emd` because it collapses each idea to one purpose only.

### 3.28 `soft_multi_purpose_basic_subordinate_routed_emd`

Uses all shared purposes softly instead of collapsing to one purpose. For each shared purpose
`p`, the routing weight is:

```text
route_weight(p) = min(P_i(p), P_j(p))
```

Then it computes a routed basic-level EMD and a routed subordinate-level EMD, and blends them:

```text
distance = 0.30 * routed_basic_emd
         + 0.50 * routed_subordinate_emd
         + 0.20 * purpose_penalty
```

where:

```text
purpose_penalty = 1 - Overlap(weighted_purpose_i, weighted_purpose_j)
```

Toy example:

- shared purpose is only `Clean`
- routing weight for `Clean` = `min(1.00, 0.35) = 0.35`
- routed basic distance on the `Clean` branch ≈ `0.358`
- routed subordinate distance on the `Clean` branch ≈ `2.600`
- purpose penalty = `0.65`

Final distance:

```text
0.30*0.358 + 0.50*2.600 + 0.20*0.650 ≈ 1.537
```

- score ≈ `-1.537`

Interpretation: this method rewards the shared purpose branch, but still penalizes detailed mechanism mismatch heavily.

### 3.29 `ancestor_path_mass_emd`

Represents each idea as normalized mass over ancestor paths:

```text
(super_cluster, basic_cluster, subordinate_cluster_or_none)
```

Core branches get higher mass before normalization. The method then solves exact optimal transport
between the two path distributions with this path cost:

```text
same path                               -> 0.00
same super, same basic, different sub   -> 0.25
same super, different basic             -> 0.60
different super                         -> 1.00
same super/basic but one lacks sub node -> 0.20
```

Final distance:

```text
0.80 * path_transport_distance + 0.20 * purpose_penalty
```

Toy example path distributions:

- A = `{(S1,B1,D1): 0.741, (S1,B2,D3): 0.259}`
- B = `{(S1,B1,D2): 0.259, (S2,B3,D4): 0.741}`

One low-cost move is:

- move `0.259` from `(S1,B1,D1)` to `(S1,B1,D2)` at cost `0.25`
- move the remaining `0.741` mass to `(S2,B3,D4)` at cost `1.0`

So the transport distance is about:

```text
0.259*0.25 + 0.741*1.0 ≈ 0.806
```

Then:

```text
0.80*0.806 + 0.20*0.650 ≈ 0.775
```

- score ≈ `-0.775`

Interpretation: the ideas get some credit for sharing the same `S1 -> B1` ancestor path, even though their detailed subordinate evidence diverges.

### 3.30 `tree_sliced_ancestor_path_wasserstein`

Starts from the same ancestor-path mass idea as `ancestor_path_mass_emd`, but does not trust one
fixed global tree. Instead it samples multiple nearby hierarchy slices, remaps each idea's paths
through each sampled slice, solves transport in every slice, and averages the results.

By default the benchmark uses:

```text
K = 16 slices
alpha = 0.80
beta = 0.20
```

Within one sampled slice, path cost is based on sampled-tree LCA depth:

```text
cost = 1 - shared_depth / 3
```

So:

- same sampled subordinate path -> `0`
- same sampled super+basic but different subordinate -> `1/3`
- same sampled super but different basic -> `2/3`
- different sampled super -> `1`

Final distance:

```text
distance = 0.80 * average_sliced_transport + 0.20 * purpose_penalty
```

Interpretation: this is a softer, uncertainty-aware version of ancestor-path transport. Nearby branches can become partially aligned in some sampled slices, which can reduce the effective mechanism distance.

### 3.31 `microtree_edit_plus_taxonomy_distance`

Builds a local micro-forest for each idea, then compares the two forests in two stages.

First, it matches nodes separately at `super`, `basic`, and `subordinate` levels using Hungarian
assignment. Pairwise node cost is:

```text
normalized_global_taxonomy_path_distance
+ 0.15 if one node is core and the other is not
```

The level distances are combined as:

```text
D_node_align = 0.20*D_super + 0.35*D_basic + 0.45*D_subordinate
```

Second, it measures structural mismatch among the matched local nodes:

```text
D_structure = 0.50*parent_mismatch_avg
            + 0.25*child_count_diff_avg
            + 0.15*core_branch_mismatch_avg
            + 0.10*top_branch_count_diff
```

Then it adds purpose mismatch:

```text
distance = 0.45*D_node_align + 0.40*D_structure + 0.15*purpose_penalty
```

Interpretation: unlike the pure transport methods, this one compares both where the mechanisms map in the global taxonomy and how each idea's local mechanism tree is internally organized.

### 3.32 `learned_ranker_over_distance_features`

This is the benchmark's learned method rather than a hand-written similarity formula.

It builds a feature vector for each idea pair using:

- direct similarity features such as `purpose_overlap_uniform`, `purpose_set_jaccard`, `super_set_jaccard`, `basic_set_jaccard`, `subordinate_weighted_jaccard`, and `hierarchical_overlap`
- transformed distance features from methods such as `core_weighted_basic_tree_emd`, `baseline_all_subordinate_emd`, `baseline_purpose_routed_emd`, and `soft_multi_purpose_basic_subordinate_routed_emd`
- simple count features such as shared purpose count and shared node counts

Distance-type scores are turned into similarity-style features with:

```text
similarity_feature = 1 / (1 + distance)
```

The model is a standardized logistic regression trained with grouped cross-validation over triples.
For each test triple, it scores the three candidate pairs and picks the pair with the highest predicted probability.

## 4. Quick Summary by Family

### Purpose-only methods

- `purpose_set_jaccard`
- `purpose_overlap_uniform`
- `purpose_overlap_weighted`
- `purpose_tfidf_cosine`

These ignore the mechanism tree and compare only purposes.

### Set / overlap methods on the tree

- `super_set_jaccard`
- `basic_set_jaccard`
- `basic_core_weighted_jaccard`
- `subordinate_set_jaccard`
- `subordinate_weighted_jaccard`
- `hierarchical_overlap`

These compare which nodes the ideas touch, usually without transport on the tree.

### Fusion methods

- `fusion_linear_purpose_mech`
- `fusion_gated`
- `fusion_multiplicative`

These blend purpose similarity and mechanism similarity.

### Core-support feature methods

- `core_purpose_aligned_basic_jaccard`
- `core_support_decay_basic_jaccard`
- `core_purpose_super_jaccard`
- `core_super_basic_hybrid_cosine`
- `core_weighted_basic_tree_emd`
- `core_weighted_super_tree_emd`
- `core_single_purpose_basic_routed_emd`
- `core_single_purpose_super_routed_emd`

These use richer feature engineering around core vs. support mechanisms.

### Baseline tree-distance methods

- `baseline_core_subordinate_emd`
- `baseline_all_subordinate_emd`
- `baseline_purpose_routed_emd`
- `baseline_basic_cluster_emd`
- `baseline_core_path`
- `core_basic_single_purpose_routed_emd`
- `soft_multi_purpose_basic_subordinate_routed_emd`

These are the most directly interpretable structural baselines.

### Ancestor-path and micro-structure methods

- `ancestor_path_mass_emd`
- `tree_sliced_ancestor_path_wasserstein`
- `microtree_edit_plus_taxonomy_distance`

These compare richer path-level or local-tree structure beyond plain node-set overlap.

### Learned method

- `learned_ranker_over_distance_features`

This uses a supervised model over the benchmark's hand-engineered pair features.
