import os
from typing import Any, Dict

from tree_pipeline_variant_utils import (
    BASIC_LEVEL2_BASIC_ONLY_PROMPT,
    BASIC_SUPER_GUIDED_PROMPT,
    DEFAULT_DESIGN_PROBLEM,
    DEFAULT_ORIGINAL_FILE_PATH,
    DEFAULT_VARIANT_OUTPUT_BASE,
    annotate_basic_entries_with_cluster_file,
    build_basic_entries_for_prompt,
    build_variant_tree,
    group_basic_profiles_by_parent_assignment,
    identify_global_clusters_from_profiles,
    identify_grouped_basic_subclusters,
    run_llm_subordinate_stage,
)


def run_pipeline_variant_2(
    original_file_path: str = DEFAULT_ORIGINAL_FILE_PATH,
    output_dir: str = os.path.join(DEFAULT_VARIANT_OUTPUT_BASE, "pipeline_2_basic_super_guided"),
    design_problem: str = DEFAULT_DESIGN_PROBLEM,
    batch_size: int = 20,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    level1_cluster_path = os.path.join(output_dir, "1_level1_basic_super_guided_clusters.json")
    level1_raw_path = os.path.join(output_dir, "2_level1_basic_super_guided_raw.json")
    level1_processed_path = os.path.join(output_dir, "2_level1_basic_super_guided_processed.json")
    level2_cluster_path = os.path.join(output_dir, "3_level2_basic_only_clusters.json")
    level2_raw_path = os.path.join(output_dir, "4_level2_basic_only_raw.json")
    level2_processed_path = os.path.join(output_dir, "4_level2_basic_only_processed.json")
    level3_cluster_path = os.path.join(output_dir, "5_level3_subordinate_clusters.json")
    level3_raw_path = os.path.join(output_dir, "6_level3_subordinate_raw.json")
    level3_processed_path = os.path.join(output_dir, "6_level3_subordinate_processed.json")
    tree_output_path = os.path.join(output_dir, "variant_2_tree.json")

    level1_entries = build_basic_entries_for_prompt(
        original_file_path=original_file_path,
        use_super=True,
        use_sub=False,
    )
    level1_clusters = identify_global_clusters_from_profiles(
        profiles=level1_entries,
        output_path=level1_cluster_path,
        prompt_template=BASIC_SUPER_GUIDED_PROMPT,
        level_name="level1_basic_super_guided",
        design_problem=design_problem,
    )
    level1_processed = annotate_basic_entries_with_cluster_file(
        original_file_path=original_file_path,
        entries=level1_entries,
        cluster_file_path=level1_cluster_path,
        raw_output_path=level1_raw_path,
        processed_output_path=level1_processed_path,
        design_problem=design_problem,
        definition_of_abstraction="Coarse first-layer mechanism family, focused on BASIC abstraction and lightly guided by SUPERORDINATE context.",
        batch_size=batch_size,
    )

    level2_entries = build_basic_entries_for_prompt(
        original_file_path=original_file_path,
        use_super=False,
        use_sub=False,
    )
    grouped_profiles = group_basic_profiles_by_parent_assignment(
        original_file_path=original_file_path,
        assignment_processed_path=level1_processed_path,
        parent_cluster_payload=level1_clusters,
        use_super=False,
        use_sub=False,
    )
    level2_clusters = identify_grouped_basic_subclusters(
        groups=grouped_profiles,
        output_path=level2_cluster_path,
        prompt_template=BASIC_LEVEL2_BASIC_ONLY_PROMPT,
        design_problem=design_problem,
        id_prefix_template="{parent_cluster_id}::level2::{local_id}",
    )
    level2_processed = annotate_basic_entries_with_cluster_file(
        original_file_path=original_file_path,
        entries=level2_entries,
        cluster_file_path=level2_cluster_path,
        raw_output_path=level2_raw_path,
        processed_output_path=level2_processed_path,
        design_problem=design_problem,
        definition_of_abstraction="Second-layer mechanism split using BASIC abstraction only.",
        batch_size=batch_size,
    )

    level3_clusters, level3_processed = run_llm_subordinate_stage(
        original_file_path=original_file_path,
        level2_cluster_file_path=level2_cluster_path,
        level2_processed_path=level2_processed_path,
        level3_cluster_file_path=level3_cluster_path,
        level3_raw_output_path=level3_raw_path,
        level3_processed_output_path=level3_processed_path,
        design_problem=design_problem,
        batch_size=batch_size,
    )

    tree = build_variant_tree(
        original_file_path=original_file_path,
        output_path=tree_output_path,
        level1_cluster_payload=level1_clusters,
        level1_processed=level1_processed,
        level2_cluster_payload=level2_clusters,
        level2_processed=level2_processed,
        level3_cluster_payload=level3_clusters,
        level3_processed=level3_processed,
        root_description=design_problem,
    )

    return {
        "pipeline": "variant_2_basic_super_guided",
        "output_dir": output_dir,
        "tree_output_path": tree_output_path,
        "tree_metadata": tree.get_metadata(),
    }


if __name__ == "__main__":
    run_pipeline_variant_2()
