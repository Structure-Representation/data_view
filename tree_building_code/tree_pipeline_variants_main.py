import argparse
import inspect
import json
from typing import Any, Callable, Dict, List

from tree_pipeline_variant_1_basic_only import run_pipeline_variant_1
from tree_pipeline_variant_2_basic_super_guided import run_pipeline_variant_2
from tree_pipeline_variant_3_embedding_lm_hybrid import run_pipeline_variant_3
from tree_pipeline_variant_4_embedding_only import run_pipeline_variant_4
from tree_pipeline_variant_5_agglomerative_lm_hybrid import run_pipeline_variant_5
from tree_pipeline_variant_6_agglomerative_only import run_pipeline_variant_6


PIPELINE_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "1": run_pipeline_variant_1,
    "2": run_pipeline_variant_2,
    "3": run_pipeline_variant_3,
    "4": run_pipeline_variant_4,
    "pipeline_1": run_pipeline_variant_1,
    "pipeline_2": run_pipeline_variant_2,
    "pipeline_3": run_pipeline_variant_3,
    "pipeline_4": run_pipeline_variant_4,
    "5": run_pipeline_variant_5,
    "6": run_pipeline_variant_6,
    "pipeline_5": run_pipeline_variant_5,
    "pipeline_6": run_pipeline_variant_6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more tree-building pipeline variants.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["1"],
        help="Pipeline ids to run. Example: --pipelines 1 3 4 5 6",
    )
    parser.add_argument(
        "--original-file-path",
        default=None,
        help="Override the default original abstraction file path.",
    )
    parser.add_argument(
        "--design-problem",
        default=None,
        help="Override the default design problem text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for LM annotation stages.",
    )
    parser.add_argument(
        "--embedding-backend",
        default="tfidf",
        choices=["tfidf", "gemini"],
        help="Embedding backend for embedding-based pipelines.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="gemini",
        help="Embedding model name passed to the embedding helper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict[str, Any]] = []

    for pipeline_id in args.pipelines:
        if pipeline_id not in PIPELINE_REGISTRY:
            raise ValueError(f"Unknown pipeline id '{pipeline_id}'.")

        runner = PIPELINE_REGISTRY[pipeline_id]
        candidate_kwargs: Dict[str, Any] = {"batch_size": args.batch_size}
        if args.original_file_path is not None:
            candidate_kwargs["original_file_path"] = args.original_file_path
        if args.design_problem is not None:
            candidate_kwargs["design_problem"] = args.design_problem
        if pipeline_id in {"3", "4", "5", "6", "pipeline_3", "pipeline_4", "pipeline_5", "pipeline_6"}:
            candidate_kwargs["embedding_backend"] = args.embedding_backend
            candidate_kwargs["embedding_model_name"] = args.embedding_model_name

        accepted = set(inspect.signature(runner).parameters.keys())
        kwargs = {key: value for key, value in candidate_kwargs.items() if key in accepted}

        results.append(runner(**kwargs))

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
