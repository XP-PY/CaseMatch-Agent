#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable

import os
from typing import Optional, Union, List
import torch
from dotenv import load_dotenv

from casematch_agent.corpus import load_lecard_corpus
from casematch_agent.extractor import HeuristicStructuredQueryExtractor, LLMStructuredQueryExtractor
from casematch_agent.llm import OpenAICompatibleClient, OpenAICompatibleConfig
from casematch_agent.models import StructuredQuery
from casematch_ranker import BGEM3CaseRanker, BM25CaseRanker, DEFAULT_HYBRID_BGE_FIELD_SPECS, HybridRanker

load_dotenv()

DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data/lecard/corpus_merged.jsonl"
DEFAULT_QUERY_PATH = PROJECT_ROOT / "data/lecard/queries.jsonl"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "data/lecard/qrels.jsonl"
DEFAULT_CANDIDATE_POOLS_PATH = PROJECT_ROOT / "data/lecard/candidate_pools.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments"
RELEVANCE_THRESHOLD = 3
PRECISION_AT_K = (1, 3, 5, 10)
NDCG_AT_K = (5, 10, 15, 20, 30)


def resolve_devices(devices: Optional[Union[str, List[str]]] = None) -> Optional[Union[str, List[str]]]:
    if devices is not None:
        return devices

    if not torch.cuda.is_available():
        return "cpu"

    cuda_visible_devices = os.getenv("CUDA_DEVICES", None)
    
    if cuda_visible_devices is not None:
        visible_indices = [int(x.strip()) for x in cuda_visible_devices.split(",")]

        if len(visible_indices) == 1:
            return f"cuda:{visible_indices[0]}"
        else:
            return [f"cuda:{i}" for i in visible_indices]
    
    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus == 1:
        return "cuda:0"

    return [f"cuda:{i}" for i in range(num_visible_gpus)]

devices = resolve_devices()

def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for item in left + right:
        if item and item not in merged:
            merged.append(item)
    return merged


def _load_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _load_labels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            query_id = str(payload.get("query_id", "")).strip()
            case_id = str(payload.get("case_id", "")).strip()
            if not query_id or not case_id:
                continue
            relevance = int(payload.get("relevance", 0))
            labels.setdefault(query_id, {})[case_id] = relevance
    return labels


def _load_candidate_pools(path: Path) -> dict[str, list[str]]:
    candidate_pools: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            query_id = str(payload.get("query_id", "")).strip()
            raw_case_ids = payload.get("candidate_case_ids", [])
            if not query_id or not isinstance(raw_case_ids, list):
                continue

            candidate_ids: list[str] = []
            for case_id in raw_case_ids:
                normalized_case_id = str(case_id).strip()
                if normalized_case_id and normalized_case_id not in candidate_ids:
                    candidate_ids.append(normalized_case_id)
            candidate_pools[query_id] = candidate_ids
    return candidate_pools


def _precision_at_k(ranked_doc_ids: list[str], ground_truth: dict[str, int], k: int) -> float:
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if ground_truth.get(str(doc_id), 0) >= RELEVANCE_THRESHOLD)
    return hits / k if k > 0 else 0.0


def _mrr(ranked_doc_ids: list[str], ground_truth: dict[str, int]) -> float:
    for index, doc_id in enumerate(ranked_doc_ids):
        if ground_truth.get(str(doc_id), 0) >= RELEVANCE_THRESHOLD:
            return 1.0 / (index + 1)
    return 0.0


def _average_precision(ranked_doc_ids: list[str], ground_truth: dict[str, int]) -> float:
    relevant_total = sum(1 for score in ground_truth.values() if score >= RELEVANCE_THRESHOLD)
    if relevant_total <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for index, doc_id in enumerate(ranked_doc_ids):
        if ground_truth.get(str(doc_id), 0) >= RELEVANCE_THRESHOLD:
            hits += 1
            precision_sum += hits / (index + 1)
    return precision_sum / relevant_total


def _ndcg_at_k(ranked_doc_ids: list[str], ground_truth: dict[str, int], k: int) -> float:
    gains = [ground_truth.get(str(doc_id), 0) for doc_id in ranked_doc_ids[:k]]
    gains.extend([0] * max(0, k - len(gains)))

    ideal_gains = sorted(ground_truth.values(), reverse=True)[:k]
    ideal_gains.extend([0] * max(0, k - len(ideal_gains)))

    dcg = sum((math.pow(2, gain) - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum((math.pow(2, gain) - 1) / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def _build_extractor(kind: str):
    if kind == "heuristic":
        return HeuristicStructuredQueryExtractor()
    config = OpenAICompatibleConfig.from_env()
    if not config.is_enabled():
        raise RuntimeError(
            "OpenAI extractor requested, but OPENAI_API_KEY is not configured. "
            "Set OPENAI_API_KEY / OPENAI_API_BASE / OPENAI_MODEL first."
        )
    return LLMStructuredQueryExtractor(
        client=OpenAICompatibleClient(config),
        fallback=HeuristicStructuredQueryExtractor(),
    )


def _build_query(query_payload: dict[str, Any], extractor) -> StructuredQuery:
    raw_query = str(query_payload.get("query_text", "")).replace("\n", " ").strip()
    crimes = [str(item).strip() for item in query_payload.get("charge_labels", []) if str(item).strip()]

    structured_query = extractor.extract(raw_query)
    structured_query.raw_query = raw_query
    structured_query.case_summary = structured_query.case_summary or raw_query
    structured_query.charges = _merge_unique(structured_query.charges, crimes)
    return structured_query


def _candidate_cases(
    query_id: str,
    candidate_pools: dict[str, list[str]],
    case_by_id: dict[str, Any],
) -> tuple[list[Any], int]:
    candidate_ids = candidate_pools.get(query_id, [])
    if not candidate_ids:
        return [], 0

    missing = 0
    cases = []
    for candidate_id in candidate_ids:
        case = case_by_id.get(candidate_id)
        if case is None:
            missing += 1
            continue
        cases.append(case)
    return cases, missing


def _build_rankers(args) -> dict[str, Any]:
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    valid_methods = {"bm25", "bge_m3", "hybrid"}
    unknown = [method for method in methods if method not in valid_methods]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")

    hybrid_fe_weight = args.hybrid_fe_weight if args.hybrid_fe_weight is not None else args.hybrid_bm25_weight
    hybrid_lc_weight = args.hybrid_lc_weight if args.hybrid_lc_weight is not None else args.hybrid_bm25_weight
    bm25_ranker = BM25CaseRanker() if "bm25" in methods else None
    needs_bge = "bge_m3" in methods or ("hybrid" in methods and args.hybrid_bge_weight > 0)
    bge_ranker = (
        BGEM3CaseRanker(
            model_name_or_path=args.bge_model_path,
            use_fp16=args.use_fp16,
            batch_size=args.bge_batch_size,
            max_length=args.bge_max_length,
            device=devices,
        )
        if needs_bge
        else None
    )
    hybrid_bge_ranker = (
        BGEM3CaseRanker(
            field_specs=DEFAULT_HYBRID_BGE_FIELD_SPECS,
            model_name_or_path=args.bge_model_path,
            use_fp16=args.use_fp16,
            batch_size=args.bge_batch_size,
            max_length=args.bge_max_length,
            device=devices,
        )
        if "hybrid" in methods and args.hybrid_bge_weight > 0
        else None
    )

    rankers: dict[str, Any] = {}
    for method in methods:
        if method == "bm25":
            rankers[method] = bm25_ranker or BM25CaseRanker()
        elif method == "bge_m3":
            rankers[method] = bge_ranker or BGEM3CaseRanker(
                model_name_or_path=args.bge_model_path,
                use_fp16=args.use_fp16,
                batch_size=args.bge_batch_size,
                max_length=args.bge_max_length,
                device=devices,
            )
        elif method == "hybrid":
            rankers[method] = HybridRanker(
                bm25_ranker=bm25_ranker,
                bge_m3_ranker=hybrid_bge_ranker,
                bm25_weight=args.hybrid_bm25_weight,
                bge_m3_weight=args.hybrid_bge_weight,
                bm25_fe_weight=hybrid_fe_weight,
                bm25_lc_weight=hybrid_lc_weight,
            )
    return rankers


def _metric_template() -> dict[str, list[float]]:
    template = {"map": [], "mrr": []}
    for k in PRECISION_AT_K:
        template[f"p@{k}"] = []
    for k in NDCG_AT_K:
        template[f"ndcg@{k}"] = []
    return template


def _summarize_metrics(metric_values: dict[str, list[float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        summary[metric_name] = round(sum(values) / len(values), 4) if values else 0.0
    return summary


def _print_summary(report: dict[str, Any]) -> None:
    header = [
        "method",
        "p@1",
        "p@3",
        "p@5",
        "p@10",
        "map",
        "mrr",
        "ndcg@5",
        "ndcg@10",
        "ndcg@15",
        "ndcg@20",
        "ndcg@30",
        "time_s",
    ]
    print("\t".join(header))
    for method_name, payload in report["methods"].items():
        metrics = payload["metrics"]
        row = [
            method_name,
            f"{metrics['p@1']:.4f}",
            f"{metrics['p@3']:.4f}",
            f"{metrics['p@5']:.4f}",
            f"{metrics['p@10']:.4f}",
            f"{metrics['map']:.4f}",
            f"{metrics['mrr']:.4f}",
            f"{metrics['ndcg@5']:.4f}",
            f"{metrics['ndcg@10']:.4f}",
            f"{metrics['ndcg@15']:.4f}",
            f"{metrics['ndcg@20']:.4f}",
            f"{metrics['ndcg@30']:.4f}",
            f"{payload['time_seconds']:.3f}",
        ]
        print("\t".join(row))


def run_experiment(args) -> dict[str, Any]:
    corpus_path = Path(args.corpus).resolve()
    query_path = Path(args.queries).resolve()
    label_path = Path(args.labels).resolve()
    candidate_pools_path = Path(args.candidate_pools).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = _build_extractor(args.extractor)
    case_by_id = {case.case_id: case for case in load_lecard_corpus(corpus_path)}
    queries = _load_queries(query_path)
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    labels = _load_labels(label_path)
    candidate_pools = _load_candidate_pools(candidate_pools_path)
    rankers = _build_rankers(args)

    predictions = {method_name: {} for method_name in rankers}
    metrics = {method_name: _metric_template() for method_name in rankers}
    timings = {method_name: 0.0 for method_name in rankers}
    missing_candidate_total = 0
    skipped_queries = 0

    for query_payload in tqdm(queries, desc="Hybrid Retrieval Experiment"):
        ridx = str(query_payload["query_id"])
        candidate_cases, missing_count = _candidate_cases(ridx, candidate_pools, case_by_id)
        missing_candidate_total += missing_count
        if not candidate_cases:
            skipped_queries += 1
            continue

        structured_query = _build_query(query_payload, extractor)
        ground_truth = labels.get(ridx, {})
        has_positive_label = bool(ground_truth) and any(score > 0 for score in ground_truth.values())
        has_binary_relevance = bool(ground_truth) and any(score >= RELEVANCE_THRESHOLD for score in ground_truth.values())

        for method_name, ranker in rankers.items():
            started_at = time.perf_counter()
            results = ranker.rank(structured_query, candidate_cases, top_k=len(candidate_cases))
            timings[method_name] += time.perf_counter() - started_at

            ranked_doc_ids = [result.case.case_id for result in results]
            predictions[method_name][ridx] = ranked_doc_ids[: args.save_top_n] if args.save_top_n > 0 else ranked_doc_ids

            if not has_positive_label:
                continue

            if has_binary_relevance:
                metrics[method_name]["map"].append(_average_precision(ranked_doc_ids, ground_truth))
                metrics[method_name]["mrr"].append(_mrr(ranked_doc_ids, ground_truth))
                for k in PRECISION_AT_K:
                    metrics[method_name][f"p@{k}"].append(_precision_at_k(ranked_doc_ids, ground_truth, k))
            for k in NDCG_AT_K:
                metrics[method_name][f"ndcg@{k}"].append(_ndcg_at_k(ranked_doc_ids, ground_truth, k))

    report = {
        "config": {
            "corpus": str(corpus_path),
            "queries": str(query_path),
            "labels": str(label_path),
            "candidate_pools": str(candidate_pools_path),
            "extractor": args.extractor,
            "methods": list(rankers.keys()),
            "bge_model_path": args.bge_model_path,
            "hybrid_bm25_weight": args.hybrid_bm25_weight,
            "hybrid_bge_weight": args.hybrid_bge_weight,
            "hybrid_fe_weight": args.hybrid_fe_weight,
            "hybrid_lc_weight": args.hybrid_lc_weight,
            "save_top_n": args.save_top_n,
            "max_queries": args.max_queries,
        },
        "dataset": {
            "corpus_size": len(case_by_id),
            "query_count": len(queries),
            "skipped_queries": skipped_queries,
            "missing_structured_candidates": missing_candidate_total,
        },
        "methods": {},
    }

    for method_name, metric_values in metrics.items():
        report["methods"][method_name] = {
            "metrics": _summarize_metrics(metric_values),
            "time_seconds": round(timings[method_name], 3),
            "prediction_count": len(predictions[method_name]),
        }

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{run_name}_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for method_name, prediction_map in predictions.items():
        prediction_path = output_dir / f"{run_name}_{method_name}_predictions.json"
        with prediction_path.open("w", encoding="utf-8") as handle:
            json.dump(prediction_map, handle, ensure_ascii=False, indent=2)

    print(f"report: {report_path}")
    _print_summary(report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM25 / BGE-M3 / Hybrid retrieval experiments on LeCaRD.")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH), help="Structured LeCaRD corpus jsonl path.")
    parser.add_argument("--queries", default=str(DEFAULT_QUERY_PATH), help="LeCaRD queries jsonl path.")
    parser.add_argument("--labels", default=str(DEFAULT_LABEL_PATH), help="LeCaRD qrels jsonl path.")
    parser.add_argument("--candidate-pools", default=str(DEFAULT_CANDIDATE_POOLS_PATH), help="LeCaRD candidate pools jsonl path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Experiment output directory.")
    parser.add_argument("--run-name", default="", help="Optional run name used for output file names.")
    parser.add_argument(
        "--methods",
        default="bm25,bge_m3,hybrid",
        help="Comma-separated methods: bm25,bge_m3,hybrid",
    )
    parser.add_argument(
        "--extractor",
        choices=["heuristic", "openai"],
        default="heuristic",
        help="Structured query extractor used for LeCaRD query texts.",
    )
    parser.add_argument("--max-queries", type=int, default=0, help="Only evaluate the first N queries. 0 means all.")
    parser.add_argument("--save-top-n", type=int, default=100, help="How many ranked candidates to save per query.")
    parser.add_argument("--bge-model-path", default="BAAI/bge-m3", help="Local or remote BGE-M3 model path.")
    parser.add_argument("--bge-batch-size", type=int, default=32, help="BGE-M3 batch size.")
    parser.add_argument("--bge-max-length", type=int, default=4096, help="BGE-M3 max sequence length.")
    parser.add_argument("--use-fp16", action="store_true", help="Enable fp16 when loading BGE-M3.")
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.45, help="BM25 branch weight in hybrid ranker.")
    parser.add_argument("--hybrid-fe-weight", type=float, default=None, help="BM25 four-elements bonus weight in hybrid ranker.")
    parser.add_argument("--hybrid-lc-weight", type=float, default=None, help="BM25 laws-and-charges bonus weight in hybrid ranker.")
    parser.add_argument("--hybrid-bge-weight", type=float, default=0.55, help="BGE-M3 branch weight in hybrid ranker.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
