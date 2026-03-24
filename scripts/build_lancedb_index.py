#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

from casematch_agent.agent import DEFAULT_LANCEDB_URI, DEFAULT_STRUCTURED_CORPUS_PATH, resolve_lecard_ranker_config
from casematch_agent.lancedb_store import LanceDBCaseStore
from casematch_ranker import BGEM3DenseEncoder


def _parse_device(value: str | None):
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if "," in stripped:
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return stripped


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or rebuild the LanceDB index for CaseMatch.")
    parser.add_argument("--corpus", default=str(DEFAULT_STRUCTURED_CORPUS_PATH), help="Structured corpus jsonl path.")
    parser.add_argument("--lancedb-uri", default=str(DEFAULT_LANCEDB_URI), help="LanceDB directory path.")
    parser.add_argument("--bge-model-path", default=None, help="Local or remote BGE-M3 model path.")
    parser.add_argument("--bge-batch-size", type=int, default=None, help="BGE-M3 batch size.")
    parser.add_argument("--bge-max-length", type=int, default=None, help="BGE-M3 max sequence length.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, cuda:0, or cuda:0,cuda:1.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if metadata says the index is fresh.")
    parser.add_argument("--json", action="store_true", help="Print the build report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    ranker_config = resolve_lecard_ranker_config(
        bge_model_path=args.bge_model_path,
        bge_use_fp16=False if args.no_fp16 else None,
        bge_batch_size=args.bge_batch_size,
        bge_max_length=args.bge_max_length,
        device=_parse_device(args.device),
    )

    corpus_path = Path(args.corpus)
    lancedb_uri = Path(args.lancedb_uri)

    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    encoder = BGEM3DenseEncoder(
        model_name_or_path=ranker_config.bge_model_path,
        use_fp16=ranker_config.bge_use_fp16,
        batch_size=ranker_config.bge_batch_size,
        max_length=ranker_config.bge_max_length,
        device=ranker_config.device,
    )
    store = LanceDBCaseStore(source_path=corpus_path, db_uri=lancedb_uri, encoder=encoder)

    started_at = time.perf_counter()
    report = store.build(force_rebuild=args.force_rebuild)
    elapsed_s = round(time.perf_counter() - started_at, 3)

    payload = {
        "corpus": str(corpus_path),
        "lancedb_uri": str(report.db_uri),
        "table_name": report.table_name,
        "metadata_path": str(report.metadata_path),
        "rebuilt": report.rebuilt,
        "case_count": report.case_count,
        "source_signature": report.source_signature,
        "built_at_utc": report.built_at_utc,
        "elapsed_s": elapsed_s,
        "bge_model_path": ranker_config.bge_model_path,
        "bge_use_fp16": ranker_config.bge_use_fp16,
        "bge_batch_size": ranker_config.bge_batch_size,
        "bge_max_length": ranker_config.bge_max_length,
        "device": ranker_config.device,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("LanceDB index ready")
    print(f"corpus={payload['corpus']}")
    print(f"lancedb_uri={payload['lancedb_uri']}")
    print(f"table_name={payload['table_name']}")
    print(f"metadata_path={payload['metadata_path']}")
    print(f"rebuilt={payload['rebuilt']}")
    print(f"case_count={payload['case_count']}")
    print(f"source_signature={payload['source_signature']}")
    print(f"built_at_utc={payload['built_at_utc']}")
    print(f"elapsed_s={payload['elapsed_s']}")
    print(f"bge_model_path={payload['bge_model_path']}")
    print(f"device={payload['device']}")


if __name__ == "__main__":
    main()
