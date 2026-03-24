#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

from casematch_agent.agent import (
    DEFAULT_CASE_DB_PATH,
    DEFAULT_LANCEDB_URI,
    DEFAULT_STRUCTURED_CORPUS_PATH,
)
from casematch_agent.import_service import (
    CaseImportRuntimeConfig,
    build_openai_import_client_from_env,
    import_cases_with_runtime,
)


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
    parser = argparse.ArgumentParser(description="Import raw criminal case jsonl into corpus_merged.jsonl and the case database.")
    parser.add_argument("--input", required=True, help="Input jsonl path. Each line must match raw_data schema.")
    parser.add_argument("--corpus", default=str(DEFAULT_STRUCTURED_CORPUS_PATH), help="Merged corpus jsonl path.")
    parser.add_argument("--db-path", default=str(DEFAULT_CASE_DB_PATH), help="SQLite fallback case index path.")
    parser.add_argument("--lancedb-uri", default=str(DEFAULT_LANCEDB_URI), help="LanceDB directory path.")
    parser.add_argument(
        "--db-backend",
        choices=["auto", "sqlite", "lancedb"],
        default="auto",
        help="Backend to sync after corpus append. `auto` prefers LanceDB and falls back to SQLite.",
    )
    parser.add_argument("--case-id-prefix", default="CASE", help="Prefix used when generating random unique case_id.")
    parser.add_argument("--bge-model-path", default=None, help="Local or remote BGE-M3 model path for LanceDB sync.")
    parser.add_argument("--bge-batch-size", type=int, default=None, help="BGE-M3 batch size.")
    parser.add_argument("--bge-max-length", type=int, default=None, help="BGE-M3 max sequence length.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, cuda:0, or cuda:0,cuda:1.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    corpus_path = Path(args.corpus)
    db_path = Path(args.db_path)
    lancedb_uri = Path(args.lancedb_uri)
    device = _parse_device(args.device)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    runtime_config = CaseImportRuntimeConfig(
        corpus_path=corpus_path,
        db_path=db_path,
        lancedb_uri=lancedb_uri,
        db_backend=args.db_backend,
        bge_model_path=args.bge_model_path,
        bge_use_fp16=False if args.no_fp16 else None,
        bge_batch_size=args.bge_batch_size,
        bge_max_length=args.bge_max_length,
        device=device,
        case_id_prefix=args.case_id_prefix,
    )
    try:
        llm_client = build_openai_import_client_from_env()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    execution_report = import_cases_with_runtime(
        input_path=input_path,
        runtime_config=runtime_config,
        llm_client=llm_client,
    )
    report = execution_report.import_report

    payload = {
        "input_path": str(report.input_path),
        "corpus_path": str(report.corpus_path),
        "imported_count": report.imported_count,
        "case_ids": report.case_ids,
        "synced_backend": execution_report.synced_backend,
        "db_backend_requested": execution_report.requested_backend,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Case import completed")
    print(f"input_path={payload['input_path']}")
    print(f"corpus_path={payload['corpus_path']}")
    print(f"imported_count={payload['imported_count']}")
    print(f"synced_backend={payload['synced_backend']}")
    print(f"case_ids={', '.join(payload['case_ids']) if payload['case_ids'] else '[]'}")


if __name__ == "__main__":
    main()
