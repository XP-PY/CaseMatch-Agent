#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip() or default


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CaseMatch data from a Hugging Face repo.")
    parser.add_argument(
        "--repo-id",
        default=_env("CASEMATCH_HF_REPO_ID"),
        help="Hugging Face repo id, e.g. your-org/CaseMatch-Agent-data.",
    )
    parser.add_argument(
        "--repo-type",
        default=_env("CASEMATCH_HF_REPO_TYPE", "dataset"),
        help="Hugging Face repo type, usually `dataset`.",
    )
    parser.add_argument(
        "--revision",
        default=_env("CASEMATCH_HF_REVISION", "main"),
        help="Repo revision, tag, or branch.",
    )
    parser.add_argument(
        "--local-dir",
        default=_env("CASEMATCH_HF_LOCAL_DIR", "data"),
        help="Local directory to place downloaded files into.",
    )
    parser.add_argument(
        "--token",
        default=_env("HF_TOKEN"),
        help="Optional Hugging Face token. Leave empty for public repos.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional allow pattern. Can be specified multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if not args.repo_id:
        raise SystemExit(
            "No Hugging Face repo id provided. Set CASEMATCH_HF_REPO_ID in .env or pass --repo-id explicitly."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Run `pip install huggingface_hub` first.") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(local_dir),
        token=args.token or None,
        allow_patterns=args.include or None,
    )

    payload = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "snapshot_path": str(snapshot_path),
        "include": args.include,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Hugging Face data download completed")
    print(f"repo_id={payload['repo_id']}")
    print(f"repo_type={payload['repo_type']}")
    print(f"revision={payload['revision']}")
    print(f"local_dir={payload['local_dir']}")
    print(f"snapshot_path={payload['snapshot_path']}")
    if payload["include"]:
        print(f"include={payload['include']}")


if __name__ == "__main__":
    main()
