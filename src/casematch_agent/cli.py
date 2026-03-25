from __future__ import annotations

import argparse
from dotenv import load_dotenv
from pathlib import Path

from .agent import (
    DEFAULT_CASE_DB_PATH,
    DEFAULT_LANCEDB_URI,
    DEFAULT_STRUCTURED_CORPUS_PATH,
    build_default_agent,
    build_lecard_agent,
    create_thread_id,
    resolve_lecard_ranker_config,
)
from .models import AgentState, ClarificationStatus, StructuredQuery

BANNER =  """
██████╗ █████╗ ███████╗███████╗███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗ 
██╔════╝██╔══██╗██╔════╝██╔════╝████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║     ███████║███████╗█████╗  ██╔████╔██║███████║   ██║   ██║     ███████║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║     ██╔══██║╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
╚██████╗██║  ██║███████║███████╗██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
 """
load_dotenv()

def _format_query(query: StructuredQuery) -> str:
    lines = [
        "结构化结果:",
        f"- 案情摘要: {query.case_summary or '未识别'}",
        f"- 罪名: {', '.join(query.charges) if query.charges else '未识别'}",
        f"- 刑事争点: {query.dispute_focus or '未识别'}",
        f"- 四要件-主体: {', '.join(query.four_element_subject) if query.four_element_subject else '未识别'}",
        f"- 四要件-客体: {', '.join(query.four_element_object) if query.four_element_object else '未识别'}",
        f"- 四要件-客观方面: {', '.join(query.four_element_objective_aspect) if query.four_element_objective_aspect else '未识别'}",
        f"- 四要件-主观方面: {', '.join(query.four_element_subjective_aspect) if query.four_element_subjective_aspect else '未识别'}",
        f"- 法条: {', '.join(query.legal_basis) if query.legal_basis else '未识别'}",
        f"- 裁判说理: {query.court_reasoning or '未识别'}",
        f"- 置信度: {query.confidence:.2f}",
    ]
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI for CaseMatch Agent.")
    parser.add_argument("--corpus", default=str(DEFAULT_STRUCTURED_CORPUS_PATH), help="Structured corpus jsonl path.")
    parser.add_argument("--db-path", default=str(DEFAULT_CASE_DB_PATH), help="SQLite fallback case index path.")
    parser.add_argument("--lancedb-uri", default=str(DEFAULT_LANCEDB_URI), help="LanceDB directory used for primary vector recall.")
    parser.add_argument(
        "--db-backend",
        choices=["auto", "sqlite", "lancedb"],
        default="auto",
        help="Candidate store backend. `auto` prefers LanceDB and falls back to SQLite.",
    )
    parser.add_argument("--candidate-limit", type=int, default=200, help="Candidate count passed from coarse recall to ranker.")
    parser.add_argument(
        "--ranker",
        choices=["simple", "bm25", "bge_m3", "hybrid"],
        default=None,
        help="Ranker used for LeCaRD retrieval. Defaults to environment config or bm25.",
    )
    parser.add_argument("--bge-model-path", default=None, help="Local or remote BGE-M3 model path.")
    parser.add_argument("--bge-batch-size", type=int, default=None, help="BGE-M3 batch size.")
    parser.add_argument("--bge-max-length", type=int, default=None, help="BGE-M3 max sequence length.")
    parser.add_argument("--hybrid-bm25-weight", type=float, default=None, help="BM25 branch weight in HybridRanker.")
    parser.add_argument("--hybrid-fe-weight", type=float, default=None, help="BM25 four-elements bonus weight in HybridRanker.")
    parser.add_argument("--hybrid-lc-weight", type=float, default=None, help="BM25 laws-and-charges bonus weight in HybridRanker.")
    parser.add_argument("--hybrid-bge-weight", type=float, default=None, help="BGE-M3 branch weight in HybridRanker.")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 when loading BGE-M3.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    corpus_path = Path(args.corpus)
    db_path = Path(args.db_path)
    lancedb_uri = Path(args.lancedb_uri)

    if corpus_path.exists():
        ranker_config = resolve_lecard_ranker_config(
            ranker_name=args.ranker,
            bge_model_path=args.bge_model_path,
            bge_use_fp16=False if args.no_fp16 else None,
            bge_batch_size=args.bge_batch_size,
            bge_max_length=args.bge_max_length,
            hybrid_bm25_weight=args.hybrid_bm25_weight,
            hybrid_fe_weight=args.hybrid_fe_weight,
            hybrid_lc_weight=args.hybrid_lc_weight,
            hybrid_bge_weight=args.hybrid_bge_weight,
        )
        agent = build_lecard_agent(
            corpus_path=corpus_path,
            db_path=db_path,
            lancedb_uri=lancedb_uri,
            db_backend=args.db_backend,
            candidate_limit=args.candidate_limit,
            ranker_name=ranker_config.ranker_name,
            bge_model_path=ranker_config.bge_model_path,
            bge_use_fp16=ranker_config.bge_use_fp16,
            bge_batch_size=ranker_config.bge_batch_size,
            bge_max_length=ranker_config.bge_max_length,
            hybrid_bm25_weight=ranker_config.hybrid_bm25_weight,
            hybrid_fe_weight=ranker_config.hybrid_fe_weight,
            hybrid_lc_weight=ranker_config.hybrid_lc_weight,
            hybrid_bge_weight=ranker_config.hybrid_bge_weight,
        )
        corpus_hint = (
            f"structured corpus @ {corpus_path} | db-backend={args.db_backend} | "
            f"lancedb={lancedb_uri} | sqlite-fallback={db_path} | "
            f"ranker={ranker_config.ranker_name}"
        )
    else:
        agent = build_default_agent()
        corpus_hint = "sample corpus"
    state: AgentState | None = None
    thread_id = create_thread_id()

    print(BANNER)
    print(f"当前语料: {corpus_hint}")
    print("输入案件描述开始检索，输入 reset 重置上下文，输入 exit 退出。")

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "reset":
            state = None
            thread_id = create_thread_id()
            print("上下文已重置。")
            continue
        if not user_input:
            continue

        response = agent.respond(user_input, thread_id=thread_id)
        state = response.state

        print(f"\nAgent: {response.narrative}")
        print(_format_query(response.structured_query))

        print("Top 类案:")
        if response.retrieval_results:
            for index, result in enumerate(response.retrieval_results, start=1):
                print(f"{index}. {result.case.case_name or result.case.case_id} | score={result.total_score:.3f}")
                print(f"   case_id={result.case.case_id}")
                if result.case.document_name:
                    print(f"   document={result.case.document_name}")
                if result.case.charges:
                    print(f"   charges={'、'.join(result.case.charges)}")
                print(f"   reasons={'；'.join(result.reasons)}")
                if result.case.fact_text:
                    print(f"   fact_text={result.case.fact_text[:160]}")
                if result.case.judgment_text:
                    print(f"   judgment_text={result.case.judgment_text[:160]}")
                if result.case.full_text:
                    print(f"   full_text={result.case.full_text[:160]}")
        else:
            print("当前没有召回到类案。")

        if response.decision.status == ClarificationStatus.NEED_MORE_INFO:
            print("为了进一步提高精度，建议继续补充以下信息:")
            for index, question in enumerate(response.decision.questions, start=1):
                print(f"{index}. {question}")
