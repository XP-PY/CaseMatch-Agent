from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from .agent import (
    DEFAULT_CASE_DB_PATH,
    DEFAULT_LANCEDB_URI,
    DEFAULT_STRUCTURED_CORPUS_PATH,
    CaseMatchAgent,
    build_default_agent,
    build_lecard_agent,
    resolve_lecard_ranker_config,
)
from .import_service import (
    CaseImportRuntimeConfig,
    build_openai_import_client_from_env,
    import_cases_with_runtime,
)
from .models import AgentResponse, AgentState, ClarificationStatus, ConversationMemory, StructuredQuery

load_dotenv()


def _default_host() -> str:
    return os.getenv("CASEMATCH_GRADIO_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _default_port() -> int:
    raw_value = os.getenv("CASEMATCH_GRADIO_PORT", "7860").strip() or "7860"
    try:
        return int(raw_value)
    except ValueError:
        return 7860


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gradio UI for CaseMatch Agent.")
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
    parser.add_argument("--host", default=_default_host(), help="Gradio server host.")
    parser.add_argument("--port", type=int, default=_default_port(), help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the Gradio app in a browser after launch.")
    return parser


def _build_agent_from_args(args: argparse.Namespace) -> tuple[CaseMatchAgent, str]:
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
        return agent, corpus_hint

    return build_default_agent(), "sample corpus"


def _header_markdown(*, corpus_hint: str, host: str, port: int) -> str:
    return "\n".join(
        [
            "# CaseMatch Agent",
            "面向中国法律类案检索的多轮检索页面。",
            f"- 当前语料: `{corpus_hint}`",
            f"- 服务地址: `{host}:{port}`",
        ]
    )


def _format_query(query: StructuredQuery) -> str:
    def _fmt_list(values: list[str]) -> str:
        return "、".join(values) if values else "未识别"

    lines = [
        "### 结构化查询",
        f"- 案情摘要: {query.case_summary or '未识别'}",
        f"- 罪名: {_fmt_list(query.charges)}",
        f"- 刑事争点: {query.dispute_focus or '未识别'}",
        f"- 四要件-主体: {_fmt_list(query.four_element_subject)}",
        f"- 四要件-客体: {_fmt_list(query.four_element_object)}",
        f"- 四要件-客观方面: {_fmt_list(query.four_element_objective_aspect)}",
        f"- 四要件-主观方面: {_fmt_list(query.four_element_subjective_aspect)}",
        f"- 法条: {_fmt_list(query.legal_basis)}",
        f"- 裁判说理: {query.court_reasoning or '未识别'}",
        f"- 置信度: {query.confidence:.2f}",
    ]
    return "\n".join(lines)


def _format_results(response: AgentResponse) -> str:
    if not response.retrieval_results:
        return "### 类案结果\n当前没有召回到类案。"

    lines = ["### 类案结果"]
    for index, result in enumerate(response.retrieval_results, start=1):
        case = result.case
        lines.extend(
            [
                f"#### {index}. {case.case_name or case.case_id}",
                f"- case_id: `{case.case_id}`",
                f"- score: `{result.total_score:.4f}`",
                f"- document_name: {case.document_name or '未标注'}",
                f"- 罪名: {'、'.join(case.charges) if case.charges else '未标注'}",
                f"- 检索理由: {'；'.join(result.reasons) if result.reasons else '无'}",
                f"- 案情摘要: {case.case_summary or '无'}",
                f"- fact_text: {case.fact_text or '无'}",
                f"- judgment_text: {case.judgment_text or '无'}",
                f"- full_text: {case.full_text or '无'}",
            ]
        )
    return "\n".join(lines)


def _format_memory(memory: ConversationMemory) -> str:
    def _fmt_list(values: list[str]) -> str:
        return "、".join(values) if values else "无"

    lines = [
        "### 会话记忆",
        f"- 已追问字段: {_fmt_list(memory.asked_fields)}",
        f"- 已回答字段: {_fmt_list(memory.answered_fields)}",
        f"- 明确不知道的字段: {_fmt_list(memory.declined_fields)}",
        f"- 待补充字段: {_fmt_list(memory.pending_fields)}",
    ]
    return "\n".join(lines)


def _assistant_message(response: AgentResponse) -> str:
    lines = [response.narrative]
    if response.retrieval_results:
        top_case = response.retrieval_results[0].case
        lines.append(f"当前最接近的类案: {top_case.case_name or top_case.case_id}")
    if response.decision.status == ClarificationStatus.NEED_MORE_INFO and response.decision.questions:
        lines.append("建议继续补充:")
        for question in response.decision.questions:
            lines.append(f"- {question}")
    return "\n".join(lines)


def _initial_query_panel() -> str:
    return "### 结构化查询\n等待用户输入。"


def _initial_results_panel() -> str:
    return "### 类案结果\n等待用户输入。"


def _initial_memory_panel() -> str:
    return "### 会话记忆\n等待用户输入。"


def create_app(
    agent: CaseMatchAgent,
    *,
    corpus_hint: str,
    host: str,
    port: int,
    agent_builder: Callable[[], tuple[CaseMatchAgent, str]],
    import_runtime_config: CaseImportRuntimeConfig,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` first.") from exc

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="zinc",
    )

    # 用普通 Python 容器保存当前运行时对象，不放进 gr.State
    runtime = {
        "agent": agent,
        "corpus_hint": corpus_hint,
    }

    def _submit_message(
        user_message: str,
        chat_history: list[dict[str, str]] | None,
        state: AgentState | None,
    ) -> tuple[str, list[dict[str, str]], AgentState | None, str, str, str]:
        message = user_message.strip()
        history = list(chat_history or [])
        if not message:
            return "", history, state, _initial_query_panel(), _initial_results_panel(), _initial_memory_panel()

        active_agent = runtime["agent"]
        response = active_agent.respond(message, state=state)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": _assistant_message(response)})

        return (
            "",
            history,
            response.state,
            _format_query(response.structured_query),
            _format_results(response),
            _format_memory(response.state.memory),
        )

    def _reset_conversation() -> tuple[list[dict[str, str]], None, str, str, str, str]:
        return (
            [],
            None,
            "",
            _initial_query_panel(),
            _initial_results_panel(),
            _initial_memory_panel(),
        )

    def _import_cases(
        uploaded_file: str | None,
        chat_history: list[dict[str, str]] | None,
        conversation_state: AgentState | None,
        current_query_panel: str,
        current_results_panel: str,
        current_memory_panel: str,
    ) -> tuple[
        str,
        list[dict[str, str]],
        AgentState | None,
        str,
        str,
        str,
        str | None,
    ]:
        history = list(chat_history or [])
        current_corpus_hint = runtime["corpus_hint"]

        if not uploaded_file:
            return (
                _header_markdown(corpus_hint=current_corpus_hint, host=host, port=port),
                history,
                conversation_state,
                current_query_panel,
                current_results_panel,
                current_memory_panel,
                "请先上传一个 raw jsonl 文件。",
            )

        try:
            llm_client = build_openai_import_client_from_env()
            report = import_cases_with_runtime(
                input_path=uploaded_file,
                runtime_config=import_runtime_config,
                llm_client=llm_client,
            )
            rebuilt_agent, rebuilt_hint = agent_builder()

            # 直接更新运行时 agent / corpus_hint
            runtime["agent"] = rebuilt_agent
            runtime["corpus_hint"] = rebuilt_hint

        except Exception as exc:
            return (
                _header_markdown(corpus_hint=current_corpus_hint, host=host, port=port),
                history,
                conversation_state,
                current_query_panel,
                current_results_panel,
                current_memory_panel,
                f"导入失败: {exc}",
            )

        imported_case_ids = "、".join(report.import_report.case_ids[:20]) if report.import_report.case_ids else "无"
        status_lines = [
            "导入完成。",
            f"- 导入数量: {report.import_report.imported_count}",
            f"- 同步后端: {report.synced_backend}",
            f"- 新增 case_id: {imported_case_ids}",
            "- 会话已重置，后续检索会使用最新数据库内容。",
        ]
        return (
            _header_markdown(corpus_hint=rebuilt_hint, host=host, port=port),
            [],
            None,
            _initial_query_panel(),
            _initial_results_panel(),
            _initial_memory_panel(),
            "\n".join(status_lines),
        )

    # with gr.Blocks(theme=theme, title="CaseMatch Agent") as app:
    with gr.Blocks(title="CaseMatch Agent") as app:
        header = gr.Markdown(_header_markdown(corpus_hint=corpus_hint, host=host, port=port))

        with gr.Row():
            with gr.Column(scale=7):
                # chatbot = gr.Chatbot(label="对话", type="messages", height=620)
                chatbot = gr.Chatbot(label="对话", height=620)
                with gr.Row():
                    user_input = gr.Textbox(
                        label="案件描述或补充信息",
                        placeholder="输入案件描述，系统会先检索，再按需要追问补充信息。",
                        lines=4,
                        scale=8,
                    )
                    send_button = gr.Button("发送", variant="primary", scale=1)
                reset_button = gr.Button("重置上下文")
            with gr.Column(scale=5):
                gr.Markdown("### 新案件导入")
                upload_file = gr.File(
                    label="上传 raw jsonl",
                    file_types=[".jsonl"],
                    type="filepath",
                )
                import_button = gr.Button("导入到数据库")
                import_status = gr.Markdown("等待导入。")
                query_panel = gr.Markdown(_initial_query_panel())
                results_panel = gr.Markdown(_initial_results_panel())
                memory_panel = gr.Markdown(_initial_memory_panel())

        state = gr.State(value=None)

        submit_outputs = [user_input, chatbot, state, query_panel, results_panel, memory_panel]
        send_button.click(_submit_message, inputs=[user_input, chatbot, state], outputs=submit_outputs)
        user_input.submit(_submit_message, inputs=[user_input, chatbot, state], outputs=submit_outputs)

        reset_button.click(
            _reset_conversation,
            inputs=None,
            outputs=[chatbot, state, user_input, query_panel, results_panel, memory_panel],
        )

        import_button.click(
            _import_cases,
            inputs=[upload_file, chatbot, state, query_panel, results_panel, memory_panel],
            outputs=[
                header,
                chatbot,
                state,
                query_panel,
                results_panel,
                memory_panel,
                import_status,
            ],
        )

    return app


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    agent, corpus_hint = _build_agent_from_args(args)
    app = create_app(
        agent,
        corpus_hint=corpus_hint,
        host=args.host,
        port=args.port,
        agent_builder=lambda: _build_agent_from_args(args),
        import_runtime_config=CaseImportRuntimeConfig(
            corpus_path=Path(args.corpus),
            db_path=Path(args.db_path),
            lancedb_uri=Path(args.lancedb_uri),
            db_backend=args.db_backend,
            bge_model_path=args.bge_model_path,
            bge_use_fp16=False if args.no_fp16 else None,
            bge_batch_size=args.bge_batch_size,
            bge_max_length=args.bge_max_length,
            case_id_prefix="CASE",
        ),
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()