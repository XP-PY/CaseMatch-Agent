from __future__ import annotations

import argparse
import html
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
    create_thread_id,
    resolve_lecard_ranker_config,
)
from .import_service import (
    CaseImportRuntimeConfig,
    build_openai_import_client_from_env,
    import_cases_with_runtime,
)
from .models import AgentResponse, AgentState, ClarificationStatus, ConversationMemory, StructuredQuery

load_dotenv()


CUSTOM_CSS = """
.gradio-container {
  background: linear-gradient(180deg, #f7f8fa 0%, #eef1f5 100%);
}

.app-shell {
  max-width: 1400px;
  margin: 0 auto;
}

.hero-card {
  padding: 1.6rem 1.8rem;
  border-radius: 1.5rem;
  background: rgba(255, 255, 255, 0.92);
  color: #0f172a;
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
}

.hero-title {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.hero-desc {
  margin: 0.55rem 0 1rem 0;
  max-width: 60rem;
  font-size: 1rem;
  line-height: 1.6;
  color: #475569;
}

.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
}

.hero-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.45rem 0.8rem;
  border-radius: 999px;
  background: #f8fafc;
  border: 1px solid #dbe3ee;
  color: #334155;
  font-size: 0.92rem;
}

.toolbar-status {
  margin-top: 0.75rem;
  color: #475569;
  font-size: 0.92rem;
}

.panel-card {
  border-radius: 1.25rem;
  padding: 1rem 1.05rem;
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.22);
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.panel-card h3 {
  margin: 0 0 0.85rem 0;
  font-size: 1rem;
  color: #0f172a;
}

.panel-muted {
  color: #64748b;
  line-height: 1.65;
}

.field-list, .memory-list {
  display: grid;
  gap: 0.7rem;
}

.field-item, .memory-item {
  padding: 0.8rem 0.9rem;
  border-radius: 1rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.field-label, .memory-label {
  display: block;
  margin-bottom: 0.28rem;
  font-size: 0.8rem;
  font-weight: 700;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.field-value, .memory-value {
  color: #0f172a;
  line-height: 1.65;
  word-break: break-word;
}

.results-shell {
  display: grid;
  gap: 0.9rem;
}

.results-summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.2rem 0.2rem 0.1rem 0.2rem;
  color: #334155;
  font-size: 0.95rem;
}

.result-card {
  border-radius: 1.2rem;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
  overflow: hidden;
}

.result-card summary {
  list-style: none;
  cursor: pointer;
}

.result-card summary::-webkit-details-marker {
  display: none;
}

.result-head {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 0.9rem;
  align-items: center;
  padding: 1rem 1.05rem;
}

.result-rank {
  width: 2.15rem;
  height: 2.15rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
  color: white;
  font-weight: 800;
}

.result-title {
  margin: 0;
  font-size: 1rem;
  font-weight: 700;
  color: #0f172a;
}

.result-subtitle {
  margin-top: 0.3rem;
  color: #64748b;
  font-size: 0.92rem;
}

.result-score {
  min-width: 6.6rem;
  text-align: right;
}

.score-label {
  display: block;
  font-size: 0.78rem;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.score-value {
  display: block;
  margin-top: 0.18rem;
  font-size: 1.05rem;
  font-weight: 800;
  color: #1d4ed8;
}

.result-body {
  padding: 0 1.05rem 1rem 1.05rem;
  border-top: 1px solid #e2e8f0;
  background: #fcfdff;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.85rem;
  margin-top: 0.95rem;
}

.detail-item {
  padding: 0.85rem 0.9rem;
  border-radius: 1rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
}

.detail-item.full-width {
  grid-column: 1 / -1;
}

.detail-label {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.78rem;
  font-weight: 700;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.detail-value {
  color: #0f172a;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}

.detail-scroll {
  max-height: 15rem;
  overflow: auto;
}

.import-card {
  border-radius: 1.3rem;
  padding: 1.2rem 1.25rem;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
}

.import-card h3 {
  margin: 0 0 0.7rem 0;
  color: #0f172a;
}

.import-card p, .import-card li {
  color: #475569;
  line-height: 1.7;
}

.import-card ul {
  margin: 0.6rem 0 0 0;
  padding-left: 1.2rem;
}

@media (max-width: 960px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .result-head {
    grid-template-columns: auto 1fr;
  }

  .result-score {
    grid-column: 1 / -1;
    text-align: left;
    padding-left: 3.05rem;
  }
}
"""


def _default_host() -> str:
    return os.getenv("CASEMATCH_GRADIO_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _default_port() -> int:
    raw_value = os.getenv("CASEMATCH_GRADIO_PORT", "7860").strip() or "7860"
    try:
        return int(raw_value)
    except ValueError:
        return 7860


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name if path.is_absolute() else str(path)


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


def _build_agent_from_args(args: argparse.Namespace, ranker_override: str | None = None) -> tuple[CaseMatchAgent, str]:
    corpus_path = Path(args.corpus)
    db_path = Path(args.db_path)
    lancedb_uri = Path(args.lancedb_uri)

    if corpus_path.exists():
        ranker_config = resolve_lecard_ranker_config(
            ranker_name=ranker_override or args.ranker,
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
            f"语料 {_display_path(corpus_path)} | "
            f"后端 {args.db_backend} | "
            f"检索器 {ranker_config.ranker_name}"
        )
        return agent, corpus_hint

    return build_default_agent(), "示例语料"


def _escape_multiline(text: str) -> str:
    return html.escape(text or "无").replace("\n", "<br>")


def _header_html(*, corpus_hint: str, host: str, port: int) -> str:
    return f"""
    <section class="app-shell">
      <div class="hero-card">
        <h1 class="hero-title">CaseMatch Agent</h1>
        <p class="hero-desc">
          面向中国刑事类案检索的多轮 Agent 页面。先基于当前信息完成一轮检索，再按需要继续追问补充信息。
        </p>
        <div class="hero-chips">
          <span class="hero-chip">当前语料: {html.escape(corpus_hint)}</span>
          <span class="hero-chip">服务地址: {html.escape(f"{host}:{port}")}</span>
          <span class="hero-chip">模式: 类案查询 / 新增数据</span>
        </div>
      </div>
    </section>
    """


def _format_query(query: StructuredQuery) -> str:
    def _fmt_list(values: list[str]) -> str:
        return html.escape("、".join(values) if values else "未识别")

    items = [
        ("案情摘要", html.escape(query.case_summary or "未识别")),
        ("罪名", _fmt_list(query.charges)),
        ("刑事争点", html.escape(query.dispute_focus or "未识别")),
        ("四要件-主体", _fmt_list(query.four_element_subject)),
        ("四要件-客体", _fmt_list(query.four_element_object)),
        ("四要件-客观方面", _fmt_list(query.four_element_objective_aspect)),
        ("四要件-主观方面", _fmt_list(query.four_element_subjective_aspect)),
        ("法条", _fmt_list(query.legal_basis)),
        ("裁判说理", html.escape(query.court_reasoning or "未识别")),
        ("置信度", html.escape(f"{query.confidence:.2f}")),
    ]
    body = "".join(
        f"""
        <div class="field-item">
          <span class="field-label">{label}</span>
          <div class="field-value">{value}</div>
        </div>
        """
        for label, value in items
    )
    return f"""
    <div class="panel-card">
      <h3>结构化查询</h3>
      <div class="field-list">{body}</div>
    </div>
    """


def _format_results(response: AgentResponse) -> str:
    if not response.retrieval_results:
        return """
        <div class="panel-card">
          <h3>类案结果</h3>
          <div class="panel-muted">当前没有召回到类案。</div>
        </div>
        """

    cards: list[str] = []
    for index, result in enumerate(response.retrieval_results, start=1):
        case = result.case
        title = html.escape(case.case_name or case.case_id)
        subtitle_parts = []
        if case.document_name:
            subtitle_parts.append(case.document_name)
        if case.charges:
            subtitle_parts.append("、".join(case.charges[:3]))
        subtitle = html.escape(" | ".join(subtitle_parts) if subtitle_parts else "点击展开查看详细内容")
        reason_text = html.escape("；".join(result.reasons) if result.reasons else "无")
        cards.append(
            f"""
            <details class="result-card">
              <summary>
                <div class="result-head">
                  <div class="result-rank">{index}</div>
                  <div>
                    <div class="result-title">{title}</div>
                    <div class="result-subtitle">{subtitle}</div>
                  </div>
                  <div class="result-score">
                    <span class="score-label">相似分</span>
                    <span class="score-value">{html.escape(f"{result.total_score:.4f}")}</span>
                  </div>
                </div>
              </summary>
              <div class="result-body">
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="detail-label">案件编号</span>
                    <div class="detail-value">{html.escape(case.case_id)}</div>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">检索理由</span>
                    <div class="detail-value">{reason_text}</div>
                  </div>
                  <div class="detail-item full-width">
                    <span class="detail-label">案情摘要</span>
                    <div class="detail-value">{_escape_multiline(case.case_summary)}</div>
                  </div>
                  <div class="detail-item full-width">
                    <span class="detail-label">案件事实</span>
                    <div class="detail-value detail-scroll">{_escape_multiline(case.fact_text)}</div>
                  </div>
                  <div class="detail-item full-width">
                    <span class="detail-label">判决结果</span>
                    <div class="detail-value detail-scroll">{_escape_multiline(case.judgment_text)}</div>
                  </div>
                  <div class="detail-item full-width">
                    <span class="detail-label">文书全文</span>
                    <div class="detail-value detail-scroll">{_escape_multiline(case.full_text)}</div>
                  </div>
                </div>
              </div>
            </details>
            """
        )
    return f"""
    <div class="results-shell">
      <div class="results-summary">
        <strong>类案结果</strong>
        <span>已返回 {len(response.retrieval_results)} 条候选，点击卡片可展开或收起详情。</span>
      </div>
      {''.join(cards)}
    </div>
    """


def _format_memory(memory: ConversationMemory) -> str:
    def _fmt_list(values: list[str]) -> str:
        return html.escape("、".join(values) if values else "无")

    items = [
        ("已追问字段", _fmt_list(memory.asked_fields)),
        ("已回答字段", _fmt_list(memory.answered_fields)),
        ("明确不知道的字段", _fmt_list(memory.declined_fields)),
        ("待补充字段", _fmt_list(memory.pending_fields)),
    ]
    body = "".join(
        f"""
        <div class="memory-item">
          <span class="memory-label">{label}</span>
          <div class="memory-value">{value}</div>
        </div>
        """
        for label, value in items
    )
    return f"""
    <div class="panel-card">
      <h3>会话记忆</h3>
      <div class="memory-list">{body}</div>
    </div>
    """


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
    return """
    <div class="panel-card">
      <h3>结构化查询</h3>
      <div class="panel-muted">等待用户输入。系统会先抽取结构化字段，再执行一轮检索。</div>
    </div>
    """


def _initial_results_panel() -> str:
    return """
    <div class="panel-card">
      <h3>类案结果</h3>
      <div class="panel-muted">等待用户输入。默认会返回 5 条候选，并支持展开查看详细文书内容。</div>
    </div>
    """


def _initial_memory_panel() -> str:
    return """
    <div class="panel-card">
      <h3>会话记忆</h3>
      <div class="panel-muted">等待用户输入。这里会显示已追问、已回答和明确不知道的字段。</div>
    </div>
    """


def create_app(
    agent: CaseMatchAgent,
    *,
    corpus_hint: str,
    host: str,
    port: int,
    agent_builder: Callable[[str | None], tuple[CaseMatchAgent, str]],
    import_runtime_config: CaseImportRuntimeConfig,
    initial_ranker: str,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` first.") from exc

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
    )

    # 用普通 Python 容器保存当前运行时对象，不放进 gr.State
    runtime = {
        "agent": agent,
        "corpus_hint": corpus_hint,
    }

    def _toolbar_status_text(ranker_name: str) -> str:
        return (
            "<div class='toolbar-status'>"
            f"当前检索器: <strong>{html.escape(ranker_name)}</strong>。切换检索器后会自动重置当前会话。"
            "</div>"
        )

    def _submit_message(
        user_message: str,
        chat_history: list[dict[str, str]] | None,
        state: AgentState | None,
        thread_id: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, str]], AgentState | None, str, str, str, str]:
        message = user_message.strip()
        history = list(chat_history or [])
        if not message:
            return "", history, state, _initial_query_panel(), _initial_results_panel(), _initial_memory_panel(), thread_id

        active_agent = runtime["agent"]
        response = active_agent.respond(message, thread_id=thread_id, top_k=int(top_k))

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": _assistant_message(response)})

        return (
            "",
            history,
            response.state,
            _format_query(response.structured_query),
            _format_results(response),
            _format_memory(response.state.memory),
            thread_id,
        )

    def _reset_conversation() -> tuple[list[dict[str, str]], None, str, str, str, str, str]:
        return (
            [],
            None,
            "",
            _initial_query_panel(),
            _initial_results_panel(),
            _initial_memory_panel(),
            create_thread_id(),
        )

    def _change_ranker(
        ranker_name: str,
    ) -> tuple[str, list[dict[str, str]], None, str, str, str, str, str]:
        current_corpus_hint = runtime["corpus_hint"]
        try:
            rebuilt_agent, rebuilt_hint = agent_builder(ranker_name)
            runtime["agent"] = rebuilt_agent
            runtime["corpus_hint"] = rebuilt_hint
        except Exception as exc:
            return (
                _header_html(corpus_hint=current_corpus_hint, host=host, port=port),
                [],
                None,
                _initial_query_panel(),
                _initial_results_panel(),
                _initial_memory_panel(),
                create_thread_id(),
                f"<div class='toolbar-status'>切换检索器失败: {html.escape(str(exc))}</div>",
            )

        return (
            _header_html(corpus_hint=rebuilt_hint, host=host, port=port),
            [],
            None,
            _initial_query_panel(),
            _initial_results_panel(),
            _initial_memory_panel(),
            create_thread_id(),
            _toolbar_status_text(ranker_name),
        )

    def _import_cases(
        uploaded_file: str | None,
        chat_history: list[dict[str, str]] | None,
        conversation_state: AgentState | None,
        thread_id: str,
        current_query_panel: str,
        current_results_panel: str,
        current_memory_panel: str,
        current_ranker: str,
    ) -> tuple[
        str,
        list[dict[str, str]],
        AgentState | None,
        str,
        str,
        str,
        str,
        str | None,
    ]:
        history = list(chat_history or [])
        current_corpus_hint = runtime["corpus_hint"]

        if not uploaded_file:
            return (
                _header_html(corpus_hint=current_corpus_hint, host=host, port=port),
                history,
                conversation_state,
                current_query_panel,
                current_results_panel,
                current_memory_panel,
                thread_id,
                "请先上传一个 raw jsonl 文件。",
            )

        try:
            llm_client = build_openai_import_client_from_env()
            report = import_cases_with_runtime(
                input_path=uploaded_file,
                runtime_config=import_runtime_config,
                llm_client=llm_client,
            )
            rebuilt_agent, rebuilt_hint = agent_builder(current_ranker)

            # 直接更新运行时 agent / corpus_hint
            runtime["agent"] = rebuilt_agent
            runtime["corpus_hint"] = rebuilt_hint

        except Exception as exc:
            return (
                _header_html(corpus_hint=current_corpus_hint, host=host, port=port),
                history,
                conversation_state,
                current_query_panel,
                current_results_panel,
                current_memory_panel,
                thread_id,
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
            _header_html(corpus_hint=rebuilt_hint, host=host, port=port),
            [],
            None,
            _initial_query_panel(),
            _initial_results_panel(),
            _initial_memory_panel(),
            create_thread_id(),
            "\n".join(status_lines),
        )

    with gr.Blocks(title="CaseMatch Agent", theme=theme, css=CUSTOM_CSS) as app:
        header = gr.HTML(_header_html(corpus_hint=corpus_hint, host=host, port=port))

        with gr.Tabs():
            with gr.Tab("类案查询"):
                with gr.Row():
                    with gr.Column(scale=7):
                        with gr.Row():
                            ranker_dropdown = gr.Dropdown(
                                choices=["simple", "bm25", "bge_m3", "hybrid"],
                                value=initial_ranker,
                                label="检索器",
                                scale=3,
                            )
                            top_k_slider = gr.Slider(
                                minimum=5,
                                maximum=10,
                                value=5,
                                step=1,
                                label="返回候选条数",
                                scale=4,
                            )
                            reset_button = gr.Button("重置上下文", variant="secondary", scale=2)
                        toolbar_status = gr.HTML(_toolbar_status_text(initial_ranker))
                        chatbot = gr.Chatbot(label="对话", height=540)
                        # chatbot = gr.Chatbot(label="对话", height=540, type="messages")
                        with gr.Row():
                            user_input = gr.Textbox(
                                label="案件描述或补充信息",
                                placeholder="输入案件描述，系统会先检索，再按需要追问补充信息。",
                                lines=4,
                                scale=8,
                            )
                            send_button = gr.Button("发送", variant="primary", scale=1)
                        results_panel = gr.HTML(_initial_results_panel())
                    with gr.Column(scale=5):
                        query_panel = gr.HTML(_initial_query_panel())
                        memory_panel = gr.HTML(_initial_memory_panel())

            with gr.Tab("新增数据"):
                with gr.Row():
                    with gr.Column(scale=6):
                        import_intro = gr.HTML(
                            """
                            <div class="import-card">
                              <h3>新增案件导入</h3>
                              <p>
                                上传符合 <code>raw_data</code> 结构的 <code>.jsonl</code> 文件。
                                系统会调用 LLM 提取结构化字段，生成新的 <code>case_id</code>，
                                并同步刷新当前数据库索引。
                              </p>
                              <ul>
                                <li>输入文件每行一个 JSON 对象</li>
                                <li>必填字段: <code>case_name</code>、<code>document_name</code>、<code>fact_text</code>、<code>judgment_text</code>、<code>full_text</code></li>
                                <li>导入成功后，查询会话会自动重置并使用最新数据</li>
                              </ul>
                            </div>
                            """
                        )
                    with gr.Column(scale=6):
                        upload_file = gr.File(
                            label="上传 raw jsonl",
                            file_types=[".jsonl"],
                            type="filepath",
                        )
                        import_button = gr.Button("导入到数据库", variant="primary")
                        import_status = gr.Markdown("等待导入。")

        state = gr.State(value=None)
        thread_state = gr.State(value=create_thread_id())

        submit_outputs = [user_input, chatbot, state, query_panel, results_panel, memory_panel, thread_state]
        send_button.click(
            _submit_message,
            inputs=[user_input, chatbot, state, thread_state, top_k_slider],
            outputs=submit_outputs,
        )
        user_input.submit(
            _submit_message,
            inputs=[user_input, chatbot, state, thread_state, top_k_slider],
            outputs=submit_outputs,
        )

        reset_button.click(
            _reset_conversation,
            inputs=None,
            outputs=[chatbot, state, user_input, query_panel, results_panel, memory_panel, thread_state],
        )

        ranker_dropdown.change(
            _change_ranker,
            inputs=[ranker_dropdown],
            outputs=[header, chatbot, state, query_panel, results_panel, memory_panel, thread_state, toolbar_status],
        )

        import_button.click(
            _import_cases,
            inputs=[upload_file, chatbot, state, thread_state, query_panel, results_panel, memory_panel, ranker_dropdown],
            outputs=[
                header,
                chatbot,
                state,
                query_panel,
                results_panel,
                memory_panel,
                thread_state,
                import_status,
            ],
        )

    return app


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    initial_ranker = resolve_lecard_ranker_config(
        ranker_name=args.ranker,
        bge_model_path=args.bge_model_path,
        bge_use_fp16=False if args.no_fp16 else None,
        bge_batch_size=args.bge_batch_size,
        bge_max_length=args.bge_max_length,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_fe_weight=args.hybrid_fe_weight,
        hybrid_lc_weight=args.hybrid_lc_weight,
        hybrid_bge_weight=args.hybrid_bge_weight,
    ).ranker_name
    agent, corpus_hint = _build_agent_from_args(args, initial_ranker)
    app = create_app(
        agent,
        corpus_hint=corpus_hint,
        host=args.host,
        port=args.port,
        agent_builder=lambda ranker_name=None: _build_agent_from_args(args, ranker_name),
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
        initial_ranker=initial_ranker,
    )
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
