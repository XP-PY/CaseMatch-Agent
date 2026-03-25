import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casematch_agent.agent import (
    CaseStoreConfig,
    LeCaRDRankerConfig,
    build_default_agent,
    build_lecard_agent,
    build_lecard_ranker,
    resolve_case_store_config,
)
from casematch_agent.clarification import HeuristicClarificationJudge, LLMClarificationJudge
from casematch_agent.corpus import load_lecard_corpus
from casematch_agent.extractor import HeuristicStructuredQueryExtractor, LLMStructuredQueryExtractor
from casematch_agent.lancedb_store import LanceDBCandidateRepository
from casematch_agent.models import ClarificationStatus
from casematch_agent.retriever import PipelineCaseRetriever, SimpleCaseRanker
from casematch_agent.sqlite_store import LeCaRDSQLiteStore, SQLiteLeCaRDCandidateRepository
from casematch_ranker import BM25CaseRanker


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        return self.responses.pop(0)


class FailingLLMClient:
    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        raise RuntimeError("llm unavailable")


class RecordingLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        return self.response


def _sample_lecard_payloads() -> list[dict]:
    return [
        {
            "case_id": "7859",
            "structured_data": {
                "case_summary": "被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
                "dispute_focus": "是否构成危险驾驶罪",
                "four_elements": {
                    "subject": ["完全刑事责任能力人"],
                    "object": ["公共交通安全"],
                    "objective_aspect": ["醉酒驾驶机动车", "在道路上行驶"],
                    "subjective_aspect": ["直接故意"],
                },
                "court_reasoning": "其行为已构成危险驾驶罪。",
                "laws_and_charges": {
                    "charges": ["危险驾驶罪"],
                    "applicable_laws": ["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
                },
                "traceability": {"reasoning_quote": "其行为已构成危险驾驶罪。"},
            },
            "raw_data": {
                "case_name": "应国平交通肇事一案",
                "document_name": "应国平交通肇事一审刑事判决书",
                "fact_text": "被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
                "judgment_text": "判决被告人犯危险驾驶罪。",
                "full_text": "被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。法院认为其行为已构成危险驾驶罪。",
            },
        },
        {
            "case_id": "9999",
            "structured_data": {
                "case_summary": "被告人趁被害人不备，秘密窃取他人财物后逃离现场。",
                "dispute_focus": "是否构成盗窃罪",
                "four_elements": {
                    "subject": ["完全刑事责任能力人"],
                    "object": ["他人财物所有权"],
                    "objective_aspect": ["秘密窃取"],
                    "subjective_aspect": ["直接故意"],
                },
                "court_reasoning": "其行为已构成盗窃罪。",
                "laws_and_charges": {
                    "charges": ["盗窃罪"],
                    "applicable_laws": ["《中华人民共和国刑法》第二百六十四条"],
                },
                "traceability": {"reasoning_quote": "其行为已构成盗窃罪。"},
            },
            "raw_data": {
                "case_name": "张某盗窃一案",
                "document_name": "张某盗窃一审刑事判决书",
                "fact_text": "被告人趁被害人不备，秘密窃取他人财物后逃离现场。",
                "judgment_text": "判决被告人犯盗窃罪。",
                "full_text": "被告人趁被害人不备，秘密窃取他人财物后逃离现场。法院认为其行为已构成盗窃罪。",
            },
        },
    ]


class CaseMatchAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = build_default_agent(client=FailingLLMClient())

    def test_requests_clarification_for_sparse_query(self) -> None:
        response = self.agent.respond("我想找危险驾驶罪的类案")
        self.assertEqual(response.decision.status, ClarificationStatus.NEED_MORE_INFO)
        self.assertTrue(response.decision.questions)
        self.assertTrue(response.retrieval_results)

    def test_retrieves_dangerous_driving_case(self) -> None:
        response = self.agent.respond("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertEqual(response.decision.status, ClarificationStatus.READY)
        self.assertEqual(response.retrieval_results[0].case.case_id, "CM-001")

    def test_follow_up_information_refines_result(self) -> None:
        first = self.agent.respond("我想找盗窃罪的类案")
        second = self.agent.respond(
            "是趁被害人不备秘密窃取财物，主观方面是直接故意。",
            state=first.state,
        )
        self.assertEqual(second.retrieval_results[0].case.case_id, "CM-002")
        self.assertEqual(second.decision.status, ClarificationStatus.READY)

    def test_thread_id_persists_multi_turn_state_without_manual_state_passing(self) -> None:
        thread_id = "thread-dangerous-driving"

        first = self.agent.respond("我想找盗窃罪的类案", thread_id=thread_id)
        second = self.agent.respond(
            "是趁被害人不备秘密窃取财物，主观方面是直接故意。",
            thread_id=thread_id,
        )

        self.assertEqual(first.state.thread_id, thread_id)
        self.assertEqual(second.state.thread_id, thread_id)
        self.assertEqual(second.retrieval_results[0].case.case_id, "CM-002")
        self.assertEqual(second.decision.status, ClarificationStatus.READY)
        restored_state = self.agent.get_thread_state(thread_id)
        self.assertIsNotNone(restored_state)
        self.assertEqual(restored_state.structured_query.charges, second.state.structured_query.charges)

    def test_llm_pipeline_can_drive_extraction_and_clarification(self) -> None:
        llm_agent = build_default_agent(
            client=FakeLLMClient(
                [
                    {
                        "case_summary": "被告人醉酒驾驶机动车，在道路上行驶。",
                        "charges": ["危险驾驶罪"],
                        "dispute_focus": "是否构成危险驾驶罪",
                        "legal_basis": [],
                        "four_elements": {
                            "subject": [],
                            "object": [],
                            "objective_aspect": ["醉酒驾驶机动车", "在道路上行驶"],
                            "subjective_aspect": [],
                        },
                        "court_reasoning": "",
                        "confidence": 0.92,
                    },
                    {
                        "need_more_info": False,
                        "reasons": ["当前结果已经有较强参考价值"],
                        "missing_fields": [],
                        "questions": [],
                    },
                ]
            )
        )
        response = llm_agent.respond("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertEqual(response.structured_query.charges, ["危险驾驶罪"])
        self.assertEqual(response.decision.status, ClarificationStatus.READY)
        self.assertEqual(response.retrieval_results[0].case.case_id, "CM-001")

    def test_llm_extractor_does_not_merge_fallback_fields_by_default(self) -> None:
        extractor = LLMStructuredQueryExtractor(
            client=FakeLLMClient(
                [
                    {
                        "case_summary": "危险驾驶案件",
                        "charges": [],
                        "dispute_focus": "是否构成危险驾驶罪",
                        "legal_basis": [],
                        "four_elements": {
                            "subject": [],
                            "object": [],
                            "objective_aspect": ["醉酒驾驶机动车"],
                            "subjective_aspect": [],
                        },
                        "court_reasoning": "",
                        "confidence": 0.82,
                    }
                ]
            ),
            fallback=HeuristicStructuredQueryExtractor(),
        )
        query = extractor.extract("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertEqual(query.charges, [])
        self.assertEqual(query.dispute_focus, "是否构成危险驾驶罪")
        self.assertEqual(query.four_element_objective_aspect, ["醉酒驾驶机动车"])

    def test_llm_extractor_falls_back_only_when_llm_output_is_unusable(self) -> None:
        extractor = LLMStructuredQueryExtractor(
            client=FakeLLMClient(
                [
                    {
                        "case_summary": "",
                        "charges": [],
                        "dispute_focus": "",
                        "legal_basis": [],
                        "four_elements": {
                            "subject": [],
                            "object": [],
                            "objective_aspect": [],
                            "subjective_aspect": [],
                        },
                        "court_reasoning": "",
                        "confidence": 0.0,
                    }
                ]
            ),
            fallback=HeuristicStructuredQueryExtractor(),
        )
        query = extractor.extract("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertEqual(query.charges, ["危险驾驶罪"])

    def test_llm_extractor_falls_back_on_exception(self) -> None:
        extractor = LLMStructuredQueryExtractor(
            client=FailingLLMClient(),
            fallback=HeuristicStructuredQueryExtractor(),
        )
        query = extractor.extract("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertEqual(query.charges, ["危险驾驶罪"])

    def test_extractor_uses_criminal_prompt_template(self) -> None:
        client = RecordingLLMClient(
            {
                "case_summary": "危险驾驶案件",
                "charges": ["危险驾驶罪"],
                "dispute_focus": "是否构成危险驾驶罪",
                "legal_basis": [],
                "four_elements": {
                    "subject": [],
                    "object": [],
                    "objective_aspect": ["醉酒驾驶机动车"],
                    "subjective_aspect": [],
                },
                "court_reasoning": "",
                "confidence": 0.91,
            }
        )
        extractor = LLMStructuredQueryExtractor(client=client, fallback=HeuristicStructuredQueryExtractor())
        extractor.extract("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        self.assertIn("刑事类案检索系统的结构化提取器", client.calls[0]["system_prompt"])
        self.assertIn("刑事类案检索需求", client.calls[0]["user_prompt"])

    def test_llm_clarification_falls_back_when_output_is_unusable(self) -> None:
        judge = LLMClarificationJudge(
            client=FakeLLMClient(
                [
                    {
                        "need_more_info": True,
                        "reasons": [],
                        "missing_fields": [],
                        "questions": [],
                    }
                ]
            ),
            fallback=HeuristicClarificationJudge(),
        )
        query = HeuristicStructuredQueryExtractor().extract("我想找危险驾驶罪的类案")
        results = build_default_agent(client=FailingLLMClient()).retriever.search(query, top_k=3)
        decision = judge.decide(query, results)
        self.assertEqual(decision.status, ClarificationStatus.NEED_MORE_INFO)
        self.assertTrue(decision.questions)

    def test_clarification_uses_criminal_prompt_template(self) -> None:
        client = RecordingLLMClient(
            {
                "need_more_info": False,
                "reasons": ["当前结果已经足够"],
                "missing_fields": [],
                "questions": [],
            }
        )
        judge = LLMClarificationJudge(client=client, fallback=HeuristicClarificationJudge())
        query = HeuristicStructuredQueryExtractor().extract("被告人醉酒驾驶机动车，在道路上行驶，是否构成危险驾驶罪？")
        results = build_default_agent(client=FailingLLMClient()).retriever.search(query, top_k=3)
        judge.decide(query, results)
        self.assertIn("刑事类案检索 agent 的澄清决策器", client.calls[0]["system_prompt"])
        self.assertIn("请按刑事检索场景", client.calls[0]["user_prompt"])

    def test_load_lecard_corpus_maps_structured_and_raw_fields(self) -> None:
        payload = _sample_lecard_payloads()[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "corpus.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            cases = load_lecard_corpus(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].charges, ["危险驾驶罪"])
        self.assertEqual(cases[0].case_name, "应国平交通肇事一案")
        self.assertTrue(cases[0].full_text)

    def test_sqlite_store_returns_candidate_cases_with_raw_fields(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            store = LeCaRDSQLiteStore(source_path=corpus_path, db_path=db_path)
            store.ensure_ready()
            query = HeuristicStructuredQueryExtractor().extract("被告人是否构成危险驾驶罪，且在道路上醉酒驾驶机动车？")
            candidate_ids = store.candidate_ids(query, limit=5)
            self.assertIn("7859", candidate_ids)

            retriever = PipelineCaseRetriever(
                repository=SQLiteLeCaRDCandidateRepository(source_path=corpus_path, db_path=db_path),
                ranker=SimpleCaseRanker(),
                candidate_limit=5,
            )
            results = retriever.search(query, top_k=1)
            self.assertEqual(results[0].case.case_id, "7859")
            self.assertEqual(results[0].case.case_name, "应国平交通肇事一案")
            self.assertTrue(results[0].case.judgment_text)

    def test_build_lecard_ranker_defaults_to_bm25(self) -> None:
        ranker = build_lecard_ranker(LeCaRDRankerConfig())
        self.assertIsInstance(ranker, BM25CaseRanker)

    def test_resolve_case_store_config_defaults_to_auto(self) -> None:
        config = resolve_case_store_config()
        self.assertIsInstance(config, CaseStoreConfig)
        self.assertEqual(config.backend, "auto")

    def test_build_lecard_agent_auto_falls_back_to_sqlite_when_lancedb_is_unavailable(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            with patch("casematch_agent.agent.BGEM3DenseEncoder", return_value=object()):
                with patch(
                    "casematch_agent.agent.LanceDBCandidateRepository",
                    side_effect=RuntimeError("LanceDB backend requires lancedb. Install it first."),
                ):
                    agent = build_lecard_agent(
                        corpus_path=corpus_path,
                        db_path=db_path,
                        db_backend="auto",
                        ranker_name="bm25",
                        candidate_limit=10,
                        client=FailingLLMClient(),
                    )

            self.assertIsInstance(agent.retriever.repository, SQLiteLeCaRDCandidateRepository)

    def test_build_lecard_agent_lancedb_backend_uses_lancedb_repository(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            with patch("casematch_agent.agent.BGEM3DenseEncoder", return_value=object()):
                with patch(
                    "casematch_agent.agent.LanceDBCandidateRepository",
                    return_value=LanceDBCandidateRepository.__new__(LanceDBCandidateRepository),
                ):
                    agent = build_lecard_agent(
                        corpus_path=corpus_path,
                        db_path=db_path,
                        db_backend="lancedb",
                        ranker_name="bm25",
                        candidate_limit=10,
                        client=FailingLLMClient(),
                    )

            self.assertIsInstance(agent.retriever.repository, LanceDBCandidateRepository)

    def test_build_lecard_agent_lancedb_backend_does_not_silently_fallback(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            with patch("casematch_agent.agent.BGEM3DenseEncoder", return_value=object()):
                with patch(
                    "casematch_agent.agent.LanceDBCandidateRepository",
                    side_effect=RuntimeError("LanceDB backend requires lancedb. Install it first."),
                ):
                    with self.assertRaises(RuntimeError):
                        build_lecard_agent(
                            corpus_path=corpus_path,
                            db_path=db_path,
                            db_backend="lancedb",
                            ranker_name="bm25",
                            candidate_limit=10,
                            client=FailingLLMClient(),
                        )

    def test_build_lecard_agent_integrates_bm25_ranker(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            agent = build_lecard_agent(
                corpus_path=corpus_path,
                db_path=db_path,
                db_backend="sqlite",
                ranker_name="bm25",
                candidate_limit=10,
                client=FailingLLMClient(),
            )
            response = agent.respond("被告人是否构成危险驾驶罪，且在道路上醉酒驾驶机动车？")

            self.assertEqual(response.retrieval_results[0].case.case_id, "7859")
            self.assertIn("bm25_total", response.retrieval_results[0].field_scores)

    def test_build_lecard_agent_integrates_hybrid_ranker(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            agent = build_lecard_agent(
                corpus_path=corpus_path,
                db_path=db_path,
                db_backend="sqlite",
                ranker_name="hybrid",
                candidate_limit=10,
                hybrid_bm25_weight=1.0,
                hybrid_bge_weight=0.0,
                client=FailingLLMClient(),
            )
            response = agent.respond("被告人是否构成危险驾驶罪，且在道路上醉酒驾驶机动车？")

            self.assertEqual(response.retrieval_results[0].case.case_id, "7859")
            self.assertIn("hybrid_total", response.retrieval_results[0].field_scores)
            self.assertEqual(response.retrieval_results[0].field_scores["hybrid_bge_m3"], 0.0)

    def test_lecard_agent_only_asks_supported_fields_and_does_not_repeat(self) -> None:
        payloads = _sample_lecard_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.jsonl"
            db_path = Path(tmp_dir) / "corpus.sqlite3"
            corpus_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )
            agent = build_lecard_agent(
                corpus_path=corpus_path,
                db_path=db_path,
                db_backend="sqlite",
                ranker_name="bm25",
                candidate_limit=10,
                client=FailingLLMClient(),
            )
            first = agent.respond("我想找危险驾驶罪的类案")
            self.assertEqual(first.decision.status, ClarificationStatus.NEED_MORE_INFO)
            self.assertTrue(
                set(first.decision.missing_fields).issubset(
                    {"charges", "dispute_focus", "four_element_objective_aspect", "four_element_subjective_aspect", "legal_basis"}
                )
            )

            second = agent.respond("这些我都不清楚，不知道具体法条和主观方面。", state=first.state)
            self.assertTrue(set(second.decision.missing_fields).isdisjoint(set(first.decision.missing_fields)))


if __name__ == "__main__":
    unittest.main()
