import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casematch_agent.import_service import CaseImportRuntimeConfig, import_cases_with_runtime


class FakeLLMClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        return self.payloads.pop(0)


class ImportServiceTests(unittest.TestCase):
    def test_import_workflow_writes_cases_and_refreshes_backend(self) -> None:
        raw_payload = {
            "case_name": "王某危险驾驶案",
            "document_name": "王某危险驾驶一审刑事判决书",
            "fact_text": "被告人醉酒驾驶机动车。",
            "judgment_text": "判决被告人犯危险驾驶罪。",
            "full_text": "被告人醉酒驾驶机动车。法院认为其行为已构成危险驾驶罪。",
        }
        llm_client = FakeLLMClient(
            [
                {
                    "case_summary": "被告人醉酒驾驶机动车。",
                    "dispute_focus": "是否构成危险驾驶罪",
                    "four_elements": {
                        "subject": ["完全刑事责任能力人"],
                        "object": ["公共交通安全"],
                        "objective_aspect": ["醉酒驾驶机动车"],
                        "subjective_aspect": ["直接故意"],
                    },
                    "court_reasoning": "其行为已构成危险驾驶罪。",
                    "laws_and_charges": {
                        "charges": ["危险驾驶罪"],
                        "applicable_laws": ["《中华人民共和国刑法》第一百三十三条之一"],
                    },
                    "traceability": {"reasoning_quote": "其行为已构成危险驾驶罪。"},
                }
            ]
        )
        synced_case_ids: list[str] = []

        def _sync_backend(cases):
            synced_case_ids.extend(case.case_id for case in cases)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "new_cases.jsonl"
            corpus_path = tmp_path / "corpus_merged.jsonl"
            input_path.write_text(json.dumps(raw_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            runtime_config = CaseImportRuntimeConfig(
                corpus_path=corpus_path,
                db_path=tmp_path / "cases.sqlite3",
                lancedb_uri=tmp_path / "cases.lancedb",
                db_backend="auto",
            )

            with patch(
                "casematch_agent.import_service._resolve_sync_backend",
                return_value=("sqlite", _sync_backend),
            ):
                report = import_cases_with_runtime(
                    input_path=input_path,
                    runtime_config=runtime_config,
                    llm_client=llm_client,
                )

        self.assertEqual(report.import_report.imported_count, 1)
        self.assertEqual(report.synced_backend, "sqlite")
        self.assertEqual(report.requested_backend, "auto")
        self.assertEqual(synced_case_ids, report.import_report.case_ids)


if __name__ == "__main__":
    unittest.main()
