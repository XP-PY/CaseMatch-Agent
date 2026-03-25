import json
import tempfile
import unittest
from pathlib import Path

from casematch_agent.case_ingestion import (
    CriminalCaseStructuredDataExtractor,
    generate_unique_case_id,
    import_raw_cases_batch_from_jsonl,
    import_raw_cases_from_jsonl,
)


class FakeLLMClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        return self.payloads.pop(0)


class CaseIngestionTests(unittest.TestCase):
    def test_generate_unique_case_id_skips_existing_ids(self) -> None:
        existing_ids = {"CASE-AAAAAAAAAAAAAAAA"}
        case_id = generate_unique_case_id(existing_ids, prefix="case")

        self.assertTrue(case_id.startswith("CASE-"))
        self.assertNotIn(case_id, existing_ids)

    def test_import_raw_cases_appends_merged_records_and_syncs_backend(self) -> None:
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
                        "applicable_laws": ["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
                    },
                    "traceability": {"reasoning_quote": "其行为已构成危险驾驶罪。"},
                }
            ]
        )
        extractor = CriminalCaseStructuredDataExtractor(client=llm_client)
        synced_cases = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "new_cases.jsonl"
            corpus_path = tmp_path / "corpus_merged.jsonl"
            input_path.write_text(json.dumps(raw_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            report = import_raw_cases_from_jsonl(
                input_path=input_path,
                corpus_path=corpus_path,
                extractor=extractor,
                sync_backend=lambda cases: synced_cases.extend(cases),
            )

            self.assertEqual(report.imported_count, 1)
            self.assertEqual(len(report.case_ids), 1)
            self.assertEqual(len(synced_cases), 1)
            self.assertEqual(synced_cases[0].charges, ["危险驾驶罪"])

            records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["case_id"], report.case_ids[0])
            self.assertEqual(records[0]["raw_data"]["case_name"], "王某危险驾驶案")
            self.assertEqual(records[0]["structured_data"]["laws_and_charges"]["charges"], ["危险驾驶罪"])
            self.assertIn("刑事裁判文书结构化提取器", llm_client.calls[0]["system_prompt"])

    def test_import_accepts_nested_raw_data_payload(self) -> None:
        nested_payload = {
            "raw_data": {
                "case_name": "张某盗窃案",
                "document_name": "张某盗窃一审刑事判决书",
                "fact_text": "被告人秘密窃取财物。",
                "judgment_text": "判决被告人犯盗窃罪。",
                "full_text": "被告人秘密窃取财物。法院认为其行为已构成盗窃罪。",
            }
        }
        llm_client = FakeLLMClient(
            [
                {
                    "case_summary": "被告人秘密窃取财物。",
                    "dispute_focus": "是否构成盗窃罪",
                    "four_elements": {
                        "subject": [],
                        "object": [],
                        "objective_aspect": ["秘密窃取"],
                        "subjective_aspect": ["直接故意"],
                    },
                    "court_reasoning": "其行为已构成盗窃罪。",
                    "laws_and_charges": {"charges": ["盗窃罪"], "applicable_laws": []},
                    "traceability": {"reasoning_quote": "其行为已构成盗窃罪。"},
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "new_cases.jsonl"
            corpus_path = tmp_path / "corpus_merged.jsonl"
            input_path.write_text(json.dumps(nested_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            report = import_raw_cases_from_jsonl(
                input_path=input_path,
                corpus_path=corpus_path,
                extractor=CriminalCaseStructuredDataExtractor(client=llm_client),
            )

            self.assertEqual(report.imported_count, 1)
            stored = json.loads(corpus_path.read_text(encoding="utf-8").strip())
            self.assertEqual(stored["raw_data"]["document_name"], "张某盗窃一审刑事判决书")

    def test_import_batch_returns_structured_cases_before_sync(self) -> None:
        raw_payload = {
            "case_name": "李某危险驾驶案",
            "document_name": "李某危险驾驶一审刑事判决书",
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
                        "subject": [],
                        "object": [],
                        "objective_aspect": ["醉酒驾驶机动车"],
                        "subjective_aspect": ["直接故意"],
                    },
                    "court_reasoning": "其行为已构成危险驾驶罪。",
                    "laws_and_charges": {"charges": ["危险驾驶罪"], "applicable_laws": []},
                    "traceability": {"reasoning_quote": "其行为已构成危险驾驶罪。"},
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "new_cases.jsonl"
            corpus_path = tmp_path / "corpus_merged.jsonl"
            input_path.write_text(json.dumps(raw_payload, ensure_ascii=False) + "\n", encoding="utf-8")

            batch = import_raw_cases_batch_from_jsonl(
                input_path=input_path,
                corpus_path=corpus_path,
                extractor=CriminalCaseStructuredDataExtractor(client=llm_client),
            )

            self.assertEqual(batch.report.imported_count, 1)
            self.assertEqual(len(batch.structured_cases), 1)
            self.assertEqual(batch.structured_cases[0].charges, ["危险驾驶罪"])


if __name__ == "__main__":
    unittest.main()
