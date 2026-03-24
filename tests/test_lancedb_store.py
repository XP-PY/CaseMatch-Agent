import unittest
from pathlib import Path

from casematch_agent.lancedb_store import LanceDBCaseStore
from casematch_agent.models import StructuredCase, StructuredQuery


class FakeEncoder:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeBuildStore(LanceDBCaseStore):
    def __init__(self, *, needs_rebuild: bool, metadata: dict | None = None):
        super().__init__(
            source_path=Path("data/process/lecard/corpus_merged.jsonl"),
            db_uri=Path("data/process/cases.lancedb"),
            encoder=FakeEncoder(),
        )
        self.needs_rebuild = needs_rebuild
        self.metadata_override = metadata or {}
        self.rebuild_calls = 0

    def _connect_if_needed(self) -> None:
        self.connection = object()
        self.table = object()

    def _needs_rebuild(self) -> bool:
        return self.needs_rebuild

    def _read_metadata(self) -> dict:
        return dict(self.metadata_override)

    def _rebuild(self) -> int:
        self.rebuild_calls += 1
        self.metadata_override = {
            "source_signature": "sig:new",
            "table_name": self.table_name,
            "case_count": 12,
            "built_at_utc": "2026-03-24T00:00:00+00:00",
        }
        self.table = object()
        return 12


class FakeTable:
    def __init__(self, count: int = 0):
        self.rows = []
        self.count = count

    def add(self, rows):
        self.rows.extend(rows)
        self.count += len(rows)

    def count_rows(self):
        return self.count


class FakeAppendStore(LanceDBCaseStore):
    def __init__(self):
        super().__init__(
            source_path=Path("data/process/lecard/corpus_merged.jsonl"),
            db_uri=Path("data/process/cases.lancedb"),
            encoder=FakeEncoder(),
        )
        self.metadata_override = {
            "source_signature": "sig:old",
            "table_name": "cases",
            "case_count": 10,
            "built_at_utc": "2026-03-24T00:00:00+00:00",
        }
        self.table = FakeTable(count=10)

    def _connect_if_needed(self) -> None:
        self.connection = object()

    def _read_metadata(self) -> dict:
        return dict(self.metadata_override)

    def _write_metadata(self, *, case_count: int) -> None:
        self.metadata_override = {
            "source_signature": "sig:new",
            "table_name": self.table_name,
            "case_count": case_count,
            "built_at_utc": "2026-03-24T02:00:00+00:00",
        }


class LanceDBStoreTests(unittest.TestCase):
    def test_build_filter_clause_uses_charge_field(self) -> None:
        store = LanceDBCaseStore(
            source_path=Path("data/process/lecard/corpus_merged.jsonl"),
            db_uri=Path("data/process/cases.lancedb"),
            encoder=FakeEncoder(),
        )
        query = StructuredQuery(raw_query="危险驾驶罪", charges=["危险驾驶罪", "交通肇事罪"])

        where_clause = store._build_filter_clause(query)

        self.assertIn("charges_text LIKE '%危险驾驶罪%'", where_clause)
        self.assertIn("charges_text LIKE '%交通肇事罪%'", where_clause)

    def test_row_to_case_maps_structured_and_raw_fields(self) -> None:
        store = LanceDBCaseStore(
            source_path=Path("data/process/lecard/corpus_merged.jsonl"),
            db_uri=Path("data/process/cases.lancedb"),
            encoder=FakeEncoder(),
        )
        case = store.row_to_case(
            {
                "case_id": "7859",
                "case_name": "应国平交通肇事一案",
                "document_name": "应国平交通肇事一审刑事判决书",
                "fact_text": "被告人醉酒驾驶机动车。",
                "judgment_text": "判决被告人犯危险驾驶罪。",
                "full_text": "完整原文",
                "charges": ["危险驾驶罪"],
                "case_summary": "被告人醉酒驾驶机动车。",
                "dispute_focus": "是否构成危险驾驶罪",
                "legal_basis": ["《中华人民共和国刑法》第一百三十三条之一"],
                "four_element_subject": ["完全刑事责任能力人"],
                "four_element_object": ["公共交通安全"],
                "four_element_objective_aspect": ["醉酒驾驶机动车"],
                "four_element_subjective_aspect": ["直接故意"],
                "court_reasoning": "其行为已构成危险驾驶罪。",
                "traceability_quote": "其行为已构成危险驾驶罪。",
            }
        )

        self.assertEqual(case.case_id, "7859")
        self.assertEqual(case.case_name, "应国平交通肇事一案")
        self.assertEqual(case.charges, ["危险驾驶罪"])
        self.assertTrue(case.full_text)

    def test_build_reuses_existing_metadata_when_index_is_fresh(self) -> None:
        store = FakeBuildStore(
            needs_rebuild=False,
            metadata={
                "source_signature": "sig:old",
                "table_name": "cases",
                "case_count": 9,
                "built_at_utc": "2026-03-24T01:00:00+00:00",
            },
        )

        report = store.build(force_rebuild=False)

        self.assertFalse(report.rebuilt)
        self.assertEqual(report.case_count, 9)
        self.assertEqual(store.rebuild_calls, 0)

    def test_build_force_rebuild_calls_rebuild(self) -> None:
        store = FakeBuildStore(needs_rebuild=False)

        report = store.build(force_rebuild=True)

        self.assertTrue(report.rebuilt)
        self.assertEqual(report.case_count, 12)
        self.assertEqual(store.rebuild_calls, 1)

    def test_add_cases_appends_rows_and_updates_metadata_count(self) -> None:
        store = FakeAppendStore()
        added = store.add_cases(
            [
                StructuredCase(
                    case_id="CASE-1",
                    case_name="王某危险驾驶案",
                    full_text="被告人醉酒驾驶机动车。",
                    charges=["危险驾驶罪"],
                    case_summary="被告人醉酒驾驶机动车。",
                    dispute_focus="是否构成危险驾驶罪",
                )
            ],
            assume_ready=True,
        )

        self.assertEqual(added, 1)
        self.assertEqual(store.table.count_rows(), 11)
        self.assertEqual(store.metadata_override["case_count"], 11)
        self.assertEqual(store.table.rows[0]["case_id"], "CASE-1")


if __name__ == "__main__":
    unittest.main()
