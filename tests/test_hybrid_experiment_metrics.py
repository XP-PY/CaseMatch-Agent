import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


def _load_hybrid_experiment_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "hybrid_experiment.py"
    spec = importlib.util.spec_from_file_location("casematch_hybrid_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HybridExperimentMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_hybrid_experiment_module()

    def test_ndcg_uses_exponential_gain(self) -> None:
        ranked_doc_ids = ["a", "b"]
        ground_truth = {"a": 3, "b": 1}

        actual = self.module._ndcg_at_k(ranked_doc_ids, ground_truth, 2)
        expected_dcg = ((2**3) - 1) / math.log2(2) + ((2**1) - 1) / math.log2(3)
        expected_idcg = ((2**3) - 1) / math.log2(2) + ((2**1) - 1) / math.log2(3)
        expected = expected_dcg / expected_idcg

        self.assertAlmostEqual(actual, expected, places=8)

    def test_ndcg_metric_list_matches_fused_bm25_script(self) -> None:
        self.assertEqual(self.module.NDCG_AT_K, (5, 10, 15, 20, 30))

    def test_load_labels_reads_jsonl_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qrels.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"query_id": "q1", "case_id": "c1", "relevance": 3}, ensure_ascii=False),
                        json.dumps({"query_id": "q1", "case_id": "c2", "relevance": 1}, ensure_ascii=False),
                        json.dumps({"query_id": "q2", "case_id": "c3", "relevance": 2}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            actual = self.module._load_labels(path)

        self.assertEqual(actual, {"q1": {"c1": 3, "c2": 1}, "q2": {"c3": 2}})

    def test_load_candidate_pools_reads_jsonl_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidate_pools.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"query_id": "q1", "candidate_case_ids": ["c1", "c2", "c1"]},
                            ensure_ascii=False,
                        ),
                        json.dumps({"query_id": "q2", "candidate_case_ids": ["c3"]}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            actual = self.module._load_candidate_pools(path)

        self.assertEqual(actual, {"q1": ["c1", "c2"], "q2": ["c3"]})


if __name__ == "__main__":
    unittest.main()
