import unittest

from casematch_agent.models import StructuredCase, StructuredQuery
from casematch_agent.retriever import InMemoryCandidateRepository, PipelineCaseRetriever
from casematch_ranker import BGEM3CaseRanker, BM25CaseRanker, DEFAULT_HYBRID_BGE_FIELD_SPECS, HybridRanker


class FakeDenseEncoder:
    VOCAB = [
        "危险驾驶罪",
        "醉酒驾驶机动车",
        "在道路上行驶",
        "自首",
        "自动投案",
        "如实供述",
        "抢劫罪",
        "持刀抢劫",
        "暴力夺取财物",
    ]

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append([float(text.count(token)) for token in self.VOCAB])
        return vectors


class BM25RankerTests(unittest.TestCase):
    def test_bm25_ranker_prioritizes_matching_case(self) -> None:
        query = StructuredQuery(
            raw_query="被告人醉酒驾驶机动车，是否构成危险驾驶罪？",
            case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
            charges=["危险驾驶罪"],
            dispute_focus="是否构成危险驾驶罪",
            four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
            four_element_subjective_aspect=["直接故意"],
            court_reasoning="其行为已构成危险驾驶罪。",
            confidence=0.9,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="危险驾驶罪案件",
                case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
                dispute_focus="是否构成危险驾驶罪",
                charges=["危险驾驶罪"],
                legal_basis=["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
                four_element_subject=["完全刑事责任能力人"],
                four_element_object=["公共交通安全"],
                four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
                four_element_subjective_aspect=["直接故意"],
                court_reasoning="被告人在道路上醉酒驾驶机动车，其行为已构成危险驾驶罪。",
            ),
            StructuredCase(
                case_id="B",
                case_name="抢劫罪案件",
                case_summary="被告人持刀抢劫他人财物，并致被害人轻伤。",
                dispute_focus="是否构成抢劫罪",
                charges=["抢劫罪"],
                legal_basis=["《中华人民共和国刑法》第二百六十三条"],
                four_element_subject=["完全刑事责任能力人"],
                four_element_object=["公私财产所有权"],
                four_element_objective_aspect=["持刀抢劫", "暴力夺取财物"],
                four_element_subjective_aspect=["直接故意"],
                court_reasoning="被告人以暴力方法劫取财物，其行为已构成抢劫罪。",
            ),
        ]

        retriever = PipelineCaseRetriever(
            repository=InMemoryCandidateRepository(candidates),
            ranker=BM25CaseRanker(),
            candidate_limit=10,
        )
        results = retriever.search(query, top_k=2)

        self.assertEqual(results[0].case.case_id, "A")
        self.assertGreater(results[0].total_score, results[1].total_score)
        self.assertIn("bm25_total", results[0].field_scores)

    def test_bm25_ranker_uses_dispute_focus_when_summary_is_sparse(self) -> None:
        query = StructuredQuery(
            raw_query="本案争议焦点是是否构成自首",
            dispute_focus="是否构成自首",
            confidence=0.8,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="自首情节",
                case_summary="案件事实较为简单。",
                dispute_focus="是否构成自首及是否可以从轻处罚",
                court_reasoning="被告人自动投案并如实供述，依法构成自首。",
            ),
            StructuredCase(
                case_id="B",
                case_name="累犯情节",
                case_summary="案件事实较为简单。",
                dispute_focus="是否构成累犯",
                court_reasoning="被告人前罪判决生效后再犯，应认定为累犯。",
            ),
        ]

        results = BM25CaseRanker().rank(query, candidates, top_k=2)

        self.assertEqual(results[0].case.case_id, "A")
        self.assertGreater(results[0].field_scores["dispute_focus"], results[1].field_scores["dispute_focus"])


class BGEM3RankerTests(unittest.TestCase):
    def test_bge_m3_ranker_prioritizes_matching_case(self) -> None:
        query = StructuredQuery(
            raw_query="被告人醉酒驾驶机动车，是否构成危险驾驶罪？",
            case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
            charges=["危险驾驶罪"],
            dispute_focus="是否构成危险驾驶罪",
            four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
            four_element_subjective_aspect=["直接故意"],
            confidence=0.9,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="危险驾驶罪案件",
                case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
                dispute_focus="是否构成危险驾驶罪",
                charges=["危险驾驶罪"],
                four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
                four_element_subjective_aspect=["直接故意"],
                court_reasoning="被告人在道路上醉酒驾驶机动车，其行为已构成危险驾驶罪。",
            ),
            StructuredCase(
                case_id="B",
                case_name="抢劫罪案件",
                case_summary="被告人持刀抢劫他人财物，并致被害人轻伤。",
                dispute_focus="是否构成抢劫罪",
                charges=["抢劫罪"],
                four_element_objective_aspect=["持刀抢劫", "暴力夺取财物"],
                four_element_subjective_aspect=["直接故意"],
                court_reasoning="被告人以暴力方法劫取财物，其行为已构成抢劫罪。",
            ),
        ]
        retriever = PipelineCaseRetriever(
            repository=InMemoryCandidateRepository(candidates),
            ranker=BGEM3CaseRanker(encoder=FakeDenseEncoder()),
            candidate_limit=10,
        )

        results = retriever.search(query, top_k=2)

        self.assertEqual(results[0].case.case_id, "A")
        self.assertGreater(results[0].total_score, results[1].total_score)
        self.assertIn("bge_m3_total", results[0].field_scores)

    def test_bge_m3_ranker_caches_candidate_embeddings(self) -> None:
        encoder = FakeDenseEncoder()
        ranker = BGEM3CaseRanker(encoder=encoder)
        query = StructuredQuery(
            raw_query="被告人自动投案并如实供述，是否构成自首？",
            dispute_focus="是否构成自首",
            court_reasoning="自动投案并如实供述是否构成自首",
            confidence=0.8,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="自首情节",
                case_summary="案情摘要较短。",
                dispute_focus="是否构成自首",
                court_reasoning="被告人自动投案并如实供述，依法构成自首。",
            ),
            StructuredCase(
                case_id="B",
                case_name="累犯情节",
                case_summary="案情摘要较短。",
                dispute_focus="是否构成累犯",
                court_reasoning="被告人前罪判决生效后再犯，应认定为累犯。",
            ),
        ]

        ranker.rank(query, candidates, top_k=2)
        first_call_count = len(encoder.calls)
        ranker.rank(query, candidates, top_k=2)
        second_call_count = len(encoder.calls)

        self.assertLessEqual(second_call_count - first_call_count, len(ranker.field_specs))


class HybridRankerTests(unittest.TestCase):
    def test_hybrid_ranker_combines_bm25_and_bge_scores(self) -> None:
        query = StructuredQuery(
            raw_query="被告人醉酒驾驶机动车，是否构成危险驾驶罪？",
            case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
            charges=["危险驾驶罪"],
            dispute_focus="是否构成危险驾驶罪",
            four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
            confidence=0.9,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="危险驾驶罪案件",
                case_summary="被告人醉酒驾驶机动车，在道路上行驶，被民警当场查获。",
                dispute_focus="是否构成危险驾驶罪",
                charges=["危险驾驶罪"],
                legal_basis=["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
                four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
                court_reasoning="其行为已构成危险驾驶罪。",
            ),
            StructuredCase(
                case_id="B",
                case_name="抢劫罪案件",
                case_summary="被告人持刀抢劫他人财物。",
                dispute_focus="是否构成抢劫罪",
                charges=["抢劫罪"],
                legal_basis=["《中华人民共和国刑法》第二百六十三条"],
                four_element_objective_aspect=["持刀抢劫", "暴力夺取财物"],
                court_reasoning="其行为已构成抢劫罪。",
            ),
        ]

        ranker = HybridRanker(
            bge_m3_ranker=BGEM3CaseRanker(encoder=FakeDenseEncoder(), field_specs=DEFAULT_HYBRID_BGE_FIELD_SPECS),
            bm25_weight=0.3,
            bge_m3_weight=0.7,
        )
        results = ranker.rank(query, candidates, top_k=2)

        self.assertEqual(results[0].case.case_id, "A")
        self.assertIn("hybrid_total", results[0].field_scores)

    def test_hybrid_ranker_bonus_breaks_tie(self) -> None:
        encoder = FakeDenseEncoder()
        query = StructuredQuery(
            raw_query="被告人醉酒驾驶机动车，是否构成危险驾驶罪？",
            case_summary="被告人醉酒驾驶机动车，在道路上行驶。",
            charges=["危险驾驶罪"],
            legal_basis=["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
            dispute_focus="是否构成危险驾驶罪",
            confidence=0.9,
        )
        candidates = [
            StructuredCase(
                case_id="A",
                case_name="危险驾驶罪案件",
                case_summary="被告人醉酒驾驶机动车，在道路上行驶。",
                dispute_focus="是否构成危险驾驶罪",
                charges=["危险驾驶罪"],
                legal_basis=["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
            ),
            StructuredCase(
                case_id="B",
                case_name="相似摘要但法条罪名不符",
                case_summary="被告人醉酒驾驶机动车，在道路上行驶。",
                dispute_focus="是否构成危险驾驶罪",
                charges=["交通肇事罪"],
                legal_basis=["《中华人民共和国刑法》第一百三十三条"],
            ),
        ]

        ranker = HybridRanker(
            bge_m3_ranker=BGEM3CaseRanker(encoder=encoder, field_specs=DEFAULT_HYBRID_BGE_FIELD_SPECS),
            bm25_weight=1.0,
            bge_m3_weight=1.0,
            bm25_fe_weight=0.0,
            bm25_lc_weight=1.0,
        )
        results = ranker.rank(query, candidates, top_k=2)

        self.assertEqual(results[0].case.case_id, "A")
        self.assertGreater(results[0].field_scores["hybrid_bm25_lc_bonus"], results[1].field_scores["hybrid_bm25_lc_bonus"])


if __name__ == "__main__":
    unittest.main()
