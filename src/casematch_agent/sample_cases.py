from __future__ import annotations

from .models import StructuredCase


def load_sample_cases() -> list[StructuredCase]:
    return [
        StructuredCase(
            case_id="CM-001",
            case_name="危险驾驶罪样例案件",
            document_name="危险驾驶罪一审刑事判决书",
            fact_text="被告人醉酒驾驶机动车，在道路上行驶，被执勤民警当场查获。",
            judgment_text="判决被告人犯危险驾驶罪。",
            full_text="被告人醉酒驾驶机动车，在道路上行驶，被执勤民警当场查获。法院认为其行为已构成危险驾驶罪。",
            charges=["危险驾驶罪"],
            case_summary="被告人醉酒驾驶机动车，在道路上行驶，被执勤民警当场查获。",
            dispute_focus="是否构成危险驾驶罪",
            legal_basis=["《中华人民共和国刑法》第一百三十三条之一第一款第（二）项"],
            four_element_subject=["完全刑事责任能力人"],
            four_element_object=["公共交通安全"],
            four_element_objective_aspect=["醉酒驾驶机动车", "在道路上行驶"],
            four_element_subjective_aspect=["直接故意"],
            court_reasoning="被告人在道路上醉酒驾驶机动车，其行为已构成危险驾驶罪。",
            traceability_quote="其行为已构成危险驾驶罪。",
        ),
        StructuredCase(
            case_id="CM-002",
            case_name="盗窃罪样例案件",
            document_name="盗窃罪一审刑事判决书",
            fact_text="被告人趁被害人不备，秘密窃取随身财物后逃离现场。",
            judgment_text="判决被告人犯盗窃罪。",
            full_text="被告人趁被害人不备，秘密窃取随身财物后逃离现场。法院认为其行为符合盗窃罪构成要件。",
            charges=["盗窃罪"],
            case_summary="被告人趁被害人不备，秘密窃取随身财物后逃离现场。",
            dispute_focus="是否构成盗窃罪",
            legal_basis=["《中华人民共和国刑法》第二百六十四条"],
            four_element_subject=["完全刑事责任能力人"],
            four_element_object=["他人财物所有权"],
            four_element_objective_aspect=["秘密窃取"],
            four_element_subjective_aspect=["直接故意"],
            court_reasoning="被告人秘密窃取他人财物，其行为已构成盗窃罪。",
            traceability_quote="其行为已构成盗窃罪。",
        ),
    ]
