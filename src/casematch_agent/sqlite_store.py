from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import load_lecard_corpus
from .models import StructuredCase, StructuredQuery
from .utils import tokenize_text


CASE_TABLE = "cases"
CASE_FTS_TABLE = "cases_fts"


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str) -> list[str]:
    return json.loads(value) if value else []


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _fts_clean_term(term: str) -> str:
    cleaned = term.replace('"', " ").replace("'", " ").replace(":", " ").strip()
    return " ".join(piece for piece in cleaned.split() if piece)


@dataclass
class LeCaRDSQLiteStore:
    source_path: Path
    db_path: Path
    connection: sqlite3.Connection | None = field(default=None, init=False)

    def ensure_ready(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        if self._needs_rebuild():
            self._rebuild()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def candidate_ids(self, query: StructuredQuery, limit: int = 200) -> list[str]:
        self.ensure_ready()
        assert self.connection is not None
        match_query = self._build_match_query(query)
        if not match_query:
            return []
        rows = self.connection.execute(
            f"""
            SELECT case_id
            FROM {CASE_FTS_TABLE}
            WHERE {CASE_FTS_TABLE} MATCH ?
            ORDER BY bm25({CASE_FTS_TABLE})
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        return [row["case_id"] for row in rows]

    def fetch_cases(self, case_ids: list[str]) -> list[StructuredCase]:
        self.ensure_ready()
        assert self.connection is not None
        if not case_ids:
            return []

        placeholders = ",".join("?" for _ in case_ids)
        rows = self.connection.execute(
            f"""
            SELECT *
            FROM {CASE_TABLE}
            WHERE case_id IN ({placeholders})
            """,
            case_ids,
        ).fetchall()
        cases_by_id = {row["case_id"]: self._row_to_case(row) for row in rows}
        return [cases_by_id[case_id] for case_id in case_ids if case_id in cases_by_id]

    def _needs_rebuild(self) -> bool:
        if not self.db_path.exists():
            return True
        if self.source_path.stat().st_mtime > self.db_path.stat().st_mtime:
            return True
        assert self.connection is not None
        tables = self.connection.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ('{CASE_TABLE}', '{CASE_FTS_TABLE}')"
        ).fetchall()
        if len(tables) < 2:
            return True
        columns = self.connection.execute(f"PRAGMA table_info({CASE_TABLE})").fetchall()
        column_names = {row["name"] for row in columns}
        return "case_name" not in column_names or "full_text" not in column_names

    def _rebuild(self) -> None:
        assert self.connection is not None
        cursor = self.connection.cursor()
        cursor.executescript(
            f"""
            DROP TABLE IF EXISTS {CASE_TABLE};
            DROP TABLE IF EXISTS {CASE_FTS_TABLE};

            CREATE TABLE {CASE_TABLE} (
                case_id TEXT PRIMARY KEY,
                case_name TEXT NOT NULL,
                document_name TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                judgment_text TEXT NOT NULL,
                full_text TEXT NOT NULL,
                charges_json TEXT NOT NULL,
                charges_text TEXT NOT NULL,
                case_summary TEXT NOT NULL,
                dispute_focus TEXT NOT NULL,
                legal_basis_json TEXT NOT NULL,
                legal_basis_text TEXT NOT NULL,
                four_element_subject_json TEXT NOT NULL,
                four_element_object_json TEXT NOT NULL,
                four_element_objective_aspect_json TEXT NOT NULL,
                four_element_subjective_aspect_json TEXT NOT NULL,
                court_reasoning TEXT NOT NULL,
                traceability_quote TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE {CASE_FTS_TABLE} USING fts5(
                case_id UNINDEXED,
                charges_text,
                dispute_focus,
                case_summary,
                court_reasoning,
                legal_basis_text,
                four_element_objective_aspect_text,
                four_element_subjective_aspect_text,
                tokenize='unicode61'
            );
            """
        )

        deduped_cases = {}
        for case in load_lecard_corpus(self.source_path):
            deduped_cases[case.case_id] = case

        case_rows = []
        fts_rows = []
        for case in deduped_cases.values():
            charges_text = " ".join(case.charges)
            legal_basis_text = " ".join(case.legal_basis)
            objective_text = " ".join(case.four_element_objective_aspect)
            subjective_text = " ".join(case.four_element_subjective_aspect)
            case_rows.append(
                (
                    case.case_id,
                    case.case_name,
                    case.document_name,
                    case.fact_text,
                    case.judgment_text,
                    case.full_text,
                    _json_dumps(case.charges),
                    charges_text,
                    case.case_summary,
                    case.dispute_focus,
                    _json_dumps(case.legal_basis),
                    legal_basis_text,
                    _json_dumps(case.four_element_subject),
                    _json_dumps(case.four_element_object),
                    _json_dumps(case.four_element_objective_aspect),
                    _json_dumps(case.four_element_subjective_aspect),
                    case.court_reasoning,
                    case.traceability_quote,
                )
            )
            fts_rows.append(
                (
                    case.case_id,
                    charges_text,
                    case.dispute_focus,
                    case.case_summary,
                    case.court_reasoning,
                    legal_basis_text,
                    objective_text,
                    subjective_text,
                )
            )

        cursor.executemany(
            f"""
            INSERT OR REPLACE INTO {CASE_TABLE} (
                case_id, case_name, document_name, fact_text, judgment_text, full_text, charges_json, charges_text,
                case_summary, dispute_focus, legal_basis_json, legal_basis_text, four_element_subject_json,
                four_element_object_json, four_element_objective_aspect_json, four_element_subjective_aspect_json,
                court_reasoning, traceability_quote
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            case_rows,
        )
        cursor.executemany(
            f"""
            INSERT INTO {CASE_FTS_TABLE} (
                case_id, charges_text, dispute_focus, case_summary, court_reasoning, legal_basis_text,
                four_element_objective_aspect_text, four_element_subjective_aspect_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )
        self.connection.commit()

    def _build_match_query(self, query: StructuredQuery) -> str:
        terms = _dedupe(
            query.charges
            + query.four_element_subject
            + query.four_element_object
            + query.four_element_objective_aspect
            + query.four_element_subjective_aspect
            + query.legal_basis
            + tokenize_text(query.dispute_focus)
            + tokenize_text(query.raw_query)
        )
        cleaned_terms = []
        for term in terms:
            cleaned = _fts_clean_term(term)
            if len(cleaned) >= 2 and cleaned not in cleaned_terms:
                cleaned_terms.append(cleaned)
        if not cleaned_terms:
            return ""
        return " OR ".join(f'"{term}"' for term in cleaned_terms[:24])

    def _row_to_case(self, row: sqlite3.Row) -> StructuredCase:
        return StructuredCase(
            case_id=row["case_id"],
            case_name=row["case_name"],
            document_name=row["document_name"],
            fact_text=row["fact_text"],
            judgment_text=row["judgment_text"],
            full_text=row["full_text"],
            charges=_json_loads(row["charges_json"]),
            case_summary=row["case_summary"],
            dispute_focus=row["dispute_focus"],
            legal_basis=_json_loads(row["legal_basis_json"]),
            four_element_subject=_json_loads(row["four_element_subject_json"]),
            four_element_object=_json_loads(row["four_element_object_json"]),
            four_element_objective_aspect=_json_loads(row["four_element_objective_aspect_json"]),
            four_element_subjective_aspect=_json_loads(row["four_element_subjective_aspect_json"]),
            court_reasoning=row["court_reasoning"],
            traceability_quote=row["traceability_quote"],
        )


@dataclass
class SQLiteLeCaRDCandidateRepository:
    source_path: Path
    db_path: Path
    store: LeCaRDSQLiteStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = LeCaRDSQLiteStore(source_path=self.source_path, db_path=self.db_path)
        self.store.ensure_ready()

    def candidate_cases(self, query: StructuredQuery, limit: int) -> list[StructuredCase]:
        candidate_ids = self.store.candidate_ids(query, limit=limit)
        return self.store.fetch_cases(candidate_ids)
