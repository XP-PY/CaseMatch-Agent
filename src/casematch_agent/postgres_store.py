from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import load_lecard_corpus
from .models import StructuredCase, StructuredQuery
from .search_profiles import case_fused_text, query_fused_text

CASE_TABLE = "cases"
CASE_INDEX_METADATA_TABLE = "case_index_metadata"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_loads(value: str) -> list[str]:
    return json.loads(value) if value else []


def _import_pg_driver():
    try:
        import psycopg  # type: ignore

        return "psycopg", psycopg
    except ImportError:
        try:
            import psycopg2  # type: ignore

            return "psycopg2", psycopg2
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL backend requires psycopg or psycopg2. "
                "Install one of them first."
            ) from exc


def _source_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@dataclass
class PostgresPGVectorCaseStore:
    source_path: Path
    dsn: str
    encoder: object
    connection: object | None = field(default=None, init=False)
    _driver_name: str | None = field(default=None, init=False, repr=False)
    _driver_module: object | None = field(default=None, init=False, repr=False)

    def ensure_ready(self) -> None:
        if self.connection is None:
            self._driver_name, self._driver_module = _import_pg_driver()
            self.connection = self._connect()
        if self._needs_rebuild():
            self._rebuild()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def candidate_ids(self, query: StructuredQuery, limit: int = 200) -> list[str]:
        self.ensure_ready()
        assert self.connection is not None

        query_text = query_fused_text(query)
        if not query_text:
            return []

        vectors = self.encoder.encode([query_text])
        if not vectors:
            return []
        query_vector = self._vector_literal(vectors[0])

        where_clause, params = self._build_filter_clause(query)
        sql = f"""
            SELECT case_id
            FROM {CASE_TABLE}
            {where_clause}
            ORDER BY fused_embedding <=> %s::vector
            LIMIT %s
        """
        rows = self._fetchall(sql, (*params, query_vector, limit))
        return [str(row["case_id"]) for row in rows]

    def fetch_cases(self, case_ids: list[str]) -> list[StructuredCase]:
        self.ensure_ready()
        assert self.connection is not None
        if not case_ids:
            return []

        placeholders = ", ".join(["%s"] * len(case_ids))
        sql = f"""
            SELECT *
            FROM {CASE_TABLE}
            WHERE case_id IN ({placeholders})
        """
        rows = self._fetchall(sql, tuple(case_ids))
        cases_by_id = {str(row["case_id"]): self._row_to_case(row) for row in rows}
        return [cases_by_id[case_id] for case_id in case_ids if case_id in cases_by_id]

    def _connect(self):
        assert self._driver_name is not None
        assert self._driver_module is not None
        if self._driver_name == "psycopg":
            connection = self._driver_module.connect(self.dsn)
            connection.autocommit = False
            return connection
        connection = self._driver_module.connect(self.dsn)
        connection.autocommit = False
        return connection

    def _needs_rebuild(self) -> bool:
        metadata = self._fetch_optional(
            f"""
            SELECT value
            FROM {CASE_INDEX_METADATA_TABLE}
            WHERE key = %s
            """,
            ("source_signature",),
            swallow_missing_table=True,
        )
        if metadata is None:
            return True
        return metadata.get("value", "") != _source_signature(self.source_path)

    def _rebuild(self) -> None:
        assert self.connection is not None
        cases = load_lecard_corpus(self.source_path)
        if not cases:
            return

        fused_vectors = self.encoder.encode([case_fused_text(case) for case in cases])
        if not fused_vectors:
            raise RuntimeError("Failed to build fused embeddings for PostgreSQL pgvector store.")
        vector_dim = len(fused_vectors[0])

        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CASE_INDEX_METADATA_TABLE} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cursor.execute(f"DROP TABLE IF EXISTS {CASE_TABLE}")
            cursor.execute(
                f"""
                CREATE TABLE {CASE_TABLE} (
                    case_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    legal_domain TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    charges_json TEXT NOT NULL,
                    charges_text TEXT NOT NULL,
                    case_summary TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    dispute_points_json TEXT NOT NULL,
                    dispute_focus TEXT NOT NULL,
                    key_facts_json TEXT NOT NULL,
                    requested_relief_json TEXT NOT NULL,
                    legal_basis_json TEXT NOT NULL,
                    four_element_subject_json TEXT NOT NULL,
                    four_element_object_json TEXT NOT NULL,
                    four_element_objective_aspect_json TEXT NOT NULL,
                    four_element_subjective_aspect_json TEXT NOT NULL,
                    court_reasoning TEXT NOT NULL,
                    traceability_quote TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    keywords_text TEXT NOT NULL,
                    fused_text TEXT NOT NULL,
                    fused_embedding vector({vector_dim}) NOT NULL
                )
                """
            )
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{CASE_TABLE}_legal_domain ON {CASE_TABLE} (legal_domain)")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{CASE_TABLE}_source_name ON {CASE_TABLE} (source_name)")

            insert_sql = f"""
                INSERT INTO {CASE_TABLE} (
                    case_id, source_name, title, legal_domain, cause, charges_json, charges_text,
                    case_summary, retrieval_text, dispute_points_json, dispute_focus, key_facts_json,
                    requested_relief_json, legal_basis_json, four_element_subject_json, four_element_object_json,
                    four_element_objective_aspect_json, four_element_subjective_aspect_json, court_reasoning,
                    traceability_quote, keywords_json, keywords_text, fused_text, fused_embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                )
            """
            for case, vector in zip(cases, fused_vectors):
                cursor.execute(insert_sql, self._case_insert_params(case, vector))

            cursor.execute(f"TRUNCATE TABLE {CASE_INDEX_METADATA_TABLE}")
            cursor.execute(
                f"""
                INSERT INTO {CASE_INDEX_METADATA_TABLE} (key, value)
                VALUES (%s, %s)
                """,
                ("source_signature", _source_signature(self.source_path)),
            )

        self.connection.commit()

    def _case_insert_params(self, case: StructuredCase, vector: list[float]) -> tuple[Any, ...]:
        return (
            case.case_id,
            case.source_name,
            case.title,
            case.legal_domain,
            case.cause,
            _json_dumps(case.charges),
            " ".join(case.charges),
            case.case_summary,
            case.retrieval_text,
            _json_dumps(case.dispute_points),
            case.dispute_focus,
            _json_dumps(case.key_facts),
            _json_dumps(case.requested_relief),
            _json_dumps(case.legal_basis),
            _json_dumps(case.four_element_subject),
            _json_dumps(case.four_element_object),
            _json_dumps(case.four_element_objective_aspect),
            _json_dumps(case.four_element_subjective_aspect),
            case.court_reasoning,
            case.traceability_quote,
            _json_dumps(case.keywords),
            " ".join(case.keywords),
            case_fused_text(case),
            self._vector_literal(vector),
        )

    def _build_filter_clause(self, query: StructuredQuery) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []

        if query.legal_domain:
            clauses.append("legal_domain = %s")
            params.append(query.legal_domain)

        if query.cause:
            clauses.append("cause = %s")
            params.append(query.cause)

        if query.charges:
            charge_clauses: list[str] = []
            for charge in query.charges[:3]:
                charge_clauses.append("charges_text ILIKE %s")
                params.append(f"%{charge}%")
            if charge_clauses:
                clauses.append("(" + " OR ".join(charge_clauses) + ")")

        if not clauses:
            return "", tuple()
        return "WHERE " + " AND ".join(clauses), tuple(params)

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        assert self.connection is not None
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            if not rows:
                return []
            columns = [description[0] for description in cursor.description]
        return [self._row_to_dict(columns, row) for row in rows]

    def _fetch_optional(
        self,
        sql: str,
        params: tuple[Any, ...],
        *,
        swallow_missing_table: bool = False,
    ) -> dict[str, Any] | None:
        try:
            rows = self._fetchall(sql, params)
        except Exception:
            if swallow_missing_table:
                return None
            raise
        return rows[0] if rows else None

    @staticmethod
    def _row_to_dict(columns: list[str], row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "_mapping"):
            return dict(row._mapping)
        return {column: value for column, value in zip(columns, row)}

    def _row_to_case(self, row: dict[str, Any]) -> StructuredCase:
        return StructuredCase(
            case_id=str(row["case_id"]),
            source_name=str(row["source_name"]),
            title=str(row["title"]),
            legal_domain=str(row["legal_domain"]),
            cause=str(row["cause"]),
            charges=_json_loads(str(row["charges_json"])),
            case_summary=str(row["case_summary"]),
            retrieval_text=str(row["retrieval_text"]),
            dispute_points=_json_loads(str(row["dispute_points_json"])),
            dispute_focus=str(row["dispute_focus"]),
            key_facts=_json_loads(str(row["key_facts_json"])),
            requested_relief=_json_loads(str(row["requested_relief_json"])),
            legal_basis=_json_loads(str(row["legal_basis_json"])),
            four_element_subject=_json_loads(str(row["four_element_subject_json"])),
            four_element_object=_json_loads(str(row["four_element_object_json"])),
            four_element_objective_aspect=_json_loads(str(row["four_element_objective_aspect_json"])),
            four_element_subjective_aspect=_json_loads(str(row["four_element_subjective_aspect_json"])),
            court_reasoning=str(row["court_reasoning"]),
            traceability_quote=str(row["traceability_quote"]),
            keywords=_json_loads(str(row["keywords_json"])),
        )


@dataclass
class PostgresPGVectorCandidateRepository:
    source_path: Path
    dsn: str
    encoder: object
    store: PostgresPGVectorCaseStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = PostgresPGVectorCaseStore(source_path=self.source_path, dsn=self.dsn, encoder=self.encoder)
        self.store.ensure_ready()

    def candidate_cases(self, query: StructuredQuery, limit: int) -> list[StructuredCase]:
        candidate_ids = self.store.candidate_ids(query, limit=limit)
        return self.store.fetch_cases(candidate_ids)
