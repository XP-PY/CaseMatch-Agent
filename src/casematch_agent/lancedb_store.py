from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import load_lecard_corpus
from .models import StructuredCase, StructuredQuery
from .search_profiles import case_fused_text, query_fused_text

CASE_TABLE = "cases"
METADATA_FILENAME = ".casematch_lancedb_metadata.json"


def _import_lancedb():
    try:
        import lancedb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("LanceDB backend requires lancedb. Install it first.") from exc
    return lancedb


def _source_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted if str(item)]
    return []


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


@dataclass
class LanceDBBuildReport:
    db_uri: Path
    table_name: str
    metadata_path: Path
    rebuilt: bool
    case_count: int
    source_signature: str
    built_at_utc: str


@dataclass
class LanceDBCaseStore:
    source_path: Path
    db_uri: Path
    encoder: object
    table_name: str = CASE_TABLE
    connection: object | None = field(default=None, init=False)
    table: object | None = field(default=None, init=False)
    _lancedb: object | None = field(default=None, init=False, repr=False)

    @property
    def metadata_path(self) -> Path:
        return self.db_uri / METADATA_FILENAME

    def ensure_ready(self) -> None:
        self.build(force_rebuild=False)

    def build(self, *, force_rebuild: bool = False) -> LanceDBBuildReport:
        self._connect_if_needed()
        rebuilt = force_rebuild or self._needs_rebuild()
        if rebuilt:
            case_count = self._rebuild()
        else:
            assert self.connection is not None
            if self.table is None:
                self.table = self.connection.open_table(self.table_name)
            metadata = self._read_metadata()
            case_count = int(metadata.get("case_count", 0))

        metadata = self._read_metadata()
        return LanceDBBuildReport(
            db_uri=self.db_uri,
            table_name=self.table_name,
            metadata_path=self.metadata_path,
            rebuilt=rebuilt,
            case_count=case_count,
            source_signature=str(metadata.get("source_signature", "")),
            built_at_utc=str(metadata.get("built_at_utc", "")),
        )

    def candidate_rows(self, query: StructuredQuery, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self.table is not None

        query_text = query_fused_text(query)
        if not query_text:
            return []

        vectors = self.encoder.encode([query_text])
        if not vectors:
            return []

        search = self.table.search(vectors[0], vector_column_name="fused_embedding")
        filter_clause = self._build_filter_clause(query)
        if filter_clause:
            search = search.where(filter_clause, prefilter=True)
        arrow_result = search.limit(limit).to_arrow()
        return arrow_result.to_pylist() if hasattr(arrow_result, "to_pylist") else []

    def add_cases(self, cases: list[StructuredCase], *, assume_ready: bool = False) -> int:
        if not cases:
            return 0

        if assume_ready:
            self._connect_if_needed()
            if self.table is None:
                assert self.connection is not None
                self.table = self.connection.open_table(self.table_name)
        else:
            self.ensure_ready()

        assert self.table is not None
        rows = self._rows_from_cases(cases)
        self.table.add(rows)

        previous_count = int(self._read_metadata().get("case_count", 0))
        current_count = self._table_row_count(default=previous_count + len(rows))
        self._write_metadata(case_count=current_count)
        return len(rows)

    def _connect_if_needed(self) -> None:
        if self.connection is None:
            self.db_uri.mkdir(parents=True, exist_ok=True)
            self._lancedb = _import_lancedb()
            self.connection = self._lancedb.connect(str(self.db_uri))

    def _read_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_metadata(self, *, case_count: int) -> None:
        self.metadata_path.write_text(
            json.dumps(
                {
                    "source_signature": _source_signature(self.source_path),
                    "table_name": self.table_name,
                    "case_count": case_count,
                    "built_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _needs_rebuild(self) -> bool:
        metadata = self._read_metadata()
        if not metadata:
            return True
        if metadata.get("source_signature") != _source_signature(self.source_path):
            return True
        if metadata.get("table_name") != self.table_name:
            return True
        if not isinstance(metadata.get("case_count"), int):
            return True
        try:
            assert self.connection is not None
            self.table = self.connection.open_table(self.table_name)
        except Exception:
            return True
        return False

    def _rebuild(self) -> int:
        assert self.connection is not None
        cases = load_lecard_corpus(self.source_path)
        if not cases:
            raise RuntimeError(f"No cases loaded from {self.source_path}.")
        rows = self._rows_from_cases(cases)

        self.table = self.connection.create_table(self.table_name, data=rows, mode="overwrite")
        self._create_indices()
        self._write_metadata(case_count=len(rows))
        return len(rows)

    def _create_indices(self) -> None:
        assert self.table is not None
        try:
            self.table.create_scalar_index("charges_text")
        except Exception:
            pass
        try:
            self.table.create_index(metric="cosine", vector_column_name="fused_embedding")
        except Exception:
            pass

    def _build_filter_clause(self, query: StructuredQuery) -> str:
        clauses: list[str] = []
        if query.charges:
            charge_clauses = [
                f"charges_text LIKE '%{_sql_quote(charge)}%'" for charge in query.charges[:3] if charge.strip()
            ]
            if charge_clauses:
                clauses.append("(" + " OR ".join(charge_clauses) + ")")
        return " AND ".join(clauses)

    def _rows_from_cases(self, cases: list[StructuredCase]) -> list[dict[str, Any]]:
        fused_texts = [case_fused_text(case) for case in cases]
        fused_vectors = self.encoder.encode(fused_texts)
        if not fused_vectors:
            raise RuntimeError("Failed to build fused embeddings for LanceDB store.")

        rows = []
        for case, fused_text, vector in zip(cases, fused_texts, fused_vectors):
            rows.append(
                {
                    "case_id": case.case_id,
                    "case_name": case.case_name,
                    "document_name": case.document_name,
                    "fact_text": case.fact_text,
                    "judgment_text": case.judgment_text,
                    "full_text": case.full_text,
                    "charges": case.charges,
                    "charges_text": " ".join(case.charges),
                    "case_summary": case.case_summary,
                    "dispute_focus": case.dispute_focus,
                    "legal_basis": case.legal_basis,
                    "legal_basis_text": " ".join(case.legal_basis),
                    "four_element_subject": case.four_element_subject,
                    "four_element_object": case.four_element_object,
                    "four_element_objective_aspect": case.four_element_objective_aspect,
                    "four_element_subjective_aspect": case.four_element_subjective_aspect,
                    "court_reasoning": case.court_reasoning,
                    "traceability_quote": case.traceability_quote,
                    "fused_text": fused_text,
                    "fused_embedding": [float(value) for value in vector],
                }
            )
        return rows

    def _table_row_count(self, *, default: int) -> int:
        assert self.table is not None
        count_rows = getattr(self.table, "count_rows", None)
        if callable(count_rows):
            try:
                return int(count_rows())
            except Exception:
                return default
        return default

    def row_to_case(self, row: dict[str, Any]) -> StructuredCase:
        return StructuredCase(
            case_id=str(row["case_id"]),
            case_name=str(row.get("case_name", "")),
            document_name=str(row.get("document_name", "")),
            fact_text=str(row.get("fact_text", "")),
            judgment_text=str(row.get("judgment_text", "")),
            full_text=str(row.get("full_text", "")),
            charges=_list_of_str(row.get("charges")),
            case_summary=str(row.get("case_summary", "")),
            dispute_focus=str(row.get("dispute_focus", "")),
            legal_basis=_list_of_str(row.get("legal_basis")),
            four_element_subject=_list_of_str(row.get("four_element_subject")),
            four_element_object=_list_of_str(row.get("four_element_object")),
            four_element_objective_aspect=_list_of_str(row.get("four_element_objective_aspect")),
            four_element_subjective_aspect=_list_of_str(row.get("four_element_subjective_aspect")),
            court_reasoning=str(row.get("court_reasoning", "")),
            traceability_quote=str(row.get("traceability_quote", "")),
        )


@dataclass
class LanceDBCandidateRepository:
    source_path: Path
    db_uri: Path
    encoder: object
    store: LanceDBCaseStore = field(init=False)

    def __post_init__(self) -> None:
        self.store = LanceDBCaseStore(source_path=self.source_path, db_uri=self.db_uri, encoder=self.encoder)
        self.store.ensure_ready()

    def candidate_cases(self, query: StructuredQuery, limit: int) -> list[StructuredCase]:
        rows = self.store.candidate_rows(query, limit=limit)
        return [self.store.row_to_case(row) for row in rows]
