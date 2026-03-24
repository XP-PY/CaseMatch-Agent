# Schema

本文档描述当前代码实际使用的 4 套 schema：

1. `corpus_merged.jsonl` 输入 schema
2. `StructuredQuery` 查询 schema
3. `StructuredCase` 内部案件 schema
4. LanceDB / SQLite 实际存储 schema

当前版本默认语料路径是：

```text
data/process/lecard/corpus_merged.jsonl
```

当前版本只面向刑事类案检索，不再区分民商事和刑事 schema。

当前实现里，`corpus_merged.jsonl` 仍然是源数据文件，不是纯历史产物：

- LanceDB 建库 / 重建时从它读取
- SQLite fallback 重建时从它读取
- 新增案件时会先把新记录追加到这个文件，再同步数据库

## 1. `corpus_merged.jsonl` 输入 schema

每行是一个 JSON object：

```json
{
  "case_id": "8",
  "structured_data": {
    "case_summary": "...",
    "dispute_focus": "...",
    "four_elements": {
      "subject": ["..."],
      "object": ["..."],
      "objective_aspect": ["..."],
      "subjective_aspect": ["..."]
    },
    "court_reasoning": "...",
    "laws_and_charges": {
      "charges": ["..."],
      "applicable_laws": ["..."]
    },
    "traceability": {
      "reasoning_quote": "..."
    }
  },
  "raw_data": {
    "case_name": "...",
    "document_name": "...",
    "fact_text": "...",
    "judgment_text": "...",
    "full_text": "..."
  }
}
```

### 1.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `case_id` | `string` | 是 | 案件唯一标识 |
| `structured_data` | `object` | 是 | 用于检索和向量化的结构化字段 |
| `raw_data` | `object` | 否 | 用于结果展示的原文字段 |

### 1.2 `structured_data`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `case_summary` | `string` | 否 | 案情摘要 |
| `dispute_focus` | `string` | 否 | 刑事争点 |
| `four_elements` | `object` | 否 | 刑法四要件 |
| `court_reasoning` | `string` | 否 | 裁判说理摘要 |
| `laws_and_charges` | `object` | 否 | 罪名和法条 |
| `traceability` | `object` | 否 | 可追溯引用信息 |

### 1.3 `structured_data.four_elements`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `subject` | `string[]` | 否 | 四要件中的主体 |
| `object` | `string[]` | 否 | 四要件中的客体 |
| `objective_aspect` | `string[]` | 否 | 四要件中的客观方面 |
| `subjective_aspect` | `string[]` | 否 | 四要件中的主观方面 |

### 1.4 `structured_data.laws_and_charges`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `charges` | `string[]` | 否 | 罪名 |
| `applicable_laws` | `string[]` | 否 | 适用法条 |

### 1.5 `structured_data.traceability`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `reasoning_quote` | `string` | 否 | 裁判说理引用片段 |

### 1.6 `raw_data`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `case_name` | `string` | 否 | 案件名称 |
| `document_name` | `string` | 否 | 法律文书名称 |
| `fact_text` | `string` | 否 | 案件事实 / 基本案情 |
| `judgment_text` | `string` | 否 | 裁判结果 |
| `full_text` | `string` | 否 | 文书全文 |

### 1.7 新增案件导入输入 schema

新增案件脚本 `scripts/add_cases_to_db.py` 接收的输入 jsonl，每行可以是下面任一形式：

形式 A，直接提供 `raw_data` 同构字段：

```json
{
  "case_name": "...",
  "document_name": "...",
  "fact_text": "...",
  "judgment_text": "...",
  "full_text": "..."
}
```

形式 B，显式包一层 `raw_data`：

```json
{
  "raw_data": {
    "case_name": "...",
    "document_name": "...",
    "fact_text": "...",
    "judgment_text": "...",
    "full_text": "..."
  }
}
```

导入时会：

1. 调用 LLM 提取 `structured_data`
2. 生成新的随机 `case_id`
3. 追加写入 `corpus_merged.jsonl`
4. 同步 LanceDB 或 SQLite

## 2. `StructuredQuery` 查询 schema

定义在 [models.py](/home/yuel_p/workspace/CaseMatch-Agent/src/casematch_agent/models.py)。

```python
StructuredQuery(
    raw_query: str,
    case_summary: str = "",
    charges: list[str] = [],
    dispute_focus: str = "",
    legal_basis: list[str] = [],
    four_element_subject: list[str] = [],
    four_element_object: list[str] = [],
    four_element_objective_aspect: list[str] = [],
    four_element_subjective_aspect: list[str] = [],
    court_reasoning: str = "",
    confidence: float = 0.0,
)
```

### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `raw_query` | `string` | 用户原始输入，始终保留 |
| `case_summary` | `string` | 用户查询的摘要化表达 |
| `charges` | `string[]` | 查询中的罪名 |
| `dispute_focus` | `string` | 查询中的刑事争点 |
| `legal_basis` | `string[]` | 查询中提到的法条 |
| `four_element_subject` | `string[]` | 查询中的主体信息 |
| `four_element_object` | `string[]` | 查询中的客体信息 |
| `four_element_objective_aspect` | `string[]` | 查询中的客观行为特征 |
| `four_element_subjective_aspect` | `string[]` | 查询中的主观方面 |
| `court_reasoning` | `string` | 查询里显式提到的裁判说理 |
| `confidence` | `float` | 提取置信度，范围 `0.0 ~ 1.0` |

### 多轮合并规则

`StructuredQuery.merge()` 的规则是：

- `raw_query`：前后轮原始文本拼接
- `case_summary`：新值优先，无新值则保留旧值
- 单值字段：新值优先
- 列表字段：去重合并
- `confidence`：取两轮中的较大值

## 3. `StructuredCase` 内部案件 schema

定义在 [models.py](/home/yuel_p/workspace/CaseMatch-Agent/src/casematch_agent/models.py)。

```python
StructuredCase(
    case_id: str,
    case_name: str = "",
    document_name: str = "",
    fact_text: str = "",
    judgment_text: str = "",
    full_text: str = "",
    charges: list[str] = [],
    case_summary: str = "",
    dispute_focus: str = "",
    legal_basis: list[str] = [],
    four_element_subject: list[str] = [],
    four_element_object: list[str] = [],
    four_element_objective_aspect: list[str] = [],
    four_element_subjective_aspect: list[str] = [],
    court_reasoning: str = "",
    traceability_quote: str = "",
)
```

### 映射关系

| `StructuredCase` 字段 | 来源 |
| --- | --- |
| `case_id` | 顶层 `case_id` |
| `case_name` | `raw_data.case_name` |
| `document_name` | `raw_data.document_name` |
| `fact_text` | `raw_data.fact_text` |
| `judgment_text` | `raw_data.judgment_text` |
| `full_text` | `raw_data.full_text` |
| `charges` | `structured_data.laws_and_charges.charges` |
| `case_summary` | `structured_data.case_summary` |
| `dispute_focus` | `structured_data.dispute_focus` |
| `legal_basis` | `structured_data.laws_and_charges.applicable_laws` |
| `four_element_subject` | `structured_data.four_elements.subject` |
| `four_element_object` | `structured_data.four_elements.object` |
| `four_element_objective_aspect` | `structured_data.four_elements.objective_aspect` |
| `four_element_subjective_aspect` | `structured_data.four_elements.subjective_aspect` |
| `court_reasoning` | `structured_data.court_reasoning` |
| `traceability_quote` | `structured_data.traceability.reasoning_quote` |

## 4. LanceDB 存储 schema

实现位于 [lancedb_store.py](/home/yuel_p/workspace/CaseMatch-Agent/src/casematch_agent/lancedb_store.py)。

默认目录：

```text
data/process/cases.lancedb
```

当前 LanceDB `cases` 表实际写入这些字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | `string` | 案件 ID |
| `case_name` | `string` | 展示字段 |
| `document_name` | `string` | 展示字段 |
| `fact_text` | `string` | 展示字段 |
| `judgment_text` | `string` | 展示字段 |
| `full_text` | `string` | 展示字段 |
| `charges` | `string[]` | 结构化字段 |
| `charges_text` | `string` | 罪名扁平化文本，用于过滤 |
| `case_summary` | `string` | 结构化字段 |
| `dispute_focus` | `string` | 结构化字段 |
| `legal_basis` | `string[]` | 结构化字段 |
| `legal_basis_text` | `string` | 法条扁平化文本 |
| `four_element_subject` | `string[]` | 结构化字段 |
| `four_element_object` | `string[]` | 结构化字段 |
| `four_element_objective_aspect` | `string[]` | 结构化字段 |
| `four_element_subjective_aspect` | `string[]` | 结构化字段 |
| `court_reasoning` | `string` | 结构化字段 |
| `traceability_quote` | `string` | 引用字段 |
| `fused_text` | `string` | 向量召回文本 |
| `fused_embedding` | `float[]` | 向量召回 embedding |

### `fused_text` 生成规则

定义在 [search_profiles.py](/home/yuel_p/workspace/CaseMatch-Agent/src/casematch_agent/search_profiles.py)：

```text
fused_text = case_summary + dispute_focus + court_reasoning
```

### LanceDB 元数据文件

除了 `cases` 表，还会生成：

```text
data/process/cases.lancedb/.casematch_lancedb_metadata.json
```

当前只存：

| 字段 | 说明 |
| --- | --- |
| `source_signature` | 源 `jsonl` 的时间戳和大小签名，用于判断是否需要重建索引 |
| `table_name` | LanceDB 表名，当前默认是 `cases` |
| `case_count` | 本次建库写入的案件数 |
| `built_at_utc` | 最近一次建库或重建的 UTC 时间 |

## 5. SQLite fallback 存储 schema

实现位于 [sqlite_store.py](/home/yuel_p/workspace/CaseMatch-Agent/src/casematch_agent/sqlite_store.py)。

默认文件：

```text
data/process/cases.sqlite3
```

### 5.1 `cases` 主表

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | `TEXT PRIMARY KEY` | 案件 ID |
| `case_name` | `TEXT` | 展示字段 |
| `document_name` | `TEXT` | 展示字段 |
| `fact_text` | `TEXT` | 展示字段 |
| `judgment_text` | `TEXT` | 展示字段 |
| `full_text` | `TEXT` | 展示字段 |
| `charges_json` | `TEXT` | 罪名数组 JSON |
| `charges_text` | `TEXT` | 罪名扁平化文本 |
| `case_summary` | `TEXT` | 案情摘要 |
| `dispute_focus` | `TEXT` | 刑事争点 |
| `legal_basis_json` | `TEXT` | 法条数组 JSON |
| `legal_basis_text` | `TEXT` | 法条扁平化文本 |
| `four_element_subject_json` | `TEXT` | 主体数组 JSON |
| `four_element_object_json` | `TEXT` | 客体数组 JSON |
| `four_element_objective_aspect_json` | `TEXT` | 客观方面数组 JSON |
| `four_element_subjective_aspect_json` | `TEXT` | 主观方面数组 JSON |
| `court_reasoning` | `TEXT` | 裁判说理 |
| `traceability_quote` | `TEXT` | 引用片段 |

### 5.2 `cases_fts` FTS5 表

当前 FTS 索引这些文本列：

| 列 | 来源 |
| --- | --- |
| `charges_text` | 罪名 |
| `dispute_focus` | 刑事争点 |
| `case_summary` | 案情摘要 |
| `court_reasoning` | 裁判说理 |
| `legal_basis_text` | 法条 |
| `four_element_objective_aspect_text` | 客观方面 |
| `four_element_subjective_aspect_text` | 主观方面 |

### 5.3 SQLite coarse recall 查询侧字段

SQLite fallback 会从 `StructuredQuery` 中抽这些字段拼 FTS 查询：

- `charges`
- `four_element_subject`
- `four_element_object`
- `four_element_objective_aspect`
- `four_element_subjective_aspect`
- `legal_basis`
- `dispute_focus`
- `raw_query`

## 6. 结果返回 schema

agent 检索完成后，外部拿到的是 `RetrievalResult`：

```python
RetrievalResult(
    case: StructuredCase,
    total_score: float,
    field_scores: dict[str, float],
    reasons: list[str],
)
```

其中：

- `case` 已经包含 `raw_data` 里的 `case_name / fact_text / judgment_text / full_text`
- 当前 CLI 和 Gradio 页面都会展示这些字段
- 精排仍然只使用 `structured_data` 对应字段，不直接拿 `raw_data` 做排序特征

## 7. 当前 schema 约束

当前版本有几个明确约束：

1. 只支持刑事类案检索
2. 查询 schema 和案件 schema 都已经按 `structured_data` 收缩
3. 不再保留民商事字段
4. `raw_data` 只用于结果展示，不用于当前主检索逻辑
5. 如果后续接多语料，建议保持：
   - 检索字段进入统一 `StructuredCase`
   - 展示字段进入原文展示层
   - 不要把展示字段和检索字段混成一套 schema
