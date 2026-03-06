#!/usr/bin/env python3
"""Build audit prompt files from the CounterFact dataset on Hugging Face."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = REPO_ROOT / "data" / "prompts"
REWRITE_PATH = PROMPTS_ROOT / "rewrite" / "prompts.txt"
NEIGHBORHOOD_PATH = PROMPTS_ROOT / "neighborhood" / "prompts.txt"
DISTRACTOR_PATH = PROMPTS_ROOT / "distractor" / "prompts.txt"


def _clean_text(value: object) -> str:
    if isinstance(value, dict) and "str" in value:
        value = value["str"]
    text = "" if value is None else str(value)
    text = text.replace("|||", " / ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _render_query(prompt_template: str, subject: str) -> str:
    template = _clean_text(prompt_template)
    subject_text = _clean_text(subject)
    if not template:
        return subject_text
    if "{" not in template:
        return template
    try:
        rendered = template.format(subject_text)
    except Exception:
        rendered = template.replace("{}", subject_text).replace("{subject}", subject_text)
    return _clean_text(rendered)


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = list(lines)
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return len(rendered)


def _supports_case_id(ds: Dataset) -> bool:
    columns = set(ds.column_names)
    return {"case_id", "requested_rewrite", "neighborhood_prompts"}.issubset(columns)


def _load_via_hub_jsonl_fallback() -> tuple[Dataset, str]:
    parquet_path = hf_hub_download(
        repo_id="azhx/counterfact",
        repo_type="dataset",
        filename="data/train-00000-of-00001-05d11247db7abce8.parquet",
    )

    table = pq.read_table(parquet_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        temp_jsonl = Path(handle.name)
        for batch in table.to_batches(max_chunksize=512):
            for row in batch.to_pylist():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        ds = load_dataset("json", data_files=str(temp_jsonl), split="train")
    finally:
        temp_jsonl.unlink(missing_ok=True)

    return ds, "azhx/counterfact (hf_hub parquet -> local jsonl compatibility fallback)"


def load_counterfact() -> tuple[Dataset, str]:
    attempts = [
        ("zjunlp/KnowEdit", "CounterFact"),
        ("azhx/counterfact", None),
        ("cfact", None),
    ]
    errors: list[str] = []

    for dataset_id, config_name in attempts:
        try:
            if config_name is None:
                ds = load_dataset(dataset_id, split="train")
            else:
                ds = load_dataset(dataset_id, config_name, split="train")
        except Exception as exc:
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            continue

        if _supports_case_id(ds):
            return ds, dataset_id if config_name is None else f"{dataset_id}/{config_name}"

        errors.append(f"{dataset_id}: loaded, but missing required CounterFact columns")

    try:
        ds, source = _load_via_hub_jsonl_fallback()
    except Exception as exc:
        errors.append(f"azhx/counterfact fallback: {type(exc).__name__}: {exc}")
    else:
        if _supports_case_id(ds):
            return ds, source
        errors.append("azhx/counterfact fallback: loaded, but missing required CounterFact columns")

    joined = "\n".join(f"- {item}" for item in errors)
    raise RuntimeError(f"Unable to load a usable CounterFact dataset.\n{joined}")


def _has_valid_requested_rewrite(record: dict) -> bool:
    rewrite = record.get("requested_rewrite") or {}
    return bool(
        record.get("case_id") is not None
        and rewrite.get("subject")
        and rewrite.get("prompt")
        and rewrite.get("relation_id")
        and rewrite.get("target_true", {}).get("str")
        and rewrite.get("target_new", {}).get("str")
    )


def _is_valid_rewrite_record(record: dict) -> bool:
    return _has_valid_requested_rewrite(record) and bool(record.get("neighborhood_prompts"))


def _is_valid_distractor_record(record: dict) -> bool:
    rewrite = record.get("requested_rewrite") or {}
    return bool(
        record.get("case_id") is not None
        and rewrite.get("subject")
        and rewrite.get("prompt")
        and rewrite.get("target_true", {}).get("str")
    )


def _build_rewrite_line(record: dict) -> str:
    rewrite = record["requested_rewrite"]
    subject = _clean_text(rewrite["subject"])
    relation = _clean_text(rewrite["relation_id"])
    original_object = _clean_text(rewrite["target_true"]["str"])
    new_object = _clean_text(rewrite["target_new"]["str"])
    query = _render_query(rewrite["prompt"], rewrite["subject"])
    return f"{subject} ||| {relation} ||| {original_object} ||| {new_object} ||| {query}"


def _build_neighborhood_line(record: dict) -> str:
    query = _clean_text(record["neighborhood_prompts"][0])
    expected_answer = _clean_text(record["requested_rewrite"]["target_true"]["str"])
    return f"{query} ||| {expected_answer}"


def _build_distractor_line(record: dict) -> str:
    rewrite = record["requested_rewrite"]
    query = _render_query(rewrite["prompt"], rewrite["subject"])
    expected_answer = _clean_text(rewrite["target_true"]["str"])
    return f"{query} ||| {expected_answer}"


def _build_distractor_candidates(record: dict) -> list[str]:
    rewrite = record["requested_rewrite"]
    expected_answer = _clean_text(rewrite["target_true"]["str"])
    candidates = [f"{_render_query(rewrite['prompt'], rewrite['subject'])} ||| {expected_answer}"]
    for prompt in record.get("paraphrase_prompts") or []:
        candidates.append(f"{_clean_text(prompt)} ||| {expected_answer}")
    return candidates


def main() -> None:
    ds, source = load_counterfact()

    rewrite_records: list[dict] = []
    distractor_band_records: list[dict] = []
    for record in ds:
        case_id = record.get("case_id")
        if case_id is None:
            continue

        if case_id < 5000 and len(rewrite_records) < 100 and _is_valid_rewrite_record(record):
            rewrite_records.append(record)

        if 8000 <= case_id <= 8099 and _is_valid_distractor_record(record):
            distractor_band_records.append(record)

        if len(rewrite_records) == 100 and case_id > 8099:
            break

    if len(rewrite_records) != 100:
        raise RuntimeError(f"Expected 100 rewrite records with case_id < 5000, found {len(rewrite_records)}")

    distractor_lines: list[str] = []
    seen_distractors: set[str] = set()
    for record in distractor_band_records:
        line = _build_distractor_line(record)
        if line not in seen_distractors:
            distractor_lines.append(line)
            seen_distractors.add(line)

    extra_distractor_queries = 0
    if len(distractor_lines) < 100:
        for record in distractor_band_records:
            for line in _build_distractor_candidates(record)[1:]:
                if line in seen_distractors:
                    continue
                distractor_lines.append(line)
                seen_distractors.add(line)
                extra_distractor_queries += 1
                if len(distractor_lines) == 100:
                    break
            if len(distractor_lines) == 100:
                break

    if len(distractor_lines) != 100:
        raise RuntimeError(
            "Expected 100 distractor prompts from case_id 8000-8099, "
            f"found {len(distractor_lines)} after adding paraphrases"
        )

    rewrite_count = _write_lines(REWRITE_PATH, (_build_rewrite_line(record) for record in rewrite_records))
    neighborhood_count = _write_lines(
        NEIGHBORHOOD_PATH,
        (_build_neighborhood_line(record) for record in rewrite_records),
    )
    distractor_count = _write_lines(DISTRACTOR_PATH, distractor_lines)

    print(f"Loaded CounterFact source: {source}")
    print(f"rewrite: {rewrite_count} lines -> {REWRITE_PATH.relative_to(REPO_ROOT)}")
    print(f"neighborhood: {neighborhood_count} lines -> {NEIGHBORHOOD_PATH.relative_to(REPO_ROOT)}")
    print(f"distractor: {distractor_count} lines -> {DISTRACTOR_PATH.relative_to(REPO_ROOT)}")
    if extra_distractor_queries:
        print(
            "distractor detail: "
            f"{len(distractor_band_records)} main prompts + {extra_distractor_queries} paraphrases from case_id 8000-8099"
        )


if __name__ == "__main__":
    main()
