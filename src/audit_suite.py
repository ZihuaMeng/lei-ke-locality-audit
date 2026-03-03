"""
audit_suite.py
KE Efficacy + Locality Audit Pipeline
Milestone: M0.5 — NO_EDIT mode skeleton
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Prompt loading ─────────────────────────────────────────────────────────────
def load_prompts(prompt_dir: Path) -> dict[str, list[str]]:
    """Load all prompt files from data/prompts/{split}/prompts.txt"""
    splits = ["rewrite", "neighborhood", "distractor"]
    prompts: dict[str, list[str]] = {}
    for split in splits:
        path = prompt_dir / split / "prompts.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file missing: {path}")
        lines = [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        prompts[split] = lines
        logger.info(f"Loaded {len(lines)} {split} prompts from {path}")
    return prompts


# ── Stub: model interaction ────────────────────────────────────────────────────
def get_model_answer(query: str, model=None) -> str:
    """
    STUB: Return a placeholder answer.
    Replace with real model forward-pass in M1.
    """
    return f"[STUB_ANSWER for: {query[:40]}]"


# ── Stub: knowledge edit ───────────────────────────────────────────────────────
def apply_edit(
    model,
    subject: str,
    relation: str,
    original_object: str,
    new_object: str,
    method: str = "ROME",
) -> None:
    """
    Apply a knowledge edit to model weights.
    NOT IMPLEMENTED — will be filled in M1 (ROME) and M2 (MEMIT).
    """
    raise NotImplementedError(
        f"apply_edit() is not implemented yet. "
        f"Planned: {method} edit [{subject} / {relation}: {original_object} -> {new_object}]. "
        "Run with --mode NO_EDIT to skip this step."
    )


# ── Core audit logic ───────────────────────────────────────────────────────────
def run_audit(
    prompts: dict[str, list[str]],
    model=None,
    mode: str = "NO_EDIT",
    output_dir: Path = Path("outputs"),
) -> dict:
    """
    Main audit loop.

    mode="NO_EDIT": skip apply_edit(), use stub answers (smoke test)
    mode="ROME":    call apply_edit(..., method="ROME")  [M1]
    mode="MEMIT":   call apply_edit(..., method="MEMIT") [M2]
    """
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Starting audit | mode={mode}")

    results = {
        "meta": {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "n_rewrite": len(prompts["rewrite"]),
            "n_neighborhood": len(prompts["neighborhood"]),
            "n_distractor": len(prompts["distractor"]),
        },
        "rewrite": [],
        "neighborhood": [],
        "distractor": [],
        "summary": {},
    }

    # ── Rewrite prompts (efficacy) ─────────────────────────────────────────────
    for raw_line in prompts["rewrite"]:
        parts = [p.strip() for p in raw_line.split("|||")]
        if len(parts) != 5:
            logger.warning(f"Skipping malformed rewrite line: {raw_line}")
            continue
        subject, relation, orig_obj, new_obj, query = parts

        c_orig = get_model_answer(query, model)
        logger.info(f"[REWRITE] C_orig: {c_orig}")

        if mode != "NO_EDIT":
            apply_edit(model, subject, relation, orig_obj, new_obj, method=mode)
            c_star = get_model_answer(query, model)
        else:
            c_star = "[NO_EDIT_MODE — edit skipped]"

        edit_success = (c_star != c_orig) if mode != "NO_EDIT" else None
        results["rewrite"].append(
            {
                "subject": subject,
                "query": query,
                "c_orig": c_orig,
                "c_star": c_star,
                "edit_success": edit_success,
            }
        )

    # ── Neighborhood prompts (locality) ───────────────────────────────────────
    for raw_line in prompts["neighborhood"]:
        parts = [p.strip() for p in raw_line.split("|||")]
        if len(parts) != 2:
            logger.warning(f"Skipping malformed neighborhood line: {raw_line}")
            continue
        query, expected = parts
        answer = get_model_answer(query, model)
        stable = (answer == expected) if mode != "NO_EDIT" else None
        results["neighborhood"].append(
            {"query": query, "expected": expected, "answer": answer, "stable": stable}
        )

    # ── Distractor prompts (side effects) ─────────────────────────────────────
    for raw_line in prompts["distractor"]:
        parts = [p.strip() for p in raw_line.split("|||")]
        if len(parts) != 2:
            logger.warning(f"Skipping malformed distractor line: {raw_line}")
            continue
        query, expected = parts
        answer = get_model_answer(query, model)
        unchanged = (answer == expected) if mode != "NO_EDIT" else None
        results["distractor"].append(
            {
                "query": query,
                "expected": expected,
                "answer": answer,
                "unchanged": unchanged,
            }
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    if mode != "NO_EDIT":
        efficacy = sum(r["edit_success"] for r in results["rewrite"] if r["edit_success"] is not None)
        locality = sum(r["stable"] for r in results["neighborhood"] if r["stable"] is not None)
        side_eff = sum(r["unchanged"] for r in results["distractor"] if r["unchanged"] is not None)
        results["summary"] = {
            "efficacy_rate": efficacy / max(len(results["rewrite"]), 1),
            "locality_rate": locality / max(len(results["neighborhood"]), 1),
            "distractor_unchanged_rate": side_eff / max(len(results["distractor"]), 1),
        }
    else:
        results["summary"] = {
            "mode": "NO_EDIT — rates not computed",
            "smoke_test": "PASS" if results["rewrite"] else "FAIL",
        }
        logger.info("NO_EDIT mode: smoke test complete, rates not computed.")

    # ── Save results.json ─────────────────────────────────────────────────────
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="KE Locality Audit Suite")
    parser.add_argument(
        "--mode",
        choices=["NO_EDIT", "ROME", "MEMIT"],
        default="NO_EDIT",
        help="Editing mode (default: NO_EDIT for smoke test)",
    )
    parser.add_argument(
        "--prompt_dir",
        type=Path,
        default=Path("data/prompts"),
        help="Root directory for prompt files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for results.json and reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"audit_suite.py started | mode={args.mode}")

    prompts = load_prompts(args.prompt_dir)
    results = run_audit(prompts, model=None, mode=args.mode, output_dir=args.output_dir)

    logger.info("audit_suite.py finished. Run src/report.py to generate audit_report.md")
