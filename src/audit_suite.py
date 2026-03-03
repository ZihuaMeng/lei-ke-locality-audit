"""
audit_suite.py
KE Efficacy + Locality Audit Pipeline
Milestone: M0.5 — NO_EDIT mode skeleton
"""

import json
import logging
import argparse
import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch

from model_utils import load_model, get_answer

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
    if model is None:
        return f"[STUB_ANSWER for: {query[:40]}]"

    tokenizer = getattr(model, "_audit_tokenizer", None)
    return get_answer(query, model, tokenizer)


# ── Stub: knowledge edit ───────────────────────────────────────────────────────
def apply_edit(
    model,
    subject: str,
    relation: str,
    original_object: str,
    new_object: str,
    query: Optional[str] = None,
    method: str = "ROME",
) -> object:
    """
    Apply a knowledge edit to model weights.

    For 4-bit quantized models, ROME may fail to update quantized parameters.
    In that case this function reloads a full-precision bf16 model and retries.
    """
    if method != "ROME":
        raise NotImplementedError(f"Method {method} is not implemented yet.")

    rome_root = Path(__file__).resolve().parent / "rome"
    if str(rome_root) not in sys.path:
        sys.path.insert(0, str(rome_root))

    from rome import ROMEHyperParams, apply_rome_to_model

    tokenizer = getattr(model, "_audit_tokenizer", None)
    if tokenizer is None:
        raise ValueError("Model tokenizer not found on model._audit_tokenizer")

    if query and subject in query:
        prompt = f"Q: {query.replace(subject, '{}')}\nA:"
    else:
        relation_text = relation.replace("_", " ").strip()
        if relation_text:
            if relation_text.startswith(("is ", "was ", "are ", "were ")):
                predicate = relation_text
            else:
                predicate = f"is {relation_text}"
            prompt = f"{{}} {predicate}"
        else:
            prompt = "{}"

    num_layers = len(getattr(getattr(model, "model", None), "layers", []))
    if num_layers == 0:
        raise ValueError("Unsupported model architecture for ROME: expected model.model.layers")

    rewrite_layer = min((num_layers * 3) // 4, num_layers - 1)
    hparams = ROMEHyperParams(
        layers=[rewrite_layer],
        fact_token="subject_last",
        v_num_grad_steps=40,
        v_lr=5e-1,
        v_loss_layer=num_layers - 1,
        v_weight_decay=0.0,
        clamp_norm_factor=10,
        kl_factor=0.0,
        mom2_adjustment=False,
        context_template_length_params=[[5, 10], [10, 10]],
        rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        mlp_module_tmp="model.layers.{}.mlp",
        attn_module_tmp="model.layers.{}.self_attn",
        ln_f_module="model.norm",
        lm_head_module="lm_head",
        mom2_dataset="wikipedia",
        mom2_n_samples=100000,
        mom2_dtype="float32",
    )

    request = {
        "prompt": prompt,
        "subject": subject,
        "target_new": {"str": new_object},
        "target_true": {"str": original_object},
    }

    quantized = bool(getattr(model, "is_loaded_in_4bit", False))
    if quantized:
        logger.warning("ROME update on 4-bit quantized model is unsupported; switching to bf16 fallback model.")
        model_name = getattr(model, "_audit_model_name", None)
        if model_name is None:
            raise RuntimeError("Cannot fallback reload model: missing model._audit_model_name")

        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fallback_model, fallback_tokenizer = load_model(model_name, use_4bit=False)
        fallback_model._audit_tokenizer = fallback_tokenizer
        fallback_model._audit_model_name = model_name
        fallback_model._audit_use_4bit = False
        model = fallback_model
        tokenizer = fallback_tokenizer

    try:
        model, _ = apply_rome_to_model(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=False,
        )
        logger.info("ROME edit applied on currently loaded model.")
        return model
    except Exception as exc:
        raise RuntimeError(f"ROME apply_edit failed: {exc}") from exc


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
            model = apply_edit(
                model,
                subject,
                relation,
                orig_obj,
                new_obj,
                query=query,
                method=mode,
            )
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
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Load model with 4-bit quantization",
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

    model = None
    if args.mode != "NO_EDIT":
        model, tokenizer = load_model(args.model_name, use_4bit=args.use_4bit)
        model._audit_tokenizer = tokenizer
        model._audit_model_name = args.model_name
        model._audit_use_4bit = args.use_4bit

    prompts = load_prompts(args.prompt_dir)
    results = run_audit(prompts, model=model, mode=args.mode, output_dir=args.output_dir)

    logger.info("audit_suite.py finished. Run src/report.py to generate audit_report.md")
