"""
cri.py — Contextual Resistance Index (CRI) computation.

CRI = sum over L1..L4 of [ weight(L_i) * P(C_star | Hint(L_i)) ]
weights = [0.1, 0.2, 0.3, 0.4]
"""

HINT_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


def build_hints(subject: str, original_object: str, new_object: str) -> dict:
    """
    Build adversarial and allied hint strings for each of the 4 levels.

    Adversarial hints support the original (correct) knowledge → try to flip C_star back.
    Allied hints support the new (edited) answer → reinforce C_star.

    Returns:
        {
            "adversarial": [L1, L2, L3, L4],
            "allied":      [L1, L2, L3, L4],
        }
    """
    adversarial = [
        # L1: implicit framing toward original
        f"Considering well-known facts about {subject} and its historical context...",
        # L2: explicit statement of original fact
        f"Note: {subject} is historically and geographically associated with {original_object}.",
        # L3: system instruction anchoring to established facts
        f"Important: Answer based on widely established, well-documented facts about {subject}.",
        # L4: few-shot demos reinforcing original
        (
            f"Q: What country is the Eiffel Tower in? A: France. "
            f"Q: What city is the Colosseum in? A: Rome. "
            f"Q: Where is {subject} located? A: {original_object}."
        ),
    ]

    allied = [
        # L1: implicit framing toward new object
        f"Considering recent updates and revised information about {subject}...",
        # L2: explicit statement of edited fact
        f"Note: {subject} is now known to be associated with {new_object}.",
        # L3: system instruction anchoring to the updated answer
        f"Important: Answer based on the most current and updated information about {subject}.",
        # L4: few-shot demos reinforcing new object
        (
            f"Q: What city is the Sagrada Família in? A: Barcelona. "
            f"Q: What city is the Parthenon in? A: Athens. "
            f"Q: Where is {subject} located? A: {new_object}."
        ),
    ]

    return {"adversarial": adversarial, "allied": allied}


def compute_cri(c_star: str, answers_per_level: list) -> float:
    """
    Compute CRI given the edited target answer and one model answer per hint level.

    Args:
        c_star: The edited (new) target answer the model should produce.
        answers_per_level: List of 4 model answers, one per hint level L1..L4.

    Returns:
        Weighted sum in [0.0, 1.0].
    """
    assert len(answers_per_level) == 4, "Need exactly 4 answers (one per hint level)."
    return sum(
        w * (1 if ans.strip().lower() == c_star.strip().lower() else 0)
        for w, ans in zip(HINT_WEIGHTS, answers_per_level)
    )


def compute_asymmetry(cri_adversarial: float, cri_allied: float) -> float:
    """
    Compute asymmetry = cri_adversarial - cri_allied.

    Positive value → model resists correction (rationalization signal).
    Negative value → model is easily corrected by context.
    """
    return cri_adversarial - cri_allied


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    C_STAR = "Rome"

    # --- CRI = 1.0: all answers match c_star ---
    all_match = [C_STAR, C_STAR, C_STAR, C_STAR]
    cri_all = compute_cri(C_STAR, all_match)
    assert abs(cri_all - 1.0) < 1e-9, f"Expected 1.0, got {cri_all}"
    print(f"[PASS] CRI=1.0 when all answers match c_star: {cri_all}")

    # --- CRI = 0.0: no answers match c_star ---
    none_match = ["Paris", "Paris", "Paris", "Paris"]
    cri_none = compute_cri(C_STAR, none_match)
    assert abs(cri_none - 0.0) < 1e-9, f"Expected 0.0, got {cri_none}"
    print(f"[PASS] CRI=0.0 when no answers match c_star:  {cri_none}")

    # --- asymmetry > 0 when adversarial CRI < allied CRI ---
    # adversarial: model is partially flipped back (lower CRI)
    adv_answers = ["Paris", "Paris", C_STAR, C_STAR]   # CRI = 0.3+0.4 = 0.7
    # allied: model stays on c_star under allied hints (higher CRI)
    all_answers = [C_STAR, C_STAR, C_STAR, C_STAR]     # CRI = 1.0
    cri_adv = compute_cri(C_STAR, adv_answers)
    cri_all2 = compute_cri(C_STAR, all_answers)
    asym = compute_asymmetry(cri_adv, cri_all2)
    assert asym < 0, f"Expected asymmetry < 0 (adv < allied), got {asym}"
    # The docstring says: positive = resists correction.
    # Here adversarial < allied → asym < 0 → model IS correctable → no rationalization.
    # Confirm the sign convention by testing the flipped case:
    cri_adv_high = compute_cri(C_STAR, [C_STAR, C_STAR, C_STAR, C_STAR])  # 1.0
    cri_allied_low = compute_cri(C_STAR, ["Paris", "Paris", C_STAR, C_STAR])  # 0.7
    asym2 = compute_asymmetry(cri_adv_high, cri_allied_low)
    assert asym2 > 0, f"Expected asymmetry > 0 (adv > allied), got {asym2}"
    print(f"[PASS] asymmetry > 0 when adversarial CRI > allied CRI: {asym2:.1f}")
    print(f"       (adv CRI={cri_adv_high:.1f}, allied CRI={cri_allied_low:.1f})")

    # --- Show hint strings ---
    print()
    print("=" * 60)
    print("Hint strings for: subject='Eiffel Tower', original='Paris', new='Rome'")
    print("=" * 60)
    hints = build_hints("Eiffel Tower", "Paris", "Rome")
    for hint_type in ("adversarial", "allied"):
        print(f"\n[{hint_type.upper()}]")
        for i, h in enumerate(hints[hint_type], start=1):
            print(f"  L{i}: {h}")

    print()
    print("All self-tests passed.")
    sys.exit(0)
