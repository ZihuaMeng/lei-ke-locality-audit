# KE Locality Audit Report

**Generated:** 2026-03-03 14:16:15

**Mode:** `NO_EDIT`

**Timestamp (run):** 2026-03-03T14:16:14.460550

---

## Summary

- Smoke test: **PASS**
- Rates not computed in NO_EDIT mode.

---

## Prompt Counts

- Rewrite: 3
- Neighborhood: 3
- Distractor: 3

---

## Rewrite Results (Efficacy)

| # | Subject | Query | C_orig | C_star | Edit Success |
| --- | --- | --- | --- | --- | --- |
| 1 | Eiffel Tower | Which city is the Eiffel Tower located in? | [STUB_ANSWER for: Which city is the Eiffel Tower located i] | [NO_EDIT_MODE — edit skipped] | None |
| 2 | Marie Curie | Where was Marie Curie born? | [STUB_ANSWER for: Where was Marie Curie born?] | [NO_EDIT_MODE — edit skipped] | None |
| 3 | The speed of light | What is the approximate speed of light? | [STUB_ANSWER for: What is the approximate speed of light?] | [NO_EDIT_MODE — edit skipped] | None |

---

## Neighborhood Results (Locality)

| # | Query | Expected | Answer | Stable |
| --- | --- | --- | --- | --- |
| 1 | What country is known for the Louvre museum? | France | [STUB_ANSWER for: What country is known for the Louvre mus] | None |
| 2 | Who discovered polonium? | Marie Curie | [STUB_ANSWER for: Who discovered polonium?] | None |
| 3 | What unit is used to measure the speed of light? | meters per second | [STUB_ANSWER for: What unit is used to measure the speed o] | None |

---

## Distractor Results (Side Effects)

| # | Query | Expected | Answer | Unchanged |
| --- | --- | --- | --- | --- |
| 1 | Who wrote Hamlet? | William Shakespeare | [STUB_ANSWER for: Who wrote Hamlet?] | None |
| 2 | What is the capital of Japan? | Tokyo | [STUB_ANSWER for: What is the capital of Japan?] | None |
| 3 | What element has atomic number 79? | Gold | [STUB_ANSWER for: What element has atomic number 79?] | None |