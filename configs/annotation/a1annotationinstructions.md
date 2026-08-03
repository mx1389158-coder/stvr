# a1 Annotation Instructions

This document explains how annotators should use the a1 rubric.

---

## 1. Annotation goal

Your task is to assess the logical quality of the test.

You are not judging:
- code style
- formatting beauty
- length
- comment quantity
- raw coverage numbers
- whether the test merely “looks complex”

Focus on whether the test meaningfully checks behavior, distinguishes important cases, and has potential to reveal defects.

---

## 2. What annotators are allowed to see

Use only:
- the provided task description / prompt
- the candidate test code

Do not use:
- source labels
- difficulty labels
- automatic metric values
- internal group assignments
- any information about whether a sample is intended to be positive / negative

---

## 3. Scoring procedure

For each sample, score the following 5 main dimensions:

1. Assertion Effectiveness
2. Boundary / Special-Input Checking
3. Exception-Path Handling
4. Behavior / Branch-Distinguishing Ability
5. Fault-Revealing Potential

### Score values
- 0 = missing / clearly weak
- 1 = partially present but insufficient
- 2 = clearly present and effective

### Exception-path special rule
For Exception-Path Handling, you may use:
- N/A if the task does not reasonably call for exception / illegal-input testing

### Important note on N/A
Use `N/A` only when exception / illegal-input testing is genuinely not relevant to the task.  
Do not use `N/A` merely because the candidate test omitted exception-related checks.

---

## 4. Auxiliary fields

Please also fill:

### 4.1 Teaching Value (auxiliary only)
- 0 = not worth learning from
- 1 = some teaching value
- 2 = clearly teachable

This field is recorded for later analysis, but it is not part of the official a1 main score.

### 4.2 Primary Weakness Tags (optional but recommended)
Choose 1–2 tags if helpful:
- `weak_assertion`
- `missing_boundary`
- `missing_exception`
- `weak_behavior_distinction`
- `low_fault_reveal`
- `surface_only_test`

### 4.3 Rater note
Add a short note when:
- the sample is clearly very strong or very weak
- the case is borderline
- you use `N/A` for exception-path handling

---

## 5. Overall total score field

If the annotation form contains an `overall_total_0_10` field:

- you may fill it as your holistic overall judgment on a 0–10 scale
- but the official a1 main score will be computed later from the 5 main dimensions using the rubric rule
- therefore, your main responsibility is to score the 5 dimensions carefully and consistently

In short:
- dimension scores are primary
- `overall_total_0_10` is optional and supportive / auxiliary only

---

## 6. Independence requirement

- Annotate independently.
- Do not discuss scores before both first-pass annotations are submitted.
- Do not try to “match” another annotator’s style.
- If uncertain, follow the rubric literally and leave a short note.

---

## 7. How to think when uncertain

When unsure, use this order:

1. Does the test actually verify something meaningful?
2. Does it cover important edge / special cases?
3. Does it check exception behavior when relevant?
4. Does it distinguish important behaviors or logical paths?
5. Would it plausibly catch a real logic mistake?

If the answer is mostly “no”, the score should not be high even if the code appears long or elaborate.

---

## 8. Treatment of surface features

The following should not directly raise the score:

- more lines of code
- more comments
- more imports
- more complex syntax
- more nested structure
- more assertions that are repetitive or shallow

Only reward these when they clearly improve test logic quality.

---

## 9. Adjudication reminder

Disagreements are expected.

The first-pass goal is:
- independent judgment
- consistent use of rubric
- clear notes for ambiguous cases

Large disagreements will later be resolved through adjudication, not by changing first-pass scores in advance.

---

## 10. Summary

Please prioritize:
- behavioral verification
- boundary relevance
- exception relevance
- behavior / path distinction
- fault-revealing potential

Do not let superficial polish substitute for real testing logic.