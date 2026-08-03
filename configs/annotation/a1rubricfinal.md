# a1 Rubric Final

This rubric is used for a1 construct validation.

Its purpose is not to judge superficial quality (e.g., length, formatting, comment count, raw coverage, or apparent complexity),  
but to assess whether test logical quality can be reasonably approximated by a set of human-interpretable dimensions.

---

## 1. Core scoring rule

There are 5 main dimensions.

Each main dimension is scored as:

- 0 = missing / clearly weak
- 1 = partially present but insufficient
- 2 = clearly present and effective

### Important exception
For Exception-Path Handling, annotators may mark:

- N/A = this task does not reasonably call for exception / illegal-input testing

### Important note on N/A
Use `N/A` sparingly.  
Only assign `N/A` when exception / illegal-input testing is genuinely not relevant to the task, rather than simply absent from the candidate test.

---

## 2. Main dimensions

### 2.1 Assertion Effectiveness
Focus on whether the test truly verifies behavior rather than merely executing code.

- 0: Assertions are missing, trivial, or do not verify core behavior.
- 1: Some useful assertions exist, but they are partial, shallow, repetitive, or weak.
- 2: Assertions clearly verify expected behavior and meaningfully constrain correctness.

#### Notes
- Bare function calls without real checking should be treated as weak evidence.
- Assertion count alone does not imply high quality.

---

### 2.2 Boundary / Special-Input Checking
Focus on whether the test covers edge cases, extreme values, or other special inputs.

- 0: No meaningful boundary or special-input cases are tested.
- 1: Some edge/special cases are touched, but coverage is partial or weak.
- 2: Boundary or special-input cases are explicitly and effectively tested.

#### Notes
- This includes empty inputs, minimum/maximum values, degenerate structures, repeated values, or other task-relevant edge conditions.
- Do not reward random unusual inputs unless they have a clear testing purpose.

---

### 2.3 Exception-Path Handling
Focus on whether invalid-input or exception-related behavior is meaningfully tested when relevant.

- N/A: The task does not reasonably require exception / illegal-input testing.
- 0: Exception-related behavior is relevant, but not checked.
- 1: Exception behavior is partially checked or weakly expressed.
- 2: Exception paths are explicitly and effectively validated.

#### Notes
- This is not limited to `try/except`; `pytest.raises`, explicit invalid-input checks, and behavior-level exception validation all count.
- Only score this dimension when exception / illegal-input behavior is reasonably relevant to the task.

---

### 2.4 Behavior / Branch-Distinguishing Ability
Focus on whether the test distinguishes different logical behaviors, paths, or outcomes.

- 0: Tests do not meaningfully distinguish different behaviors or logical paths.
- 1: Some behavior / branch distinction exists, but it is incomplete or weak.
- 2: Tests clearly separate important logical paths, behaviors, or outcome modes.

#### Notes
- This is a logic-level judgment, not a raw structural branch-coverage number.
- A test may score well here even without explicit structural branch coverage if it clearly distinguishes meaningful behavioral alternatives.

---

### 2.5 Fault-Revealing Potential
Focus on whether the test is likely to expose realistic logic bugs.

- 0: The test is unlikely to reveal real defects.
- 1: The test may reveal some defects, but diagnostic value is limited.
- 2: The test has clear potential to expose realistic logic defects.

#### Notes
- This dimension asks whether the test would plausibly catch non-trivial mistakes, not whether it simply “looks complete.”
- Tests that only confirm the happy path usually score low here.

---

## 3. Auxiliary fields (NOT included in the main a1 score)

### 3.1 Teaching Value (auxiliary only)
This field is recorded for later analysis, but must not be included in the a1 main construct score.

- 0: Not worth learning from as a positive example.
- 1: Some teaching value, but limited.
- 2: Clearly useful as a teachable positive example.

#### Why auxiliary only
a1 validates the construct of test logical quality.  
“Teaching value” is related, but conceptually closer to later positive-sample analysis (e.g., a3), so it is not part of the main a1 score.

---

### 3.2 Primary Weakness Tags (optional but recommended)
Annotators may choose 1–2 tags that best describe the main weakness.

Suggested tags:
- `weak_assertion`
- `missing_boundary`
- `missing_exception`
- `weak_behavior_distinction`
- `low_fault_reveal`
- `surface_only_test`

These tags are optional but useful for later comparison with failure taxonomy.

---

## 4. Official scoring rule

Let the 5 main dimensions be:

1. Assertion Effectiveness
2. Boundary / Special-Input Checking
3. Exception-Path Handling
4. Behavior / Branch-Distinguishing Ability
5. Fault-Revealing Potential

### 4.1 Main-dimension total
For each sample, sum the scores of all applicable main dimensions.

### 4.2 Official a1 score
The official a1 score is normalized to 0–10 as:

\[
a1\_score = 10 \times \frac{\sum \text{main-dimension scores}}{2 \times \text{number of applicable dimensions}}
\]

### 4.3 Important rules
- If Exception-Path Handling = N/A, it is excluded from the denominator.
- Teaching Value must not be included in the official a1 score.
- If an annotation form includes an `overall_total_0_10` field, that field is supportive / auxiliary only and does not override the official a1 score derived from the main dimensions.

---

## 5. What should NOT drive the main score

The following may be recorded elsewhere, but should not determine the main a1 score:

- raw test length
- comment count
- formatting neatness
- surface complexity
- a single coverage number
- whether the test “looks long” or “looks sophisticated”

These belong more naturally to surface-control analysis (e.g., a5), not to the core construct definition.

---

## 6. Annotation consistency reminders

When uncertain, prefer the following order of judgment:

1. Does the test actually verify meaningful behavior?
2. Does it cover important edge / special cases?
3. Does it check exception behavior when relevant?
4. Does it distinguish important behaviors or logical paths?
5. Would it plausibly reveal a real logic mistake?

Do not reward a test merely because it is longer, more decorated, or more syntactically elaborate.

---

## 7. Deliverables linked to this rubric

This rubric is designed to support:
- dual independent annotation
- disagreement adjudication
- inter-rater consistency statistics
- correlation with automatic metrics
- group-level construct validation analysis