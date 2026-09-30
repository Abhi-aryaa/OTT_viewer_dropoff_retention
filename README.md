# OTT Viewer Drop-off Prediction — ML Pipeline & Data Quality Audit

## Project Overview

This project builds a complete machine learning pipeline on a synthetic OTT
viewer dataset (33,171 rows, 22 columns) to predict whether a viewer will
drop off after watching an episode. The project goes beyond model building —
it critically audits the dataset to detect embedded synthetic generation rules
and identifies leakage columns that cause artificially inflated metrics.

**Business Problem:**
A viewer finishes watching an episode on an OTT platform. Their
session signals are recorded (watch percentage, pauses, rewinds, etc.). The
question is: based on this session's behaviour, will this viewer return to
watch the next episode?

The model prediction fires after the episode session ends, before the next
episode autoplays - enabling the platform to trigger retention actions
(recommendation changes, autoplay tweaks) for at-risk viewers.

---

## Dataset

- **Source:** OTT Viewer Drop-off Retention Dataset (US, synthetic)
- **Rows:** 33,171 viewer-episode interactions
- **Columns:** 22 (15 after cleaning)
- **Target:** `drop_off` (0 = stayed, 1 = dropped off)
- **Class distribution:** 85.5% stayed (28,367) | 14.5% dropped off (4,804)
- **Imbalance ratio:** 5.9 : 1

### Feature Categories

| Category | Columns |
|---|---|
| Identifiers (dropped) | show_id, title |
| Content metadata | platform, genre, release_year, episode_number, episode_duration_min |
| Content quality | pacing_score, hook_strength, dialogue_density, visual_intensity, cognitive_load |
| Viewer behaviour | avg_watch_percentage, pause_count, rewind_count, skip_intro |
| Leakage (dropped) | drop_off_probability, retention_risk, attention_required, night_watch_safe, season_number |

---


## Step-by-Step Pipeline

### Step 1 — Data Loading & Basic Inspection
- Loaded CSV (33,171 rows × 22 columns)
- Checked dtypes, shape, null values (0 missing), duplicates (0)
- Found `season_number` has only 1 unique value across all rows — flagged for removal

### Step 2 — Exploratory Data Analysis (EDA)

**Numerical features vs drop_off (boxplot + describe):**

Key findings from groupby analysis:

| Feature | Stayed mean | Dropped mean | Signal |
|---|---|---|---|
| avg_watch_percentage | 60.1 | 37.1 | Strong — 23pt gap |
| cognitive_load | 5.9 | 8.0 | Strong — clear separation |
| pacing_score | 5.6 | 4.2 | Strong |
| hook_strength | 5.7 | 4.2 | Strong |
| visual_intensity | 6.0 | 6.0 | Near zero — weak predictor |
| rewind_count | 2.0 | 2.0 | Near zero — weak predictor |

**Categorical features vs drop_off (crosstab % distribution):**

| Feature | Key Finding |
|---|---|
| attention_required | medium → 0.0% drop-off (16,293 rows, zero exceptions) |
| night_watch_safe | value=1 → 0.0% drop-off (527 rows, zero exceptions) |
| retention_risk | high → 100% drop-off, low/medium → 0% — perfect leakage |
| dialogue_density | high → 30.4% drop-off vs low → 6.3% |
| pause_count | monotonic: 0 pauses → 2.5%, 10 pauses → 77.1% |
| skip_intro | skipped → 23.3% drop-off vs not skipped → 5.9% |
| Contract | Drama → 24.2%, Comedy → 2.9% |

### Step 3 — Leakage Detection & Column Removal

**Confirmed leakage columns — dropped before modelling:**

```
retention_risk       → high maps to 100% drop_off, low/medium to 0%
                       This IS the target encoded as a string category.

drop_off_probability → drop_off=0 max: 0.599, drop_off=1 min: 0.600
                       Perfect non-overlapping split. Target as float.

attention_required   → medium → 0.0% drop-off across 16,293 rows
                       Zero exceptions — derived from label during generation.

night_watch_safe     → value=1 → 0.0% drop-off across 527 rows
                       Zero exceptions — probable label-derived column.

season_number        → Only unique value = 1 across all 33,171 rows
                       Zero variance — zero information.

show_id, title       → Identifiers — no predictive signal.
```

### Step 4 — Feature Engineering

Two new features engineered from domain knowledge:

**content_complexity = cognitive_load × dialogue_density_num**
- dialogue_density mapped: low→1, medium→2, high→3
- Captures how mentally demanding the content is overall
- Stayed mean: 11.7 | Dropped mean: 21.0 (large separation)

**engagement_score = hook_strength + pacing_score**
- Captures how compelling and well-paced the content is
- Stayed mean: 11.3 | Dropped mean: 8.4 (clear separation)

Both features visualised via boxplot and KDE histogram split by drop_off class.
`dialogue_density_num` helper column dropped after engineering.

### Step 5 — Train / Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Train: 26,536 rows | Test: 6,635 rows
# stratify=y preserves 14.5% drop-off rate in both splits
```

### Step 6 — Preprocessing Pipeline

```
Numerical (13 cols): StandardScaler
Categorical (3 cols): SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown=ignore)
```

Final feature set after cleaning and engineering:
- **Categorical:** platform, genre, dialogue_density
- **Numerical:** release_year, episode_number, episode_duration_min, pacing_score,
  hook_strength, visual_intensity, avg_watch_percentage, pause_count,
  rewind_count, skip_intro, cognitive_load, engagement_score, content_complexity

### Step 7 — Model Training & Results

Three models trained and compared:

| Model | Accuracy | ROC-AUC | Recall (drop-off) | F1 (drop-off) |
|---|---|---|---|---|
| Logistic Regression | 99.76% | 0.9999 | 0.99 | 0.99 |
| Random Forest | 98.91% | 0.9990 | 0.96 | 0.96 |
| XGBoost | 99.80% | 0.9999 | 0.99 | 0.99 |

### Step 8 — Feature Importance (Random Forest)

Top features by RF importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | avg_watch_percentage | 32.2% |
| 2 | cognitive_load | 10.7% |
| 3 | engagement_score (engineered) | 10.4% |
| 4 | content_complexity (engineered) | 9.5% |
| 5 | pause_count | 7.1% |
| 6 | pacing_score | 6.2% |
| 7 | hook_strength | 6.1% |

LR coefficients confirmed same ranking:
avg_watch_percentage (−15.19) > cognitive_load (+9.02) > pause_count (+5.00)

### Step 9 — Data Quality Audit

**Finding 1 — cognitive_load is a near-perfect decision rule:**
```
cognitive_load = 4 → 0.0% drop-off
cognitive_load = 5 → 0.0% drop-off
cognitive_load = 6 → 0.0% drop-off
cognitive_load = 7 → 17.0% drop-off
cognitive_load = 9 → 79.6% drop-off
(Note: value 8 never appears in 33,171 rows)
```

**Finding 2 — avg_watch_percentage ranges do not overlap:**
```
Stayed  range: 34 – 100%
Dropped range: 13 – 55%
Overlap window: 34–55% only
Above 55%: nobody dropped off. Below 34%: nobody stayed.
```

**Finding 3 — retention_risk perfectly encodes the target:**
```
retention_risk = high   → drop_off = 1 (4,804 rows, 100%)
retention_risk = low    → drop_off = 0 (965 rows, 100%)
retention_risk = medium → drop_off = 0 (27,402 rows, 100%)
```

**Conclusion — Data was generated using hard if-else rules:**
```python
IF cognitive_load IN [4, 5, 6]   → drop_off = 0  (always)
IF cognitive_load == 9            → drop_off = 1  (80% of time)
IF avg_watch_percentage > 55      → drop_off = 0  (always)
IF avg_watch_percentage < 34      → drop_off = 1  (always)
```

The model learned these exact generation rules — not real viewer behaviour.
ROC-AUC of 0.9999 reflects synthetic rule-learning, not genuine generalisation.
**Real OTT churn models typically achieve ROC-AUC of 0.75–0.85.**

---

## Key Takeaways

| | OTT Dataset (synthetic) | Real OTT Data (expected) |
|---|---|---|
| ROC-AUC | 0.9999 | 0.75 – 0.85 |
| Class boundaries | Zero overlap | Heavily overlapping |
| Uncertain predictions | < 1% | 20 – 35% |
| Leakage columns | 4 confirmed | None (if built correctly) |
| Feature distributions | Near-perfect rules | Noisy, gradual |

This project demonstrates the complete ML pipeline methodology and the critical
analytical skill of auditing synthetic datasets — identifying when model metrics
reflect data generation artefacts rather than genuine predictive performance.

---

## Tools & Technologies

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Anaconda

---


# Launch notebook
jupyter notebook ott_churn_syntheticds.ipynb
```
