# Sustainability Data Audit

Audit scope: read-only review of the current frontend, FastAPI backend, database schema, seed path, calculators, and tests. No sustainability implementation, database record, dataset, split, or training process was changed.

## CO2 Audit

**Frontend source:**

- `Frontend/src/pages/SustainabilityDashboard.jsx`: displays `analytics.co2_saved_kg` from `GET /api/analytics/sustainability`.
- `Frontend/src/pages/ManufacturerDashboard.jsx`: displays `analytics.co2_savings_kg` from `GET /api/analytics/manufacturer`.
- `Frontend/src/pages/RecyclerDashboard.jsx`: displays `selectedBatch.analysis.co2_savings` from the batch API response.
- The React components do not assign a numeric CO2 result or calculate a CO2 value.

**API:**

- `GET /api/analytics/sustainability` sums `AnalysisResult.co2_savings` across database rows.
- `GET /api/analytics/manufacturer` sums `AnalysisResult.co2_savings` for batches owned by the current manufacturer.
- Batch analysis endpoints return the persisted `AnalysisResult.co2_savings` field.
- `POST /api/predict/material` returns `SustainabilityService`'s `environmental_impact.co2_savings` for a user-supplied material and quantity.

**Backend service and calculation:**

The normal batch analysis path calls `algorithms.calculate_environmental_impact(material_for_analysis, batch.quantity, w_category)` and saves the result. Its formula is:

```text
CO2 savings = quantity_kg * material-specific CO2 factor
```

The active factor table in `Backend/algorithms.py` is:

```text
Cotton 2.2, Polyester 1.9, Wool 3.7, Silk 4.1,
Linen 2.4, Denim 2.6, Nylon 2.0, Rayon 1.6,
Acrylic 1.4, Mixed Fabrics 1.5
```

For `Hazardous Textile Waste`, the function returns `0.0`. Unknown materials use the fallback factor `1.5`. Results are rounded to two decimal places.

There is a second calculator in `Backend/app/services/sustainability_service.py` using `CARBON_FACTORS` from `Backend/app/core/sustainability_constants.py`:

```text
CO2 savings = quantity * CARBON_FACTORS[material]
```

Its factors differ, for example Cotton is `2.4` rather than `2.2`, and its unknown fallback is `3.0`. The service is called during batch analysis, but the batch route separately calls `algorithms.calculate_environmental_impact` when persisting `AnalysisResult.co2_savings`. This is an implementation inconsistency, not a frontend hardcoded result.

**Database:**

`Backend/database.py` defines `AnalysisResult.co2_savings` as a persisted Float field (`kg CO2`). Batch analysis writes the calculated value before committing. Analytics endpoints read database sums. The database therefore stores final calculated results, not only factors.

**Reference factor classification:** `REFERENCE_FACTOR`.

The factors are constants in code, but no scientific citation or external source is established in the current project. They must be treated as project-defined assumptions rather than verified lifecycle-assessment data.

**Final result classification:** `REAL_DATABASE_VALUE` for dashboard and batch-detail values, because those values are calculated from quantity/material inputs and then persisted. The underlying calculation is a project-defined estimate, not a measured emission result.

## WATER Audit

**Frontend source:**

- `Frontend/src/pages/SustainabilityDashboard.jsx`: displays `analytics.water_saved_liters` from `GET /api/analytics/sustainability`.
- `Frontend/src/pages/RecyclerDashboard.jsx`: displays `selectedBatch.analysis.water_savings` from the batch API response.
- No frontend component assigns a numeric water result or calculates water usage.

**API:**

- `GET /api/analytics/sustainability` sums `AnalysisResult.water_savings` across database rows.
- Batch analysis endpoints return the persisted `AnalysisResult.water_savings` field.
- `POST /api/predict/material` returns `SustainabilityService`'s `environmental_impact.water_savings`.

**Backend service and calculation:**

The normal batch analysis path uses `algorithms.calculate_environmental_impact`:

```text
Water savings = quantity_kg * material-specific water factor
```

The active factors in `Backend/algorithms.py` are:

```text
Cotton 2500.0, Polyester 350.0, Wool 1600.0, Silk 2100.0,
Linen 1800.0, Denim 2900.0, Nylon 400.0, Rayon 600.0,
Acrylic 300.0, Mixed Fabrics 1000.0
```

Unknown materials use fallback factor `500`. Hazardous textile waste returns `0.0`. Results are rounded to two decimal places.

The second calculator in `SustainabilityService` uses `WATER_FACTORS` from `sustainability_constants.py`:

```text
Water savings = quantity * WATER_FACTORS[material]
```

Its factors differ substantially, for example Cotton is `2700` rather than `2500`, Wool is `10000` rather than `1600`, and its unknown fallback is `1500`. This duplicate factor system means different API paths can produce different estimates for the same inputs.

**Database:**

`Backend/database.py` defines `AnalysisResult.water_savings` as a persisted Float field (liters). Batch analysis calculates and saves it. Analytics endpoints aggregate the stored values.

**Reference factor classification:** `REFERENCE_FACTOR`.

No scientific source is documented for either water-factor table. They are project-defined assumptions pending source verification.

**Final result classification:** `REAL_DATABASE_VALUE` for dashboard and batch-detail values; `REAL_DYNAMIC_CALCULATION` for the direct material-analysis response. These are calculated estimates, not sensor measurements or verified water-accounting results.

## Sustainability Score Audit

**Frontend source:**

- `Frontend/src/pages/RecyclerDashboard.jsx` displays `selectedBatch.analysis.overall_circularity_score`.
- `Frontend/src/pages/ManufacturerDashboard.jsx` displays `batch.analysis.overall_circularity_score` and `analytics.average_circularity`.
- `Frontend/src/pages/SustainabilityDashboard.jsx` displays `analytics.circularity_avg`.

**API/database path:**

Batch analysis calculates scores and persists `AnalysisResult.sustainability_score`, `material_recovery_score`, and `overall_circularity_score`. Analytics averages `AnalysisResult.overall_circularity_score`. The score is not calculated from the CO2 or water output.

**Formula:**

`Backend/app/services/scoring_service.py` defines the canonical weighted formula, implemented through `algorithms.calculate_scores`:

```text
circularity score =
  material recyclability * 0.35
+ material condition * 0.20
+ reuse potential * 0.20
+ environmental benefit * 0.15
+ processing feasibility * 0.10
```

The environmental-benefit input is a hardcoded material score map in `Backend/algorithms.py` (for example Cotton `95`, Polyester `80`, Wool `98`), with a hazardous-category override to `0`. It is not derived from CO2 or water factors.

**Classification:** `REAL_DYNAMIC_CALCULATION` for the weighted rule calculation, with `REFERENCE_FACTOR`/hardcoded rule tables as inputs. It is not a measured sustainability score and is not an environmental-impact calculation.

## Hardcoded Values Found

| File | Value | Purpose | Classification |
|---|---:|---|---|
| `Backend/app/core/sustainability_constants.py` | `CARBON_FACTORS`, including Cotton `2.4` | Reference factors used by `SustainabilityService` | `REFERENCE_FACTOR`; source not established |
| `Backend/app/core/sustainability_constants.py` | `WATER_FACTORS`, including Cotton `2700` | Reference factors used by `SustainabilityService` | `REFERENCE_FACTOR`; source not established |
| `Backend/algorithms.py` | `co2_factors`, including Cotton `2.2` | Reference factors used by normal batch persistence | `REFERENCE_FACTOR`; source not established |
| `Backend/algorithms.py` | `water_factors`, including Cotton `2500` | Reference factors used by normal batch persistence | `REFERENCE_FACTOR`; source not established |
| `Backend/algorithms.py` | Unknown CO2 fallback `1.5`; unknown water fallback `500` | Fallback assumptions | `REFERENCE_FACTOR`/assumption |
| `Backend/algorithms.py` | Material environmental scores such as Cotton `95`, Wool `98` | Circularity score input | `HARDCODED_RESULT` as a rule input, not a final displayed result |
| `Backend/main.py` seed path | Sample quantities/materials and generated analysis records | Initial demo database content | `DEMO_DATA` when seeded into an empty database |
| `Backend/main.py` | Milestone thresholds `500`, `100000`, `1000` | Achievement flags in sustainability analytics | Hardcoded business thresholds, not CO2/water results |
| `Frontend/src/pages/SustainabilityDashboard.jsx` | Tree conversion divisor `22` | Presentation-only CO2-to-tree estimate | Hardcoded presentation assumption |

No hardcoded final CO2 or water result was found in the active React sustainability components. No assignments equivalent to `co2 = 12.5`, `water = 2500`, `setCo2(...)`, or `setWater(...)` were found there.

## Real Calculations Found

| File/function | Formula | Inputs | Output |
|---|---|---|---|
| `Backend/algorithms.py:calculate_environmental_impact` | `quantity_kg * co2_factor`; `quantity_kg * water_factor` | material, quantity, waste category | CO2, water, landfill estimates |
| `Backend/app/services/sustainability_service.py:_estimate_environmental_impact` | `quantity * CARBON_FACTORS[material]`; `quantity * WATER_FACTORS[material]` | material, quantity, waste category | CO2, water, energy, landfill, recovery estimates |
| `Backend/main.py:get_sustainability_analytics` | SQL `SUM(AnalysisResult.co2_savings)` and `SUM(AnalysisResult.water_savings)` | persisted analysis rows | dashboard totals |
| `Backend/main.py:get_manufacturer_analytics` | SQL `SUM(AnalysisResult.co2_savings)` filtered by owner | persisted analysis rows | manufacturer CO2 total |
| `Backend/app/services/scoring_service.py:calculate_circularity_score` | weighted 35/20/20/15/10 sum | rule-table scores and condition/category flags | circularity/sustainability score |

## Demo Data Found

`Backend/main.py` seeds six sample waste batches when the database has no users. Five are marked `Analyzed`, and the seed path calls `algorithms.calculate_environmental_impact` to create their CO2/water values. These values are calculated from seeded sample inputs, but the records themselves are demo/sample data. Existing database query results during this audit contained `29` batches and `16` analysis rows, with aggregate stored values of `1364.1` kg CO2 and `899050.0` L water. The presence of stored values does not by itself prove the records are production measurements.

Tests also use controlled expected calculations, such as `100 * 2.2 = 220.0` kg CO2 and `100 * 2500 = 250000.0` L water. These are test assertions, not production display values.

No static sustainability JSON or frontend hardcoded sustainability chart array was found in the inspected active paths. The Sustainability Dashboard contains KPI cards and PDF export, both fed by the analytics API; no independent chart dataset was found.

## Endpoint and UI Trace Summary

| UI | Field | Endpoint/data source | Backend/database source | Classification |
|---|---|---|---|---|
| Sustainability Dashboard | CO2 prevented | `GET /api/analytics/sustainability` | SQL sum of `AnalysisResult.co2_savings` | `REAL_DATABASE_VALUE` |
| Sustainability Dashboard | Water saved | `GET /api/analytics/sustainability` | SQL sum of `AnalysisResult.water_savings` | `REAL_DATABASE_VALUE` |
| Manufacturer Dashboard | CO2 savings | `GET /api/analytics/manufacturer` | SQL sum of `AnalysisResult.co2_savings` by user | `REAL_DATABASE_VALUE` |
| Recycler Batch Details | CO2 savings | batch details/analyze response | `AnalysisResult.co2_savings` written by calculator | `REAL_DATABASE_VALUE` |
| Recycler Batch Details | Water saved | batch details/analyze response | `AnalysisResult.water_savings` written by calculator | `REAL_DATABASE_VALUE` |
| Recycler/Manufacturer dashboards | Circularity score | batch/analytics response | weighted rule score and database field/average | `REAL_DYNAMIC_CALCULATION` |
| Manual material prediction | CO2/water impact | `POST /api/predict/material` | `SustainabilityService` calculation | `REAL_DYNAMIC_CALCULATION` |

## Dynamic Behavior Assessment

The code is quantity-sensitive: changing `quantity_kg` changes CO2, water, landfill, energy, and recovery values proportionally, except hazardous waste, which returns zero. Material-sensitive factor maps select different values for different materials. The database persistence path can be tested through isolated function inputs without changing production records; existing tests already assert quantity/material-specific CO2 and water results.

However, the two factor tables mean material-sensitive results depend on which endpoint/path is used. The implementation does not establish that either factor set is scientifically sourced, and it does not measure actual emissions or water consumption.

## Final Verdict

**CO2: MIXED**

CO2 values shown on the dashboards and batch details are calculated and persisted, then aggregated from the database. They are not frontend hardcoded final numbers. However, the factors are hardcoded project assumptions with no established source, there are two inconsistent factor tables, and seeded records are demo data. Therefore the correct audit classification is mixed: real calculation/storage mechanics, estimated assumptions and possible demo provenance.

**WATER: MIXED**

Water values are calculated from quantity and material factors and persisted/aggregated. They are not frontend hardcoded final numbers. The factor tables are undocumented project assumptions, inconsistent across services, and current records may include seeded demo data. Therefore the correct classification is mixed.

**SUSTAINABILITY SCORE: MIXED**

The score is dynamically calculated by a weighted rule engine and stored/aggregated, but its inputs are hardcoded material/condition/category scores. It is not derived from CO2 or water and should be described as a rule-based circularity estimate. Demo records can also contribute to dashboard averages.

## Training Safety Confirmation

- EfficientNet training interrupted: **NO**
- Training restarted: **NO**
- Dataset modified: **NO**
- Train split modified: **NO**
- Validation split modified: **NO**
- Test split modified: **NO**
- ML training configuration modified: **NO**
- Sustainability implementation modified: **NO**
- Database records modified: **NO**

Audit artifact created: `reports/sustainability_data_audit.md`.
