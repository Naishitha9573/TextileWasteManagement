# API DOCUMENTATION — AI Textile Waste Intelligence Platform

## Base URL

`http://localhost:8000`

---

## Authentication Endpoints

### 1. Register User
`POST /api/auth/register`
- **Request Body:** `{ "username": "...", "email": "...", "password": "...", "role": "..." }`
- **Response:** User object with role assignment.

### 2. Login & Token Acquisition
`POST /api/auth/token`
- **Request Body:** Form data (`username`, `password`)
- **Response:** `{ "access_token": "...", "token_type": "bearer" }`

---

## Analysis & AI Endpoints

### 3. Standalone Unified Image Analysis
`POST /api/analyze`
- **Headers:** `Authorization: Bearer <token>`
- **Form Data:**
  - `file`: Image file (JPG/PNG/WEBP, max 10MB)
  - `fabric_type_hint` (optional): `Cotton` | `Polyester` | `Wool` | `Denim` | etc.
  - `condition` (optional): `Excellent` | `Good` | `Fair` | `Poor` | `Contaminated`
  - `quantity_kg` (optional): float
- **Response:** Unified response containing explicit source provenance:
  ```json
  {
    "status": "success",
    "material_prediction": {
      "material": "Cotton",
      "confidence": 64.0,
      "source": "MODEL"
    },
    "waste_classification": {
      "waste_category": "Reusable",
      "source": "RULE_ENGINE"
    },
    "circularity_score": {
      "circularity_score": 82.5,
      "sustainability_rating": "Good",
      "source": "RULE_ENGINE"
    },
    "recommendation": {
      "strategy": "Fabric Reuse & Donation",
      "source": "RULE_ENGINE"
    },
    "environmental_impact": {
      "co2_savings_kg": 6.0,
      "water_savings_liters": 6750.0,
      "disclaimer": "All environmental impact values are mathematical estimates based on lifecycle emission factors.",
      "source": "ESTIMATED"
    }
  }
  ```

### 4. Batch Analysis
`POST /api/batches/{id}/analyze`
- **Headers:** `Authorization: Bearer <token>`
- **File:** Optional image upload
- **Behavior:** Persists analysis results and waste category to database.

---

## Model Metrics & Training Endpoints

### 5. Model Metrics
`GET /api/model-metrics`
- Returns measured accuracy (53.33%), precision, recall, F1, and confusion matrix.

### 6. Training Pipeline Status
`POST /api/train`
- Initiates model training and saves updated `.keras` model artifact.

---

## Reports & Exports

- `GET /api/reports/pdf` — PDF report with prediction source disclaimers
- `GET /api/reports/excel` — XLSX export
- `GET /api/reports/csv` — CSV dump
