# Implementation Plan — Textile Waste Intelligence Platform
## Architecture Diagram Mapping

This plan maps **every component** shown in the architecture diagram to real code.

---

## Architecture Overview

```
Access Channels → API Gateway (FastAPI) → Microservices Layer → AI/ML Layer → Data Layer
```

---

## Proposed Project Structure

```
Textile/
├── Backend/
│   ├── gateway/             ← API Gateway (FastAPI main entrypoint)
│   │   ├── main.py          ← Routes, CORS, Rate Limiting, Auth middleware
│   │   └── middleware.py    ← Rate limiter, Logger, Request validator
│   │
│   ├── services/            ← 12 Microservices
│   │   ├── user_service/    ← User & Access Management
│   │   ├── inventory_service/  ← Inventory & Waste Management
│   │   ├── ingestion_service/  ← Image & Data Ingestion
│   │   ├── image_analysis/     ← Textile Image Analysis
│   │   ├── material_classification/ ← Material Classification
│   │   ├── waste_classification/    ← Waste Classification
│   │   ├── recommendation_service/  ← Recycling & Reuse Recommendations
│   │   ├── sustainability_service/  ← Sustainability Intelligence
│   │   ├── environmental_service/   ← Environmental Impact
│   │   ├── scoring_service/         ← Scoring & Ranking
│   │   ├── report_service/          ← Report & Export
│   │   └── notification_service/    ← Notifications & Alerts
│   │
│   ├── ml/                  ← AI/ML & Analytics Layer
│   │   ├── fabric_classifier.py     ← CNN/EfficientNet fabric classification
│   │   ├── fiber_predictor.py       ← Multi-label fiber composition CNN
│   │   ├── recyclability_model.py   ← XGBoost/LightGBM prediction
│   │   ├── damage_detector.py       ← YOLOv8-style damage/contamination
│   │   ├── reuse_potential.py       ← Random Forest upcycling potential
│   │   ├── sustainability_model.py  ← Regression sustainability impact
│   │   └── recommendation_engine.py ← Heuristic + ML hybrid engine
│   │
│   ├── data/                ← Data Processing & Integration Layer
│   │   ├── connectors.py    ← Data connectors (APIs / uploads)
│   │   ├── data_lake.py     ← Raw data storage manager
│   │   ├── pipeline.py      ← ETL/Data Pipeline
│   │   ├── feature_store.py ← Feature extraction & encoding
│   │   └── image_pipeline.py ← Image preprocessing & augmentation
│   │
│   ├── database.py          ← PostgreSQL (SQLAlchemy ORM)
│   ├── mongodb_client.py    ← MongoDB collections setup
│   ├── redis_client.py      ← Redis caching layer
│   ├── schemas.py           ← Pydantic validation models
│   ├── auth.py              ← JWT auth & RBAC
│   └── requirements.txt
│
├── Frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── RecyclerDashboard.jsx        ← Recycling facility view
│   │   │   ├── SustainabilityDashboard.jsx  ← ESG analytics
│   │   │   ├── ManufacturerDashboard.jsx    ← Production waste view
│   │   │   └── AdminDashboard.jsx           ← User management & monitoring
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx            ← Drag-drop CV image uploader
│   │   │   ├── CircularityGauge.jsx         ← Animated score radial
│   │   │   ├── NotificationCenter.jsx       ← Alert drawer
│   │   │   ├── MaterialChart.jsx            ← Chart.js fabric distributions
│   │   │   ├── CarbonTracker.jsx            ← CO2/water savings chart
│   │   │   └── BatchTable.jsx               ← Sortable inventory table
│   │   ├── App.jsx
│   │   ├── index.css                        ← Tailwind base + glassmorphism
│   │   └── main.jsx
│   ├── tailwind.config.js
│   └── package.json
│
├── docker-compose.yml       ← Orchestrates: backend, frontend, postgres, mongodb, redis
└── README.md
```

---

## Proposed Changes (Detailed by Layer)

### Layer 1: API Gateway

#### [MODIFY] `Backend/gateway/main.py`
FastAPI application with:
- JWT Bearer token middleware injected on every request
- Rate-limiting: max 100 requests/minute per IP (using `slowapi`)
- Request ID injection for distributed logging
- CORS configured for `localhost:5173` in dev, `*` in Docker

---

### Layer 2: 12 Microservices

#### [MODIFY] `services/user_service/router.py`
- `POST /api/auth/register` — User creation with role assignment  
- `POST /api/auth/token` — JWT login  
- `POST /api/auth/oauth-mock` — Google/GitHub OAuth simulation  
- `GET /api/users` — Admin: list all users  
- `PUT /api/users/{id}/role` — Admin: change role  
- `DELETE /api/users/{id}` — Admin: remove user  

#### [MODIFY] `services/inventory_service/router.py`
- Full CRUD for waste batches  
- Collection management with batch grouping  
- Source tracking by facility, supplier, or manufacturer unit  
- Inventory dashboard metrics  

#### [NEW] `services/ingestion_service/router.py`
- `POST /api/ingest/image` — Upload image to local storage / S3  
- `POST /api/ingest/batch-upload` — Multi-file batch upload  
- Image validation (format, max size, corruption check)  
- Pre-processing pipeline trigger on upload  

#### [NEW] `services/image_analysis/router.py`
- `POST /api/analyze/image/{batch_id}` — Run CV pipeline  
- OpenCV: HSV color histograms, Sobel edge/texture analysis  
- Contour detection for damage/stains  
- Returns: color, texture, pattern, damage flag, contamination flag  

#### [NEW] `services/material_classification/router.py`
- `GET /api/classify/material/{batch_id}` — Run ML classification  
- Scikit-Learn Random Forest predicts fiber composition %  
- Returns: `{cotton: 85%, elastane: 15%, quality: GradeA}`  

#### [NEW] `services/waste_classification/router.py`
- `GET /api/classify/waste/{batch_id}` — Classify waste category  
- Ensemble model: condition + contamination + composition → category  
- Maps to: Recyclable / Reusable / Repairable / Upcyclable / Compostable / Hazardous  

#### [MODIFY] `services/recommendation_service/router.py`
- `GET /api/recommend/{batch_id}` — Optimal recycling strategy  
- Heuristic + ML hybrid lookup  
- Returns ordered list of: strategy, options, feasibility score  

#### [NEW] `services/sustainability_service/router.py`
- `GET /api/sustainability/analytics` — Platform-wide ESG summary  
- Carbon footprint, waste diversion, circular economy score  
- Benchmarking against industry averages  

#### [NEW] `services/environmental_service/router.py`
- `GET /api/environmental/impact/{batch_id}` — Per-batch impact  
- CO2 saved, water conserved, landfill reduction  
- Resource conservation score  

#### [MODIFY] `services/scoring_service/router.py`
- `GET /api/score/{batch_id}` — Full scoring report  
- Circularity Score (weighted formula)  
- Recyclability, Reuse, Sustainability, Material Recovery, Overall Scores  

#### [MODIFY] `services/report_service/router.py`
- `GET /api/reports/pdf` — Generate PDF (ReportLab)  
- `GET /api/reports/excel` — Generate XLSX (openpyxl)  
- Scheduled reports with date-range filters  

#### [NEW] `services/notification_service/router.py`
- Background alert dispatcher  
- Triggers: contamination detected, milestone reached, low inventory  
- In-app notifications + email webhook stubs  

---

### Layer 3: AI/ML Models

#### [NEW] `ml/fabric_classifier.py`
- Scikit-Learn `RandomForestClassifier` trained on synthetic textile features  
- Inputs: HSV hue peak, edge density, pixel variance  
- Outputs: fabric type (Cotton, Polyester, Wool, Silk, Denim, Mixed…)  

#### [NEW] `ml/recyclability_model.py`
- `GradientBoostingClassifier` (approximates XGBoost/LightGBM)  
- Inputs: fabric_type, condition, damage, contamination, composition  
- Outputs: recyclability category + confidence %  

#### [NEW] `ml/damage_detector.py`
- OpenCV contour-based damage region finder  
- Detects stain patches via color deviation from fabric mean  
- Simulates YOLOv8-style bounding box output with regions & confidence  

#### [NEW] `ml/recommendation_engine.py`
- Heuristic lookup table + Scikit-Learn `KNeighborsClassifier`  
- Finds closest training sample to current batch features  
- Returns: top-3 recycling strategies with feasibility scores  

---

### Layer 4: Data Processing

#### [NEW] `data/image_pipeline.py`
- OpenCV image preprocessing: resize, denoise, normalize  
- Feature extraction: dominant color (k-means), texture (LBP), edges (Canny)  
- Augmentation stubs for training data generation  

#### [NEW] `data/feature_store.py`
- Stores extracted feature vectors in MongoDB (`feature_store` collection)  
- Enables reuse for ML inference without re-processing images  

---

### Layer 5: Database Layer

#### [MODIFY] `Backend/database.py`
- Switch to PostgreSQL connection string  
- Keep SQLAlchemy ORM for `users`, `waste_batches`, `analysis_results`, `notifications`  

#### [NEW] `Backend/mongodb_client.py`
- MongoDB collections: `cv_analysis`, `feature_store`, `esg_telemetry`, `audit_logs`  

#### [NEW] `Backend/redis_client.py`
- Redis cache for dashboard analytics queries (TTL: 60s)  
- Session token blacklist for logout  

---

### Layer 6: Frontend Updates

#### [MODIFY] `src/index.css`
- Add Tailwind directives + keep glassmorphism custom classes  

#### [NEW] `src/components/MaterialChart.jsx`
- Chart.js doughnut showing fabric type distribution  

#### [NEW] `src/components/CarbonTracker.jsx`
- Chart.js line chart tracking cumulative CO2/water savings over time  

#### [MODIFY] All 4 Dashboard pages
- Tailwind CSS layout classes replacing inline styles  
- Chart.js embedded analytics panels  

---

### Layer 7: Docker Compose

#### [MODIFY] `docker-compose.yml`
Five services:
```yaml
services:
  postgres:     image: postgres:15
  mongodb:      image: mongo:7
  redis:        image: redis:7-alpine
  backend:      build: ./Backend  (depends on postgres, mongodb, redis)
  frontend:     build: ./Frontend (depends on backend)
```

---

## Verification Plan

### Automated Tests
```bash
# Backend unit + integration
python -m pytest Backend/tests/ -v

# API smoke test
curl http://localhost:8000/api/auth/token -d '{"username":"recycler","password":"recycler123"}'
```

### Manual Walkthrough
1. Login with each of the 4 roles
2. Upload fabric image → verify AI classification fires
3. View Circularity Score chart on Recycler dashboard
4. Export PDF/Excel from Sustainability dashboard
5. Admin: change a user's role

> [!IMPORTANT]
> The ML models use Scikit-Learn (no GPU required) and generate synthetic training data on startup — no external datasets needed. This makes it fully self-contained and offline-capable.
