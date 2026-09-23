# Milestone 4 Status

## Verified Scope

The platform uses React and FastAPI with PostgreSQL as the primary database. Inventory, batch analysis, sustainability analytics, material inference, and report exports use backend/API data. Structural damage and chemical contamination are excluded from the active application.

Material inference uses the existing EfficientNet-B0 checkpoint and nine-class mapping. The model is loaded once by the backend service; the checkpoint is mounted read-only in the production Compose configuration.

## Data Flow

`React -> FastAPI -> PostgreSQL / EfficientNet-B0 / configured factor service -> JSON -> React`

CO2 and water are estimates from actual quantity and material-specific configured factors. They are not presented as laboratory measurements. Circularity and recovery recommendations remain backend rule-engine outputs.

DeepFashion assets and caption-derived material preparation utilities remain available in the repository. No separate validated garment inference model is connected to the active upload response, so a garment result must be treated as unavailable rather than fabricated.

## Deployment

`docker-compose.yml` defines frontend, backend, PostgreSQL, and MongoDB services. PostgreSQL credentials are supplied through `.env`; `.env` is ignored. The backend health endpoint is `/api/health`. No cloud deployment or URL has been claimed.

## Validation

Run from the repository root:

```powershell
Set-Location Backend; py -3 -m pytest -q
Set-Location ..\Frontend; npm run build
```

The exact result depends on the current database and environment. Do not infer production metrics from test fixtures or documentation.

## Limitations

- No validated structural-damage detector is connected.
- No chemical sensor/spectroscopy detector is connected.
- DeepFashion has dataset/label preparation support, but no active validated garment model response.
- Environmental factors are configured project estimates requiring domain/source validation.