# Fabric Classification API

## Architecture

The React Fabric Classification view sends the selected image as multipart form data to FastAPI. FastAPI validates the upload and passes the decoded RGB image to a process-level `FabricClassifier` singleton. The singleton loads EfficientNet-B0 once, performs deterministic validation/test preprocessing, and returns softmax probabilities from the checkpoint.

## Configuration

Set `FABRIC_MODEL_PATH` to the trained checkpoint path. Relative paths are resolved from the backend working directory.

```env
FABRIC_MODEL_PATH=../models/best_efficientnet_b0_no_weights.pth
```

The checkpoint directory must also contain `class_names.json`. The mapping is loaded from that file and determines the output class order. The service stays available to the backend while the checkpoint is absent and reports `model_loaded: false`.

## Endpoints

### `GET /api/health`

Returns runtime state without exposing filesystem paths:

```json
{
  "status": "ok",
  "model_loaded": false,
  "model_name": "EfficientNet-B0",
  "device": "cpu",
  "num_classes": null
}
```

### `POST /api/predict`

Requires the existing bearer token and accepts `multipart/form-data` with a `file` field. Supported formats are JPEG, PNG, and WebP, with a 10 MB limit.

When the model is ready:

```json
{
  "success": true,
  "prediction": {"class_name": "<model output>", "confidence": 0.0},
  "top_predictions": [
    {"class_name": "<model output>", "confidence": 0.0}
  ]
}
```

The confidence values are raw softmax probabilities in the range 0 to 1, sorted descending. The frontend renders them as percentages.

When the checkpoint is unavailable, the endpoint returns HTTP 503:

```json
{
  "success": false,
  "error": "MODEL_NOT_READY",
  "message": "The EfficientNet-B0 fabric classification model is still being trained or is not available."
}
```

Invalid or unreadable uploads return HTTP 400. Processing failures return HTTP 422. Stack traces and filesystem paths are not sent to clients.

## Preprocessing and handoff

Inference uses RGB conversion, EXIF orientation correction, resize to 256 pixels, center crop to 224 pixels, ImageNet mean/std normalization, and `torch.inference_mode()`. This matches the deterministic transform used by the EfficientNet training utilities.

After training finishes, place the completed checkpoint at the configured `FABRIC_MODEL_PATH` and restart the backend. No React changes are required.

## Running

From the repository root:

```powershell
cd Backend
python -m uvicorn main:app --reload
```

In a second terminal:

```powershell
cd Frontend
npm run dev
```
