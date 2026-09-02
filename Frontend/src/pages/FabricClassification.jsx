import React, { useEffect, useState } from 'react';
import { CheckCircle2, ImagePlus, RotateCcw, UploadCloud } from 'lucide-react';

export default function FabricClassification() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then((health) => setModelReady(health.model_loaded))
      .catch(() => setModelReady(null));
    fetch('/api/deepfashion/status', { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } })
      .then((response) => response.json())
      .then(setDeepFashion)
      .catch(() => setDeepFashion(null));
  }, []);

  const chooseFile = (event) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
    setError('');
  };

  const classify = async () => {
    if (!file) {
      setError('Choose an image before classifying.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        if (payload.error === 'MODEL_NOT_READY') {
          setError('Fabric classification model is still being trained. Please try again when the model is ready.');
        } else {
          setError(payload.detail || payload.message || 'The image could not be classified.');
        }
        return;
      }
      const material = payload.prediction;
      const colors = payload.dominant_colors || [];
      setResult({
        ...payload,
        prediction: material || null,
        top_predictions: payload.top_predictions || [],
        probabilities: payload.probabilities || {},
        color: payload.color,
        dominant_colors: colors,
      });
      setModelReady(true);
    } catch {
      setError('Unable to reach the classification service.');
    } finally {
      setLoading(false);
    }
  };

  const analyzeTexture = async () => {
    if (!file) {
      setTextureError('Choose an image before running texture analysis.');
      return;
    }
    setTextureLoading(true);
    setTextureError('');
    setTextureResult(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/texture-analysis', {
        method: 'POST',
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        setTextureError(payload.detail || payload.message || 'The image could not be analyzed for texture.');
        return;
      }
      setTextureResult(payload);
    } catch {
      setTextureError('Unable to reach the texture-analysis service.');
    } finally {
      setTextureLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview('');
    setResult(null);
    setTextureResult(null);
    setError('');
    setTextureError('');
  };

  const formatPercentage = (value) => `${Number(value).toFixed(1)}%`;

  return (
    <section className="classification-page animate-fade-in">
      <div className="classification-heading">
        <div>
          <p className="eyebrow">REAL-TIME MATERIAL INTELLIGENCE</p>
          <h1>Fabric Classification</h1>
          <p className="page-subtitle">Upload a fabric image for EfficientNet-B0 analysis.</p>
        </div>
        <span className={`model-pill ${modelReady ? 'ready' : ''}`}>
          <span className="model-dot" /> {modelReady ? 'Model ready' : 'Model training'}
        </span>
      </div>

      <div className="classification-grid">
        <div className="glass-panel classification-upload-panel">
          <div className="panel-label"><UploadCloud size={18} /> Input image</div>
          <label className="dropzone">
            {preview ? <img src={preview} alt="Selected fabric" /> : <><ImagePlus size={42} /><strong>Choose a fabric image</strong><span>JPG, PNG, or WebP up to 10 MB</span></>}
            <input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile} />
          </label>
          <div className="classification-actions">
            <button className="btn btn-primary" onClick={classify} disabled={!file || loading}>
              {loading ? 'Analyzing fabric...' : 'Classify Fabric'}
            </button>
            <button className="btn btn-secondary" onClick={analyzeTexture} disabled={!file || textureLoading}>
              {textureLoading ? 'Analyzing texture...' : 'Texture Analysis'}
            </button>
            <button className="btn btn-secondary" onClick={reset} disabled={loading || textureLoading || (!file && !result && !textureResult)} title="Reset classification">
              <RotateCcw size={17} /> Reset
            </button>
          </div>
          {error && <div className="classification-error">{error}</div>}
          {textureError && <div className="classification-error">{textureError}</div>}
        </div>

        <div className="glass-panel classification-result-panel">
          <div className="panel-label"><CheckCircle2 size={18} /> Classification result</div>
          {result ? <>
            <p className="result-caption">Fabric Analysis</p>
            {result.prediction ? <>
              <div className="result-name">{result.prediction.class_name}</div>
              {result.prediction.confidence != null && <div className="result-confidence">{formatPercentage(result.prediction.confidence * 100)} confidence</div>}
              {result.top_predictions.length > 0 && <div className="top-predictions"><h3>Top 3 Predictions</h3>{result.top_predictions.map((prediction, index) => <div className="prediction-row" key={prediction.class_name}><span>{index + 1}. {prediction.class_name}</span><strong>{formatPercentage(prediction.confidence * 100)}</strong></div>)}</div>}
              <div className="top-predictions"><h3>Probability Distribution</h3>{Object.entries(result.probabilities).map(([className, probability]) => <div className="prediction-row" key={className}><span>{className}</span><strong>{formatPercentage(probability * 100)}</strong></div>)}</div>
            </> : <div className="classification-error">Fabric model is currently unavailable.</div>}
            {result.environmental_impact && <div className="top-predictions"><h3>Estimated Environmental Impact</h3><div className="prediction-row"><span>CO2 Savings</span><strong>{result.environmental_impact.co2?.value == null ? 'Not available' : `${result.environmental_impact.co2.value} ${result.environmental_impact.co2.unit}`}</strong></div><div className="prediction-row"><span>Water Saved</span><strong>{result.environmental_impact.water?.value == null ? 'Not available' : `${result.environmental_impact.water.value} ${result.environmental_impact.water.unit}`}</strong></div><div className="color-detail">Status: {result.environmental_impact.calculation_status || 'Not available'}</div><details className="color-detail"><summary>Calculation Basis</summary><div>Material: {result.material_prediction?.material || 'Not available'}</div><div>Quantity: {result.quantity_kg ?? 'Not available'} kg</div><div>CO2 factor: {result.environmental_impact.co2?.factor ?? 'Not available'} {result.environmental_impact.co2?.factor_unit || ''}</div><div>Water factor: {result.environmental_impact.water?.factor ?? 'Not available'} {result.environmental_impact.water?.factor_unit || ''}</div><div>Source/Basis: {result.environmental_impact.basis || 'Not available'}</div></details></div>}
            {result.color && <div className="top-predictions color-analysis"><h3>Color Analysis</h3><div className="dominant-color"><span className="color-swatch" style={{ backgroundColor: result.color.hex }} /> <strong>{result.color.name}</strong><strong>{formatPercentage(result.color.percentage)}</strong></div><div className="color-detail">RGB: {result.color.rgb.join(', ')}</div><div className="color-detail">HEX: {result.color.hex}</div><h3>Dominant Color Palette</h3>{result.dominant_colors.map((color) => <div className="prediction-row" key={color.hex}><span><span className="color-swatch" style={{ backgroundColor: color.hex }} /> {color.name}</span><strong>{formatPercentage(color.percentage)}</strong></div>)}</div>}
            <div className="top-predictions"><h3>Garment Analysis</h3><div className="prediction-row"><span>DeepFashion dataset</span><strong>{deepFashion?.dataset_available ? 'AVAILABLE' : 'NOT AVAILABLE'}</strong></div><div className="prediction-row"><span>Garment inference</span><strong>MODEL NOT READY</strong></div><div className="color-detail">No garment category or attributes are shown without a compatible trained model.</div></div>
          </> : <div className="empty-result">Your model result will appear here after analysis.</div>}

          <div className="top-predictions" style={{ marginTop: '1.5rem' }}>
            <h3>Texture Analysis</h3>
            {textureResult ? <>
              <div className="prediction-row"><span>Texture type</span><strong>{textureResult.summary?.texture_type || 'Unknown'}</strong></div>
              <div className="prediction-row"><span>GLCM contrast</span><strong>{Number(textureResult.glcm?.contrast || 0).toFixed(4)}</strong></div>
              <div className="prediction-row"><span>GLCM homogeneity</span><strong>{Number(textureResult.glcm?.homogeneity || 0).toFixed(4)}</strong></div>
              <div className="prediction-row"><span>LBP variance</span><strong>{Number(textureResult.lbp?.variance || 0).toFixed(2)}</strong></div>
              <div className="prediction-row"><span>Gabor dominant orientation</span><strong>{Number(textureResult.gabor?.dominant_orientation || 0).toFixed(1)}°</strong></div>
              <div className="color-detail">Observations: {(textureResult.summary?.observations || []).join(' • ') || 'No observations available.'}</div>
              <details className="color-detail" style={{ marginTop: '0.75rem' }}>
                <summary>Raw texture JSON</summary>
                <pre style={{ whiteSpace: 'pre-wrap', marginTop: '0.5rem', fontSize: '0.78rem' }}>
                  {JSON.stringify({
                    glcm: textureResult.glcm,
                    lbp: textureResult.lbp,
                    gabor: textureResult.gabor,
                    summary: textureResult.summary,
                  }, null, 2)}
                </pre>
              </details>
            </> : <div className="empty-result">Run texture analysis after selecting an image.</div>}
          </div>
        </div>
      </div>
    </section>
  );
}
