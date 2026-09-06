import React, { useState } from 'react';
import { ImagePlus, Recycle, UploadCloud } from 'lucide-react';

export default function WasteCategorization() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const classify = async () => {
    if (!file) return setError('Choose an image before classifying.');
    setLoading(true); setError(''); setResult(null);
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch('/api/waste/predict', { method: 'POST', headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }, body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Waste model is unavailable.');
      setResult(payload);
    } catch (requestError) { setError(requestError.message); } finally { setLoading(false); }
  };

  return <section className="classification-page animate-fade-in">
    <div className="classification-heading"><div><p className="eyebrow">END-OF-LIFE INTELLIGENCE</p><h1>Waste Categorization</h1><p className="page-subtitle">Evaluate a second-hand garment with the independent waste model.</p></div></div>
    <div className="classification-grid">
      <div className="glass-panel classification-upload-panel"><div className="panel-label"><UploadCloud size={18} /> Input image</div>
        <label className="dropzone"><ImagePlus size={42} /><strong>{file ? file.name : 'Choose a garment image'}</strong><span>JPG, PNG, or WebP</span><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => { setFile(event.target.files?.[0] || null); setResult(null); }} /></label>
        <button className="btn btn-primary" onClick={classify} disabled={!file || loading}>{loading ? 'Analyzing garment...' : 'Categorize Waste'}</button>
        {error && <div className="classification-error">{error}</div>}
      </div>
      <div className="glass-panel classification-result-panel"><div className="panel-label"><Recycle size={18} /> Waste result</div>
        {result ? <><p className="result-caption">Predicted category</p><div className="result-name">{result.predicted_category}</div><div className="result-confidence">{(result.confidence * 100).toFixed(1)}% confidence</div><div className="top-predictions"><h3>Class probabilities</h3>{Object.entries(result.probabilities).map(([name, probability]) => <div className="prediction-row" key={name}><span>{name}</span><strong>{(probability * 100).toFixed(1)}%</strong></div>)}</div><div className="top-predictions"><h3>Recommendation</h3><p>Use the predicted category with garment condition and material evidence in the recommendation engine.</p></div></> : <div className="empty-result">Your waste result will appear here after analysis.</div>}
      </div>
    </div>
  </section>;
}