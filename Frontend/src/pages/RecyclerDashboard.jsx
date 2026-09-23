import React, { useState, useEffect } from 'react';
import { Plus, Upload, BarChart3, RefreshCw, Layers, ShieldCheck, Download, Trash2, Eye } from 'lucide-react';
import * as XLSX from 'xlsx';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';
export default function RecyclerDashboard() {
  const [batches, setBatches] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedBatch, setSelectedBatch] = useState(null);
  
  // Form fields for new batch
  const [fabricType, setFabricType] = useState('');
  const [source, setSource] = useState('');
  const [quantity, setQuantity] = useState('');
  const [color, setColor] = useState('');
  const [condition, setCondition] = useState('Good');
  const [collectionDate, setCollectionDate] = useState(new Date().toISOString().split('T')[0]);
  
  // Image analysis file state
  const [uploadFile, setUploadFile] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [batchError, setBatchError] = useState('');
  const [textureFile, setTextureFile] = useState(null);
  const [textureResult, setTextureResult] = useState(null);
  const [textureLoading, setTextureLoading] = useState(false);
  const [textureError, setTextureError] = useState('');

  const fetchModelStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/model/status', { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) setModelStatus(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchBatches = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/batches', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setBatches(data);
        if (data.length > 0 && !selectedBatch) {
          setSelectedBatch(data[0]);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };
  useEffect(() => {
    fetchBatches();
    fetchModelStatus();
  }, []);
  const handleRegisterBatch = async (e) => {
    e.preventDefault();
    setBatchError('');
    if (!fabricType || !source || !quantity || !color || Number(quantity) <= 0) {
      setBatchError('Enter a material, source, color, and quantity greater than zero.');
      return;
    }
    
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/batches', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          fabric_type: fabricType,
          source,
          quantity: parseFloat(quantity),
          color,
          condition,
          collection_date: collectionDate
        })
      });
      if (res.ok) {
        setIsModalOpen(false);
        // Clear fields
        setSource('');
        setQuantity('');
        setColor('');
        fetchBatches();
      } else {
        const payload = await res.json().catch(() => ({}));
        setBatchError(payload.detail || 'Unable to save batch. Please check database connection.');
      }
    } catch (err) {
      console.error(err);
      setBatchError('Unable to save batch. Please check database connection.');
    }
  };
  const handleDelete = async (id) => {
    if (!confirm(`Are you sure you want to delete Batch #${id}?`)) return;
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/batches/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setBatches(prev => prev.filter(b => b.id !== id));
        if (selectedBatch && selectedBatch.id === id) {
          setSelectedBatch(null);
        }
      }
    } catch (err) {
      console.error(err);
    }
  };
  const handleImageAnalysis = async (batchId) => {
    setIsAnalyzing(true);
    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      if (uploadFile) {
        formData.append('file', uploadFile);
      }
      const res = await fetch(`/api/batches/${batchId}/analyze`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });
      if (res.ok) {
        const updatedBatch = await res.json();
        // Update local list
        setBatches(prev => prev.map(b => b.id === batchId ? updatedBatch : b));
        setSelectedBatch(updatedBatch);
        setUploadFile(null);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTextureAnalysis = async () => {
    if (!textureFile) {
      setTextureError('Choose an image before running texture analysis.');
      return;
    }

    setTextureLoading(true);
    setTextureError('');
    setTextureResult(null);

    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('file', textureFile);

      const res = await fetch('/api/texture-analysis', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        setTextureError(payload.detail || payload.message || 'The image could not be analyzed for texture.');
        return;
      }

      setTextureResult(payload);
    } catch (err) {
      console.error(err);
      setTextureError('Unable to reach the texture-analysis service.');
    } finally {
      setTextureLoading(false);
    }
  };
  const getScoreColor = (score) => {
    if (score >= 85) return '#10b981'; // Emerald
    if (score >= 70) return '#06b6d4'; // Teal
    if (score >= 50) return '#f59e0b'; // Amber
    return '#ef4444'; // Rose
  };
  const getBadgeClass = (category) => {
    if (!category) return 'badge-disposal';
    if (category.includes('Excellent')) return 'badge-excellent';
    if (category.includes('High')) return 'badge-high';
    if (category.includes('Moderate')) return 'badge-moderate';
    if (category.includes('Limited')) return 'badge-limited';
    return 'badge-disposal';
  };
  // API-based Report Exporters (Module 13)
  const exportReportAPI = async (format) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`/api/reports/${format}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `textile_waste_report.${format === 'pdf' ? 'pdf' : format === 'excel' ? 'xlsx' : 'csv'}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } else {
        alert('Failed to export report');
      }
    } catch (err) {
      console.error('Export error:', err);
      alert('Error exporting report');
    }
  };

  const exportToExcel = () => {
    exportReportAPI('excel');
  };

  const exportToPDF = () => {
    exportReportAPI('pdf');
  };

  const exportToCSV = () => {
    exportReportAPI('csv');
  };
  // Compute local metrics for Operator view
  const totalWeight = batches.reduce((acc, curr) => acc + curr.quantity, 0);
  const analyzedWeight = batches.filter(b => b.status === 'Analyzed' && b.analysis)
                               .reduce((acc, curr) => acc + curr.quantity, 0);
  const diversionRate = totalWeight > 0 ? (analyzedWeight / totalWeight) * 100 : 0;
  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Upper Metrics Grid */}
      <div className="analytics-grid">
        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Registered Waste</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{totalWeight.toLocaleString()} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>kg</span></h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)' }}>Active Inventory Tracked</span>
          </div>
          <div style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', padding: '12px', borderRadius: '12px' }}>
            <Layers size={24} />
          </div>
        </div>
        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Analyzed & Classified</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{batches.filter(b => b.status === 'Analyzed').length} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>batches</span></h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)' }}>Ready for redirection</span>
          </div>
          <div style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--accent-teal)', padding: '12px', borderRadius: '12px' }}>
            <ShieldCheck size={24} />
          </div>
        </div>
        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Waste Diversion Rate</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{diversionRate.toFixed(1)}%</h2>
            <div style={{ width: '100%', background: 'rgba(255,255,255,0.05)', height: '4px', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' }}>
              <div style={{ width: `${diversionRate}%`, background: 'var(--grad-primary)', height: '100%' }}></div>
            </div>
          </div>
          <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-indigo)', padding: '12px', borderRadius: '12px' }}>
            <BarChart3 size={24} />
          </div>
        </div>
      </div>
      {/* Main split grid */}
      <div className="dashboard-grid">
        
        {/* Left Side: Waste Inventory */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1.25rem', fontFamily: 'var(--font-header)' }}>Inventory Registry</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button onClick={() => setIsModalOpen(true)} className="btn btn-primary">
                <Plus size={18} /> Log Batch
              </button>
              <button 
                onClick={exportToExcel} 
                className="btn btn-secondary" 
                title="Export Excel"
              >
                <Download size={18} /> Excel
              </button>
              <button 
                onClick={exportToPDF} 
                className="btn btn-secondary" 
                title="Export PDF Report"
              >
                <Download size={18} /> PDF
              </button>
              <button 
                onClick={exportToCSV} 
                className="btn btn-secondary" 
                title="Export CSV Report"
              >
                <Download size={18} /> CSV
              </button>
            </div>
          </div>
          <div className="custom-table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Fabric</th>
                  <th>Quantity (kg)</th>
                  <th>Source</th>
                  <th>Condition</th>
                  <th>Status</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {batches.map((batch) => (
                  <tr 
                    key={batch.id} 
                    style={{ 
                      cursor: 'pointer', 
                      background: selectedBatch && selectedBatch.id === batch.id ? 'rgba(255,255,255,0.03)' : 'transparent' 
                    }}
                    onClick={() => setSelectedBatch(batch)}
                  >
                    <td>#{batch.id}</td>
                    <td style={{ fontWeight: '500' }}>{batch.fabric_type}</td>
                    <td>{batch.quantity} kg</td>
                    <td>{batch.source}</td>
                    <td>
                      <span style={{
                        fontSize: '0.8rem',
                        color: batch.condition === 'Excellent' ? '#10b981' : batch.condition === 'Good' ? '#3b82f6' : batch.condition === 'Fair' ? '#f59e0b' : '#ef4444'
                      }}>
                        {batch.condition}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${batch.status === 'Analyzed' ? 'badge-excellent' : 'badge-disposal'}`}>
                        {batch.status}
                      </span>
                    </td>
                    <td style={{ textAlign: 'right' }} onClick={(e) => e.stopPropagation()}>
                      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                        <button 
                          onClick={() => setSelectedBatch(batch)}
                          style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
                          title="View Details"
                        >
                          <Eye size={16} />
                        </button>
                        <button 
                          onClick={() => handleDelete(batch.id)}
                          style={{ background: 'transparent', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer' }}
                          title="Delete Batch"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {batches.length === 0 && (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                      No batches logged yet. Use "Log Batch" to start tracking.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        {/* Right Side: Image Analysis & Circularity Analytics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

          {modelStatus && (
            <div className="glass-panel" style={{ padding: '16px', borderLeft: `4px solid ${modelStatus.available ? 'var(--accent-teal)' : 'var(--accent-rose)'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h4 style={{ fontFamily: 'var(--font-header)', margin: 0 }}>Material Classifier — EfficientNet-B0</h4>
                <span style={{
                  fontSize: '0.7rem', padding: '2px 8px', borderRadius: '4px', fontWeight: 600,
                  background: modelStatus.available ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  color: modelStatus.available ? 'var(--accent-emerald)' : 'var(--accent-rose)'
                }}>
                  {modelStatus.available ? 'ML MODEL ACTIVE' : (modelStatus.model_status || 'UNAVAILABLE')}
                </span>
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'grid', gap: '4px' }}>
                <span>Architecture: {modelStatus.architecture || 'EfficientNet-B0'} ({modelStatus.device || 'unknown'})</span>
                <span>Classes: {modelStatus.classes?.join(', ') || 'Unavailable until checkpoint loads'}</span>
                <span>Model status: {modelStatus.model_loaded ? 'AVAILABLE' : 'MODEL_NOT_READY'}</span>
              </div>
            </div>
          )}
          
          {selectedBatch ? (
            <div className="glass-panel" style={{ padding: '24px' }}>
              <div style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '16px', marginBottom: '16px' }}>
                <h3 style={{ fontFamily: 'var(--font-header)' }}>Batch Intelligence Details</h3>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>ID: #{selectedBatch.id} | Fabric: {selectedBatch.fabric_type}</span>
              </div>
              {selectedBatch.status === 'Registered' ? (
                // Unanalyzed State - Image Upload Section
                <div style={{ textAlign: 'center', padding: '20px 0' }}>
                  <div style={{
                    border: '2px dashed rgba(255,255,255,0.1)',
                    borderRadius: '12px',
                    padding: '30px 20px',
                    cursor: 'pointer',
                    background: 'rgba(255,255,255,0.01)',
                    transition: 'var(--transition-smooth)'
                  }}
                  className="image-dropzone"
                  onClick={() => document.getElementById('fabric-image-input').click()}
                  >
                    <Upload size={32} color="var(--accent-emerald)" style={{ marginBottom: '12px' }} />
                    <h4 style={{ fontSize: '0.95rem', marginBottom: '4px' }}>Upload Fabric Sample Image</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Supports JPG, PNG (Max 5MB)</p>
                    <input 
                      type="file" 
                      id="fabric-image-input" 
                      style={{ display: 'none' }} 
                      accept="image/*"
                      onChange={(e) => {
                        if (e.target.files.length > 0) {
                          setUploadFile(e.target.files[0]);
                        }
                      }}
                    />
                  </div>
                  {uploadFile && (
                    <div style={{ marginTop: '14px', fontSize: '0.85rem', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                      Selected: {uploadFile.name}
                    </div>
                  )}
                  <button 
                    onClick={() => handleImageAnalysis(selectedBatch.id)}
                    className="btn btn-primary"
                    style={{ marginTop: '20px', width: '100%', justifyContent: 'center' }}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing ? (
                      <>
                        <RefreshCw className="animate-spin" size={18} /> Processing Image...
                      </>
                    ) : (
                      'Trigger AI Classification Engine'
                    )}
                  </button>
                </div>
              ) : (
                // Analyzed State — ML classification and rule-engine recovery
                // assessment are intentionally displayed as separate sections.
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {(() => {
                    const mc = selectedBatch.analysis?.classification_report?.material_classification;
                    const source = mc?.source || selectedBatch.analysis?.prediction_source;
                    const mlActive = source === 'MODEL' && (mc?.model_available !== false);
                    const mlUnavailable = !mlActive && (source === 'MODEL_NOT_AVAILABLE' || mc?.model_available === false);
                    const probabilities = mc?.probabilities && Object.keys(mc.probabilities).length > 0 ? mc.probabilities : null;
                    const maxProb = probabilities ? Math.max(...Object.values(probabilities)) : 0;

                    return (
                      /* ── SECTION A: ML MATERIAL CLASSIFICATION ── */
                      <div style={{
                        background: (mc?.manual_review_required || selectedBatch.analysis?.manual_review_required)
                          ? 'rgba(239, 68, 68, 0.08)' : 'rgba(255,255,255,0.02)',
                        border: (mc?.manual_review_required || selectedBatch.analysis?.manual_review_required)
                          ? '1px solid rgba(239, 68, 68, 0.25)' : '1px solid rgba(255,255,255,0.06)',
                        borderRadius: '10px', padding: '14px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <h4 style={{ fontSize: '0.9rem', fontFamily: 'var(--font-header)', margin: 0 }}>Material Classification</h4>
                          <span style={{
                            fontSize: '0.7rem', padding: '2px 8px', borderRadius: '4px', fontWeight: 600,
                            background: mlActive ? 'rgba(16, 185, 129, 0.15)' : (mlUnavailable ? 'rgba(239, 68, 68, 0.15)' : 'rgba(99, 102, 241, 0.2)'),
                            color: mlActive ? 'var(--accent-emerald)' : (mlUnavailable ? 'var(--accent-rose)' : '#818cf8')
                          }}>
                            {mlActive ? 'ML MODEL ACTIVE' : (mlUnavailable ? 'ML MODEL UNAVAILABLE' : 'NOT AVAILABLE')}
                          </span>
                        </div>

                        {mlActive && (
                          <div style={{ fontSize: '0.85rem', display: 'grid', gap: '6px' }}>
                            <div><strong>Predicted Material:</strong> {mc?.predicted_fabric || selectedBatch.analysis?.predicted_material || 'UNKNOWN / UNSUPPORTED'}</div>
                            <div><strong>ML Confidence:</strong> {mc?.confidence ?? selectedBatch.analysis?.material_confidence ?? '—'}% ({mc?.confidence_status || selectedBatch.analysis?.confidence_status || 'N/A'})</div>
                            <div><strong>Model:</strong> {mc?.model_name || 'EfficientNet-B0'}</div>
                            <div><strong>Model Status:</strong> {mc?.model_status || 'MODEL_NOT_READY'}</div>
                            {probabilities && (
                              <div style={{ marginTop: '6px', display: 'grid', gap: '4px' }}>
                                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Class probability distribution</span>
                                {Object.entries(probabilities).map(([cls, pct]) => (
                                  <div key={cls} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 52px', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
                                    <span>{cls}</span>
                                    <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                                      <div style={{ width: `${pct}%`, maxWidth: '100%', height: '100%', background: pct === maxProb ? 'var(--grad-primary)' : 'rgba(255,255,255,0.18)' }} />
                                    </div>
                                    <span style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>{Number(pct).toFixed(1)}%</span>
                                  </div>
                                ))}
                              </div>
                            )}
                            {(mc?.manual_review_required || selectedBatch.analysis?.manual_review_required || mc?.unknown_or_unsupported) && (
                              <div style={{ color: 'var(--accent-rose)', fontWeight: 600, marginTop: '4px', background: 'rgba(239,68,68,0.1)', padding: '6px 10px', borderRadius: '6px' }}>
                                Unable to reliably identify this material with the current model.
                              </div>
                            )}
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>ML confidence comes only from the model's prediction probability and is independent of the recovery score below.</div>
                          </div>
                        )}

                        {mlUnavailable && (
                          <div style={{ fontSize: '0.85rem', display: 'grid', gap: '6px' }}>
                            <div>Material classification could not be performed.</div>
                            <div style={{ color: 'var(--accent-rose)', fontWeight: 600 }}>Reason: {mc?.error || mc?.warning || 'Model could not be loaded for inference.'}</div>
                            <div style={{ color: 'var(--text-secondary)' }}>Rule-based recovery assessment is still available below.</div>
                          </div>
                        )}

                        {!mlActive && !mlUnavailable && (
                          <div style={{ fontSize: '0.85rem', display: 'grid', gap: '6px' }}>
                            <div><strong>ML Classification:</strong> Not Available</div>
                            <div><strong>ML Confidence:</strong> —</div>
                            {selectedBatch.fabric_type && (
                              <div><strong>User-declared fabric (not ML-verified):</strong> {selectedBatch.fabric_type}</div>
                            )}
                            <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                              Historical Result — no model prediction was stored for this batch.
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* ── SECTION B: TEXTURE ANALYSIS ── */}
                  <div style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '10px',
                    padding: '14px',
                    display: 'grid',
                    gap: '10px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h4 style={{ fontSize: '0.9rem', fontFamily: 'var(--font-header)', margin: 0 }}>Texture Analysis</h4>
                      <span style={{
                        fontSize: '0.7rem', padding: '2px 8px', borderRadius: '4px', fontWeight: 600,
                        background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc'
                      }}>
                        HANDCRAFTED
                      </span>
                    </div>

                    <div style={{ display: 'grid', gap: '10px' }}>
                      <label style={{ display: 'grid', gap: '6px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        Fabric sample image for texture scan
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => {
                            const nextFile = e.target.files?.[0] || null;
                            setTextureFile(nextFile);
                            setTextureError('');
                            setTextureResult(null);
                          }}
                          style={{ color: 'var(--text-primary)' }}
                        />
                      </label>

                      {textureFile && (
                        <div style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                          Selected: {textureFile.name}
                        </div>
                      )}

                      <button
                        onClick={handleTextureAnalysis}
                        className="btn btn-secondary"
                        style={{ width: '100%', justifyContent: 'center' }}
                        disabled={!textureFile || textureLoading}
                      >
                        {textureLoading ? 'Analyzing texture...' : 'Run Texture Analysis'}
                      </button>

                      {textureError && (
                        <div style={{ color: 'var(--accent-rose)', fontSize: '0.8rem', background: 'rgba(239, 68, 68, 0.08)', borderRadius: '8px', padding: '8px 10px' }}>
                          {textureError}
                        </div>
                      )}

                      {textureResult && (
                        <div style={{ display: 'grid', gap: '8px', fontSize: '0.82rem' }}>
                          <div><strong>Texture type:</strong> {textureResult.summary?.texture_type || 'Unknown'}</div>
                          <div><strong>GLCM contrast:</strong> {Number(textureResult.glcm?.contrast || 0).toFixed(4)}</div>
                          <div><strong>GLCM homogeneity:</strong> {Number(textureResult.glcm?.homogeneity || 0).toFixed(4)}</div>
                          <div><strong>LBP variance:</strong> {Number(textureResult.lbp?.variance || 0).toFixed(2)}</div>
                          <div><strong>Gabor dominant orientation:</strong> {Number(textureResult.gabor?.dominant_orientation || 0).toFixed(1)}°</div>

                          <div style={{ color: 'var(--text-secondary)' }}>
                            Observations: {(textureResult.summary?.observations || []).join(' • ') || 'No observations available.'}
                          </div>

                          <details style={{ color: 'var(--text-secondary)' }}>
                            <summary style={{ cursor: 'pointer' }}>Raw texture JSON</summary>
                            <pre style={{ marginTop: '8px', whiteSpace: 'pre-wrap', fontSize: '0.76rem', background: 'rgba(0,0,0,0.15)', borderRadius: '8px', padding: '8px' }}>
                              {JSON.stringify({
                                glcm: textureResult.glcm,
                                lbp: textureResult.lbp,
                                gabor: textureResult.gabor,
                                summary: textureResult.summary,
                              }, null, 2)}
                            </pre>
                          </details>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* ── SECTION C: RULE-BASED RECOVERY ASSESSMENT ── */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div className="score-circle-wrapper">
                      <svg className="score-radial" width="120" height="120">
                        <defs>
                          <linearGradient id="emeraldTealGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#10b981" />
                            <stop offset="100%" stopColor="#06b6d4" />
                          </linearGradient>
                        </defs>
                        <circle className="bg" cx="60" cy="60" r="50" />
                        <circle 
                          className="progress" 
                          cx="60" 
                          cy="60" 
                          r="50" 
                          strokeDasharray={2 * Math.PI * 50}
                          strokeDashoffset={2 * Math.PI * 50 * (1 - (selectedBatch.analysis?.overall_circularity_score || 0) / 100)}
                        />
                      </svg>
                      <div className="score-circle-value">
                        {selectedBatch.analysis?.overall_circularity_score}%
                      </div>
                    </div>

                    <div style={{ display: 'grid', gap: '6px' }}>
                      <h4 style={{ fontSize: '1rem', margin: 0 }}>Recovery Potential</h4>
                      <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-emerald)', fontWeight: 600 }}>
                        RULE ENGINE
                      </span>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                        Assessment Method: Rule Engine
                      </div>
                      <span className={`badge ${getBadgeClass(selectedBatch.analysis?.circularity_category)}`}>
                        {selectedBatch.analysis?.circularity_category}
                      </span>
                    </div>
                  </div>
                  {/* ── SECTION C: RECYCLING RECOMMENDATION (RULE ENGINE) ── */}
                  <div style={{ background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '10px', padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <h4 style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem', fontFamily: 'var(--font-header)', margin: 0 }}>Recycling Recommendation</h4>
                      <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(16, 185, 129, 0.15)', color: 'var(--accent-emerald)', fontWeight: 600 }}>
                        RULE ENGINE
                      </span>
                    </div>
                    <p style={{ fontSize: '0.85rem', lineHeight: '1.4', color: 'var(--text-primary)' }}>
                      {selectedBatch.analysis?.recycling_strategy}
                    </p>
                  </div>
                  {/* Environmental Impacts */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Estimated Environmental Impact</span>
                      <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(6, 182, 212, 0.15)', color: 'var(--accent-teal)', fontWeight: 600 }}>
                        ESTIMATED
                      </span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>CO2 Savings</span>
                        <p style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--accent-emerald)', margin: '2px 0' }}>
                          {selectedBatch.analysis?.co2_savings == null ? 'Not available' : `${selectedBatch.analysis.co2_savings} kg CO2e`}
                        </p>
                      </div>
                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Water Saved</span>
                        <p style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--accent-teal)', margin: '2px 0' }}>
                          {selectedBatch.analysis?.water_savings == null ? 'Not available' : `${selectedBatch.analysis.water_savings} L`}
                        </p>
                      </div>
                    </div>
                    <details style={{ marginTop: '10px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      <summary style={{ cursor: 'pointer' }}>Calculation Basis</summary>
                      <div style={{ display: 'grid', gap: '4px', marginTop: '8px' }}>
                        <span>Material: {selectedBatch.analysis?.predicted_material || selectedBatch.fabric_type || 'Unavailable'}</span>
                        <span>Quantity: {selectedBatch.quantity ?? 'Unavailable'} kg</span>
                        <span>CO2 factor: {selectedBatch.analysis?.co2_factor ?? 'Not available'} kg CO2e/kg</span>
                        <span>Water factor: {selectedBatch.analysis?.water_factor ?? 'Not available'} L/kg</span>
                        <span>Status: {selectedBatch.analysis?.co2_savings == null || selectedBatch.analysis?.water_savings == null ? 'Not available' : 'ESTIMATED'}</span>
                        <span>Source/Basis: {selectedBatch.analysis?.methodology || 'Not available'}</span>
                      </div>
                    </details>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="glass-panel" style={{ padding: '40px 24px', textAlign: 'center', color: 'var(--text-muted)' }}>
              Select a batch from the registry to perform AI image classification and inspect Circular Economy metrics.
            </div>
          )}
        </div>
      </div>
      {/* Register Batch Modal */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel animate-fade-in">
            <h3 style={{ fontFamily: 'var(--font-header)', fontSize: '1.25rem', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '10px' }}>Register Textile Waste Batch</h3>
            
            <form onSubmit={handleRegisterBatch}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div className="form-group">
                  <label>Fabric Type</label>
                  <select className="form-select" value={fabricType} onChange={(e) => setFabricType(e.target.value)}>
                    <option value="">Select material</option>
                    <option>Cotton</option>
                    <option>Polyester</option>
                    <option>Wool</option>
                    <option>Silk</option>
                    <option>Linen</option>
                    <option>Denim</option>
                    <option>Nylon</option>
                    <option>Rayon</option>
                    <option>Acrylic</option>
                    <option>Mixed Fabrics</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Batch Quantity (kg)</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    value={quantity} 
                    onChange={(e) => setQuantity(e.target.value)} 
                    placeholder="e.g. 150" 
                    required
                  />
                </div>
              </div>
              <div className="form-group">
                <label>Source / Supplier</label>
                <input 
                  type="text" 
                  className="form-input" 
                  value={source} 
                  onChange={(e) => setSource(e.target.value)} 
                  placeholder="e.g. GreenSpin Manufacturers" 
                  required
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div className="form-group">
                  <label>Dominant Fabric Color</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    value={color} 
                    onChange={(e) => setColor(e.target.value)} 
                    placeholder="e.g. Cream White" 
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Material Condition</label>
                  <select className="form-select" value={condition} onChange={(e) => setCondition(e.target.value)}>
                    <option>Excellent</option>
                    <option>Good</option>
                    <option>Fair</option>
                    <option>Poor</option>
                    <option>Contaminated</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label>Collection Date</label>
                <input 
                  type="date" 
                  className="form-input" 
                  value={collectionDate} 
                  onChange={(e) => setCollectionDate(e.target.value)} 
                  required
                />
              </div>
              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
                <button type="button" onClick={() => setIsModalOpen(false)} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Log to Inventory
                </button>
              </div>
              {batchError && <div style={{ color: 'var(--accent-rose)', marginTop: '12px', fontSize: '0.85rem' }}>{batchError}</div>}
            </form>
          </div>
        </div>
      )}
    </div>
  );
}