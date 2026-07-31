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
  const [fabricType, setFabricType] = useState('Cotton');
  const [source, setSource] = useState('');
  const [quantity, setQuantity] = useState('');
  const [color, setColor] = useState('');
  const [condition, setCondition] = useState('Good');
  const [collectionDate, setCollectionDate] = useState(new Date().toISOString().split('T')[0]);
  
  // Image analysis file state
  const [uploadFile, setUploadFile] = useState(null);
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
  }, []);
  const handleRegisterBatch = async (e) => {
    e.preventDefault();
    if (!source || !quantity || !color) return;
    
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
      }
    } catch (err) {
      console.error(err);
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
                // Analyzed State - Score & Analysis Details
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  
                  {/* Radial Circularity Score */}
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
                    <div>
                      <h4 style={{ fontSize: '1rem', marginBottom: '4px' }}>Circularity Assessment</h4>
                      <span className={`badge ${getBadgeClass(selectedBatch.analysis?.circularity_category)}`}>
                        {selectedBatch.analysis?.circularity_category}
                      </span>
                    </div>
                  </div>
                  {/* Feature Breakdown Table */}
                  <div style={{ background: 'rgba(255,255,255,0.02)', padding: '14px', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Detected Color:</span>
                      <span style={{ fontWeight: '600' }}>{selectedBatch.analysis?.fabric_color}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Texture Pattern:</span>
                      <span style={{ fontWeight: '600' }}>{selectedBatch.analysis?.fabric_texture} / {selectedBatch.analysis?.fabric_pattern}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Structural Damage:</span>
                      <span style={{ color: selectedBatch.analysis?.damage_detected ? 'var(--accent-rose)' : 'var(--accent-emerald)', fontWeight: '600' }}>
                        {selectedBatch.analysis?.damage_detected ? 'Yes (Deduction)' : 'None'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Chemical Contamination:</span>
                      <span style={{ color: selectedBatch.analysis?.contamination_detected ? 'var(--accent-rose)' : 'var(--accent-emerald)', fontWeight: '600' }}>
                        {selectedBatch.analysis?.contamination_detected ? 'Alert (High risk)' : 'None'}
                      </span>
                    </div>
                  </div>
                  {/* Recommendation Card */}
                  <div style={{ background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '10px', padding: '16px' }}>
                    <h4 style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem', marginBottom: '6px', fontFamily: 'var(--font-header)' }}>Recycling Recommendation</h4>
                    <p style={{ fontSize: '0.85rem', lineHeight: '1.4', color: 'var(--text-primary)' }}>
                      {selectedBatch.analysis?.recycling_strategy}
                    </p>
                  </div>
                  {/* Environmental Impacts */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>CO2 Savings</span>
                      <p style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--accent-emerald)', margin: '2px 0' }}>-{selectedBatch.analysis?.co2_savings} kg</p>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Water Saved</span>
                      <p style={{ fontSize: '1.1rem', fontWeight: '700', color: 'var(--accent-teal)', margin: '2px 0' }}>{selectedBatch.analysis?.water_savings} L</p>
                    </div>
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
            </form>
          </div>
        </div>
      )}
    </div>
  );
}