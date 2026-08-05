import React, { useState, useEffect } from 'react';
import { Plus, Recycle, ShieldCheck, Factory, BarChart3, AlertCircle } from 'lucide-react';

export default function ManufacturerDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Form states
  const [fabricType, setFabricType] = useState('Cotton');
  const [source, setSource] = useState('Cutting Floor (Surplus)');
  const [quantity, setQuantity] = useState('');
  const [color, setColor] = useState('');
  const [condition, setCondition] = useState('Excellent');

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('token');
      // Fetch manufacturing specific analytics
      const resAnalytic = await fetch('/api/analytics/manufacturer', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const resBatches = await fetch('/api/batches', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (resAnalytic.ok && resBatches.ok) {
        setAnalytics(await resAnalytic.json());
        setBatches(await resBatches.json());
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRegisterWaste = async (e) => {
    e.preventDefault();
    if (!quantity || !color) return;

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
          collection_date: new Date().toISOString().split('T')[0]
        })
      });

      if (res.ok) {
        setIsModalOpen(false);
        setQuantity('');
        setColor('');
        fetchData();
      }
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)' }}>
        Loading Plant Analytics...
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Manufacturer Circular Portal</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Log factory surplus, manage cutting floor waste, and monitor recovery benchmarks.</p>
        </div>
        <button onClick={() => setIsModalOpen(true)} className="btn btn-primary">
          <Plus size={18} /> Register Production Waste
        </button>
      </div>

      {/* Metrics Row */}
      <div className="analytics-grid">
        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Surplus Logged</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{analytics?.waste_generated_kg.toLocaleString()} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>kg</span></h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Surplus diverted from landfill</span>
          </div>
          <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-indigo)', padding: '12px', borderRadius: '12px' }}>
            <Factory size={24} />
          </div>
        </div>

        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Average Circularity Score</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{analytics?.average_circularity}%</h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)' }}>Circularity Health Rating</span>
          </div>
          <div style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', padding: '12px', borderRadius: '12px' }}>
            <ShieldCheck size={24} />
          </div>
        </div>

        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Diverted for Recycling</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{analytics?.recycled_percentage}%</h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)' }}>Active recycling ratio</span>
          </div>
          <div style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--accent-teal)', padding: '12px', borderRadius: '12px' }}>
            <Recycle size={24} />
          </div>
        </div>
      </div>

      {/* Main Grid split */}
      <div className="dashboard-grid">
        
        {/* Left Side: Manufacturer's logs */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '20px', fontFamily: 'var(--font-header)' }}>Production Logs</h3>
          
          <div className="custom-table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Batch ID</th>
                  <th>Fabric Type</th>
                  <th>Floor Source</th>
                  <th>Weight (kg)</th>
                  <th>Circularity</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {batches.map((batch) => (
                  <tr key={batch.id}>
                    <td>#{batch.id}</td>
                    <td style={{ fontWeight: '500' }}>{batch.fabric_type}</td>
                    <td>{batch.source}</td>
                    <td>{batch.quantity} kg</td>
                    <td style={{ fontWeight: '600', color: batch.analysis ? '#10b981' : '#6b7280' }}>
                      {batch.analysis ? `${batch.analysis.overall_circularity_score}%` : 'Pending AI'}
                    </td>
                    <td>
                      <span className={`badge ${batch.status === 'Analyzed' ? 'badge-excellent' : 'badge-disposal'}`}>
                        {batch.status}
                      </span>
                    </td>
                  </tr>
                ))}
                {batches.length === 0 && (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
                      No plant waste logs registered. Register cutting floor offcuts to begin tracking circularity.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Side: Circular Economy Insights */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-header)' }}>Circularity Analytics</h3>

          {/* Environmental offset card */}
          <div style={{ background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '12px', padding: '20px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)', fontWeight: '600' }}>Factory Offset impact</span>
            <h2 style={{ fontSize: '1.8rem', margin: '4px 0', color: '#fff' }}>-{analytics?.co2_savings_kg.toLocaleString()} kg CO₂</h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4', marginTop: '6px' }}>
              By supplying raw post-production waste directly to verified chemical and mechanical recyclers, your facility has prevented this volume of carbon emissions from initial fabric fabrication.
            </p>
          </div>

          {/* Materials logged breakdown */}
          <div>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '12px', color: 'var(--text-secondary)' }}>Logged Material Breakdown</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {analytics?.waste_by_source && Object.keys(analytics.waste_by_source).length > 0 ? (
                Object.entries(analytics.waste_by_source).map(([fabric, count]) => (
                  <div key={fabric} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                      <span>{fabric}</span>
                      <span style={{ fontWeight: '600' }}>{count} batches</span>
                    </div>
                    <div style={{ width: '100%', background: 'rgba(255,255,255,0.03)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ 
                        width: `${(count / batches.length) * 100}%`, 
                        background: 'var(--grad-primary)', 
                        height: '100%' 
                      }}></div>
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No distribution data.</div>
              )}
            </div>
          </div>
        </div>

      </div>

      {/* Log Waste Modal */}
      {isModalOpen && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel animate-fade-in">
            <h3 style={{ fontFamily: 'var(--font-header)', fontSize: '1.25rem', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '10px' }}>Log Production Waste</h3>
            
            <form onSubmit={handleRegisterWaste}>
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
                  <label>Weight (kg)</label>
                  <input 
                    type="number" 
                    className="form-input" 
                    value={quantity} 
                    onChange={(e) => setQuantity(e.target.value)} 
                    placeholder="e.g. 50" 
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Production Source Unit</label>
                <select className="form-select" value={source} onChange={(e) => setSource(e.target.value)}>
                  <option>Cutting Floor (Surplus)</option>
                  <option>Assembly Line (Offcuts)</option>
                  <option>Weaving Yard (Weft Defects)</option>
                  <option>Overstock Inventory</option>
                </select>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div className="form-group">
                  <label>Fabric Color</label>
                  <input 
                    type="text" 
                    className="form-input" 
                    value={color} 
                    onChange={(e) => setColor(e.target.value)} 
                    placeholder="e.g. Indigo Blue" 
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Cleanliness & State</label>
                  <select className="form-select" value={condition} onChange={(e) => setCondition(e.target.value)}>
                    <option>Excellent</option>
                    <option>Good</option>
                    <option>Fair</option>
                    <option>Poor</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '24px' }}>
                <button type="button" onClick={() => setIsModalOpen(false)} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Log Production Waste
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
