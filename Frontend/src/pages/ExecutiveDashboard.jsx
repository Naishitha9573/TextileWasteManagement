import React, { useEffect, useState } from 'react';
import { BarChart3, Droplet, Leaf, Package, Recycle, TrendingUp } from 'lucide-react';

const formatNumber = (value, suffix = '') => value == null ? 'No data available' : `${Number(value).toLocaleString()}${suffix}`;

function Distribution({ title, values }) {
  const entries = Object.entries(values || {});
  const maximum = Math.max(...entries.map(([, value]) => Number(value)), 1);
  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <h3 style={{ fontSize: '1.05rem', marginBottom: '18px', fontFamily: 'var(--font-header)' }}>{title}</h3>
      {entries.length === 0 ? <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data available</div> : entries.map(([label, value]) => (
        <div key={label} style={{ display: 'grid', gridTemplateColumns: '110px 1fr 48px', gap: '10px', alignItems: 'center', marginBottom: '10px', fontSize: '0.8rem' }}>
          <span>{label}</span>
          <div style={{ background: 'rgba(255,255,255,0.06)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}><div style={{ width: `${Number(value) / maximum * 100}%`, height: '100%', background: 'var(--grad-primary)' }} /></div>
          <strong style={{ textAlign: 'right' }}>{value}</strong>
        </div>
      ))}
    </div>
  );
}

export default function ExecutiveDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/analytics/executive', { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } })
      .then(async (response) => { if (!response.ok) throw new Error('Dashboard data unavailable'); return response.json(); })
      .then(setData)
      .catch((reason) => setError(reason.message));
  }, []);

  if (error) return <div className="classification-error">{error}</div>;
  if (!data) return <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)' }}>Loading Executive Dashboard...</div>;

  const cards = [
    ['Total Waste Registered', formatNumber(data.total_waste_kg, ' kg'), LayersIcon],
    ['Total Batches', formatNumber(data.total_batches), Package],
    ['Analyzed Batches', formatNumber(data.analyzed_batches), BarChart3],
    ['Recyclable Waste', formatNumber(data.category_quantities.Recyclable, ' kg'), Recycle],
    ['Reusable Waste', formatNumber(data.category_quantities.Reusable, ' kg'), Leaf],
    ['Upcyclable Waste', formatNumber(data.category_quantities.Upcyclable, ' kg'), TrendingUp],
    ['Diversion Rate', formatNumber(data.diversion_rate, '%'), BarChart3],
    ['Estimated CO2 Savings', formatNumber(data.co2_saved_kg, ' kg CO2e'), Leaf],
    ['Estimated Water Savings', formatNumber(data.water_saved_liters, ' L'), Droplet],
    ['Circularity Score', formatNumber(data.circularity_average, '%'), Recycle],
  ];

  return (
    <section className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div><p className="eyebrow">BACKEND-DERIVED OPERATING VIEW</p><h1>Executive Dashboard</h1><p className="page-subtitle">Live inventory, recovery, and sustainability metrics.</p></div>
      <div className="analytics-grid">
        {cards.map(([label, value, Icon]) => <div className="glass-panel glass-card" key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}><div><span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</span><h2 style={{ fontSize: '1.55rem', margin: '6px 0' }}>{value}</h2><span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Database/API value</span></div><Icon size={23} color="var(--accent-teal)" /></div>)}
      </div>
      <div className="dashboard-grid">
        <Distribution title="Material Distribution" values={data.material_distribution} />
        <Distribution title="Waste Category Distribution" values={data.category_distribution} />
      </div>
    </section>
  );
}

function LayersIcon(props) { return <Package {...props} />; }
