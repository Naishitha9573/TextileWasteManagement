import React, { useState } from 'react';
import { Download, FileBarChart } from 'lucide-react';

export default function Reports() {
  const [message, setMessage] = useState('');
  const download = async (format) => {
    setMessage('');
    try {
      const response = await fetch(`/api/reports/${format}`, { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` } });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Report data unavailable');
      }
      const blob = await response.blob();
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `textile-waste-report.${format === 'excel' ? 'xlsx' : format}`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (error) {
      setMessage(error.message);
    }
  };

  return (
    <section className="classification-page animate-fade-in">
      <div className="classification-heading"><div><p className="eyebrow">DATABASE-BACKED EXPORTS</p><h1>Reports</h1><p className="page-subtitle">Generate reports from the current authorized batch records.</p></div></div>
      <div className="analytics-grid">
        {['pdf', 'excel', 'csv'].map((format) => <div className="glass-panel glass-card" key={format}><FileBarChart size={24} color="var(--accent-teal)" /><h3 style={{ margin: '14px 0 6px' }}>{format.toUpperCase()} report</h3><p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', minHeight: '40px' }}>Real batch, material, recovery, circularity, and environmental records.</p><button className="btn btn-primary" onClick={() => download(format)}><Download size={17} /> Download</button></div>)}
      </div>
      {message && <div className="classification-error">{message}</div>}
    </section>
  );
}
