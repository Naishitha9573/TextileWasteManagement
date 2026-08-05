import React, { useState, useEffect } from 'react';
import { Globe, Droplet, FileDown, Shield, Leaf, Award } from 'lucide-react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

export default function SustainabilityDashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const fetchAnalytics = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/analytics/sustainability', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    fetchAnalytics();
  }, []);
  const downloadESGReport = () => {
    if (!analytics) return;

    const doc = new jsPDF();
    autoTable(doc, {
      head: [['Metric', 'Value achieved', 'Environmental Significance']],
      body: [
        ['Carbon Footprint Mitigation', `${(analytics.co2_saved_kg || 0).toLocaleString()} kg CO2`, 'Equivalent greenhouse gases offset from raw manufacturing'],
        ['Hydric Resource Preservation', `${(analytics.water_saved_liters || 0).toLocaleString()} Liters`, 'Water offset from pesticide/growth processes in cotton/linen'],
        ['Landfill Diversion Weight', `${(analytics.landfill_diverted_kg || 0).toLocaleString()} kg`, 'Solid textile waste completely prevented from entering standard landfills'],
        ['Waste Diversion Efficiency', `${(analytics.diversion_rate || 0)}%`, 'Percentage of registered textile batches successfully recycled/reused'],
        ['Average Circularity Index', `${(analytics.circularity_avg || 0)}%`, 'Average recovery score across all processed textile components']
      ],
      startY: 45,
      theme: 'striped',
      headStyles: { fillColor: [99, 102, 241] },
      styles: { fontSize: 9 }
    });

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(20);
    doc.text('Corporate Sustainability & ESG Action Report', 14, 20);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text(`Report Period: FY2026 Q3 | Generated: ${new Date().toLocaleDateString()}`, 14, 30);

    doc.setDrawColor(16, 185, 129);
    doc.setLineWidth(1);
    doc.line(14, 36, 196, 36);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text('Executive Summary', 14, 42);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text(
      'This document outlines the environmental offset statistics achieved through the deployment of our AI-powered Textile Waste Intelligence Platform. By analyzing, identifying, and systematically redirecting fabric components, we have diverted substantial volumes of post-consumer and pre-consumer textile waste into circular economy channels.',
      14,
      48,
      { maxWidth: 180 }
    );

    const finalY = doc.lastAutoTable.finalY + 12;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text('Sustainability Milestones', 14, finalY);

    let currentY = finalY + 8;
    (analytics.milestones || []).forEach((milestone) => {
      const status = milestone.achieved ? '[ACHIEVED]' : '[PENDING]';
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.text(`${status} - ${milestone.title}`, 14, currentY);
      doc.setFont('helvetica', 'normal');
      doc.text(milestone.desc, 14, currentY + 5);
      currentY += 15;
    });

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.text('Certified by:', 14, currentY + 10);
    doc.setFont('helvetica', 'normal');
    doc.text('Global sustainability coordination desk', 14, currentY + 15);
    doc.text('Textile Waste Intelligence Platform Network', 14, currentY + 20);

    doc.save('ESG_Impact_Report.pdf');
  };
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)' }}>
        Loading Environmental Offset Data...
      </div>
    );
  }
  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Sustainability Intelligence Center</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Track ESG metrics, carbon footprints, water conservation, and corporate sustainability metrics.</p>
        </div>
        <button onClick={downloadESGReport} className="btn btn-primary">
          <FileDown size={18} /> Export ESG Report (PDF)
        </button>
      </div>
      {/* Main Grid: Left Column Cards, Right Column Milestones */}
      <div className="dashboard-grid">
        
        {/* Left Side: Impact Counters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <div className="analytics-grid">
            <div className="glass-panel" style={{ padding: '24px', display: 'flex', gap: '20px', alignItems: 'center' }}>
              <div style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', padding: '16px', borderRadius: '16px' }}>
                <Leaf size={32} />
              </div>
              <div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>CO₂ Emissions Prevented</span>
                <h2 style={{ fontSize: '2.2rem', margin: '4px 0' }}>{analytics?.co2_saved_kg.toLocaleString()} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>kg CO₂</span></h2>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Equivalent to planting {Math.round(analytics?.co2_saved_kg / 22)} trees/year</span>
              </div>
            </div>
            <div className="glass-panel" style={{ padding: '24px', display: 'flex', gap: '20px', alignItems: 'center' }}>
              <div style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--accent-teal)', padding: '16px', borderRadius: '16px' }}>
                <Droplet size={32} />
              </div>
              <div>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Hydric Offset Saved</span>
                <h2 style={{ fontSize: '2.2rem', margin: '4px 0' }}>{analytics?.water_saved_liters.toLocaleString()} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>Liters</span></h2>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Conserved manufacturing process water</span>
              </div>
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1.2rem', marginBottom: '20px', fontFamily: 'var(--font-header)' }}>Diverted Circularity Progress</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '16px' }}>
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Solid Waste Diverted</span>
                <h3 style={{ fontSize: '1.8rem', margin: '6px 0', color: 'var(--accent-indigo)' }}>{analytics?.landfill_diverted_kg.toLocaleString()} kg</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Saved from landfill decay</span>
              </div>
              
              <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.04)', borderRadius: '12px', padding: '20px', textAlign: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Avg. Circularity Index</span>
                <h3 style={{ fontSize: '1.8rem', margin: '6px 0', color: 'var(--accent-emerald)' }}>{analytics?.circularity_avg}%</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Quality & Recyclability baseline</span>
              </div>
            </div>
            <div style={{ marginTop: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                <span>Overall Landfill Diversion Rate</span>
                <span style={{ fontWeight: '600' }}>{analytics?.diversion_rate}%</span>
              </div>
              <div style={{ width: '100%', background: 'rgba(255,255,255,0.04)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${analytics?.diversion_rate}%`, background: 'var(--grad-primary)', height: '100%', borderRadius: '4px' }}></div>
              </div>
            </div>
          </div>
        </div>
        {/* Right Side: Achievements / Milestones */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '20px', fontFamily: 'var(--font-header)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Award size={20} color="var(--accent-amber)" /> ESG Milestones
          </h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {analytics?.milestones.map((m, idx) => (
              <div 
                key={idx} 
                style={{ 
                  padding: '16px', 
                  borderRadius: '12px', 
                  background: m.achieved ? 'rgba(16, 185, 129, 0.03)' : 'rgba(255, 255, 255, 0.01)',
                  border: m.achieved ? '1px solid rgba(16, 185, 129, 0.15)' : '1px solid rgba(255, 255, 255, 0.03)',
                  display: 'flex', 
                  gap: '14px',
                  alignItems: 'flex-start'
                }}
              >
                <div style={{ 
                  background: m.achieved ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.02)',
                  color: m.achieved ? 'var(--accent-emerald)' : 'var(--text-muted)',
                  borderRadius: '50%',
                  width: '28px',
                  height: '28px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: '28px'
                }}>
                  {m.achieved ? '✓' : '○'}
                </div>
                <div>
                  <h4 style={{ fontSize: '0.95rem', color: m.achieved ? '#fff' : 'var(--text-secondary)', marginBottom: '4px' }}>
                    {m.title}
                  </h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                    {m.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
