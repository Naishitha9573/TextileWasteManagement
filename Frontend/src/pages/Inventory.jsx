import React, { useState, useEffect } from 'react';
import { Plus, Search, Filter, ChevronLeft, ChevronRight, Eye, Edit, Trash2, X } from 'lucide-react';

const EMPTY_FORM = {
  batch_identifier: '',
  fabric_type: 'Cotton',
  source: '',
  manufacturer: '',
  quantity: '',
  unit: 'kg',
  color: '',
  condition: 'Good',
  collection_date: new Date().toISOString().split('T')[0],
  location: '',
  status: 'Registered',
  waste_category: '',
  notes: '',
};

export default function Inventory() {
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [query, setQuery] = useState('');
  const [fabricFilter, setFabricFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [conditionFilter, setConditionFilter] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [editingItem, setEditingItem] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);

  const fetchInventory = async (nextPage = page) => {
    try {
      const token = localStorage.getItem('token');
      const params = new URLSearchParams({
        page: nextPage.toString(),
        page_size: pageSize.toString(),
      });
      if (fabricFilter) params.append('fabric_type', fabricFilter);
      if (statusFilter) params.append('status', statusFilter);
      if (conditionFilter) params.append('condition', conditionFilter);
      const res = await fetch(`/api/inventory?${params.toString()}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setItems(data);
      }
    } catch (err) {
      console.error('Fetch inventory error', err);
    }
  };
  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/api/inventory/statistics', { headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) { console.error(err); }
  };
  useEffect(() => { fetchInventory(page); fetchStats(); }, [page, fabricFilter, statusFilter]);

  const resetForm = () => {
    setForm(EMPTY_FORM);
    setEditingItem(null);
  };

  const openCreate = () => {
    resetForm();
    setShowModal(true);
  };

  const openEdit = (item) => {
    setEditingItem(item);
    setForm({
      batch_identifier: item.batch_identifier || '',
      fabric_type: item.fabric_type || 'Cotton',
      source: item.source || '',
      manufacturer: item.manufacturer || '',
      quantity: item.quantity || '',
      unit: item.unit || 'kg',
      color: item.color || '',
      condition: item.condition || 'Good',
      collection_date: item.collection_date || new Date().toISOString().split('T')[0],
      location: item.location || '',
      status: item.status || 'Registered',
      waste_category: item.waste_category || '',
      notes: item.notes || '',
    });
    setShowModal(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const token = localStorage.getItem('token');
      const payload = {
        ...form,
        quantity: Number(form.quantity),
        collection_date: form.collection_date,
      };

      const res = await fetch(editingItem ? `/api/inventory/${editingItem.id}` : '/api/inventory', {
        method: editingItem ? 'PUT' : 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setShowModal(false);
        resetForm();
        fetchInventory(1);
        fetchStats();
      } else {
        const errorData = await res.json();
        alert(errorData.detail || 'Unable to save inventory item');
      }
    } catch (err) {
      console.error(err);
      alert('Unable to save inventory item');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this inventory item?')) return;
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/inventory/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) {
        fetchInventory(page);
        fetchStats();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return fetchInventory(1);
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/inventory/search?query=${encodeURIComponent(query)}`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) {
        const data = await res.json();
        setItems(data);
      }
    } catch (err) { console.error(err); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontFamily: 'var(--font-header)' }}>Inventory</h2>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-primary" onClick={openCreate}><Plus size={16} /> Create</button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        <div style={{ flex: 1 }}>
          <form onSubmit={handleSearch} style={{ display: 'flex', gap: '8px' }}>
            <input className="form-input" placeholder="Search by Batch ID / Fabric / Manufacturer" value={query} onChange={(e)=>setQuery(e.target.value)} />
            <button className="btn btn-secondary" type="submit"><Search size={16} /> Search</button>
          </form>
        </div>
        <div style={{ width: '220px' }}>
          <select className="form-select" value={fabricFilter} onChange={(e)=>setFabricFilter(e.target.value)}>
            <option value="">All Fabrics</option>
            <option>Cotton</option>
            <option>Polyester</option>
            <option>Denim</option>
            <option>Mixed Fabrics</option>
          </select>
        </div>
        <div style={{ width: '180px' }}>
          <select className="form-select" value={statusFilter} onChange={(e)=>setStatusFilter(e.target.value)}>
            <option value="">All Statuses</option>
            <option>Registered</option>
            <option>Analyzed</option>
            <option>Processed</option>
          </select>
        </div>
        <div style={{ width: '180px' }}>
          <select className="form-select" value={conditionFilter} onChange={(e)=>setConditionFilter(e.target.value)}>
            <option value="">Any Condition</option>
            <option>Excellent</option>
            <option>Good</option>
            <option>Fair</option>
            <option>Poor</option>
            <option>Contaminated</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '12px' }}>
        <div className="glass-panel" style={{ padding: '12px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Inventory</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{stats?stats.total_inventory:0}</div>
        </div>
        <div className="glass-panel" style={{ padding: '12px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Quantity (kg)</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{stats?stats.total_quantity:0}</div>
        </div>
        <div className="glass-panel" style={{ padding: '12px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Pending</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{stats?stats.pending:0}</div>
        </div>
        <div className="glass-panel" style={{ padding: '12px' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Processed</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{stats?stats.processed:0}</div>
        </div>
      </div>

      <div className="glass-panel custom-table-container">
        <table className="custom-table">
          <thead>
            <tr>
              <th>Batch ID</th>
              <th>Fabric</th>
              <th>Qty</th>
              <th>Source</th>
              <th>Category</th>
              <th>Condition</th>
              <th>Status</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map(it => (
              <tr key={it.id}>
                <td>{it.batch_identifier || `#${it.id}`}</td>
                <td>{it.fabric_type}</td>
                <td>{it.quantity} {it.unit || 'kg'}</td>
                <td>{it.source}</td>
                <td>{it.waste_category || '—'}</td>
                <td>{it.condition || '—'}</td>
                <td><span className={`badge ${it.status === 'Analyzed' ? 'badge-excellent' : 'badge-disposal'}`}>{it.status}</span></td>
                <td style={{ textAlign: 'right' }}>
                  <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                    <button className="btn btn-secondary" onClick={() => openEdit(it)}><Edit size={14} /></button>
                    <button className="btn btn-danger" onClick={() => handleDelete(it.id)}><Trash2 size={14} /></button>
                  </div>
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr><td colSpan="8" style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>No inventory found.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', alignItems: 'center' }}>
        <button className="btn btn-secondary" onClick={()=>setPage(p=>Math.max(1,p-1))}><ChevronLeft /></button>
        <div>Page {page}</div>
        <button className="btn btn-secondary" onClick={()=>setPage(p=>p+1)}><ChevronRight /></button>
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel animate-fade-in" style={{ maxWidth: '760px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontFamily: 'var(--font-header)', margin: 0 }}>{editingItem ? 'Edit Inventory Item' : 'Add Inventory Item'}</h3>
              <button className="btn btn-secondary" onClick={() => { setShowModal(false); resetForm(); }}><X size={16} /></button>
            </div>
            <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '12px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Batch ID</label>
                  <input className="form-input" value={form.batch_identifier} onChange={(e)=>setForm({...form,batch_identifier:e.target.value})} placeholder="e.g. B-1001" />
                </div>
                <div className="form-group">
                  <label>Fabric Type</label>
                  <select className="form-select" value={form.fabric_type} onChange={(e)=>setForm({...form,fabric_type:e.target.value})}>
                    <option>Cotton</option><option>Polyester</option><option>Wool</option><option>Silk</option><option>Linen</option><option>Denim</option><option>Nylon</option><option>Rayon</option><option>Acrylic</option><option>Mixed Fabrics</option>
                  </select>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Source</label>
                  <input className="form-input" value={form.source} onChange={(e)=>setForm({...form,source:e.target.value})} required />
                </div>
                <div className="form-group">
                  <label>Manufacturer</label>
                  <input className="form-input" value={form.manufacturer} onChange={(e)=>setForm({...form,manufacturer:e.target.value})} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Quantity</label>
                  <input className="form-input" type="number" min="0" value={form.quantity} onChange={(e)=>setForm({...form,quantity:e.target.value})} required />
                </div>
                <div className="form-group">
                  <label>Unit</label>
                  <input className="form-input" value={form.unit} onChange={(e)=>setForm({...form,unit:e.target.value})} />
                </div>
                <div className="form-group">
                  <label>Color</label>
                  <input className="form-input" value={form.color} onChange={(e)=>setForm({...form,color:e.target.value})} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Condition</label>
                  <select className="form-select" value={form.condition} onChange={(e)=>setForm({...form,condition:e.target.value})}>
                    <option>Excellent</option><option>Good</option><option>Fair</option><option>Poor</option><option>Contaminated</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Status</label>
                  <select className="form-select" value={form.status} onChange={(e)=>setForm({...form,status:e.target.value})}>
                    <option>Registered</option><option>Analyzed</option><option>Processed</option>
                  </select>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Collection Date</label>
                  <input className="form-input" type="date" value={form.collection_date} onChange={(e)=>setForm({...form,collection_date:e.target.value})} required />
                </div>
                <div className="form-group">
                  <label>Location</label>
                  <input className="form-input" value={form.location} onChange={(e)=>setForm({...form,location:e.target.value})} />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div className="form-group">
                  <label>Waste Category</label>
                  <input className="form-input" value={form.waste_category} onChange={(e)=>setForm({...form,waste_category:e.target.value})} placeholder="Reusable / Recyclable" />
                </div>
                <div className="form-group">
                  <label>Notes</label>
                  <input className="form-input" value={form.notes} onChange={(e)=>setForm({...form,notes:e.target.value})} placeholder="Details / sorting notes" />
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '8px' }}>
                <button type="button" className="btn btn-secondary" onClick={() => { setShowModal(false); resetForm(); }}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>{submitting ? 'Saving...' : 'Save Item'}</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
