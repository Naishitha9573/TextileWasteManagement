import React, { useState, useEffect } from 'react';
import { Users, Server, Shield, Trash2, Cpu, Activity, UserCog } from 'lucide-react';

export default function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAdminData = async () => {
    try {
      const token = localStorage.getItem('token');
      const resUsers = await fetch('/api/users', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const resAnalytics = await fetch('/api/analytics/admin', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (resUsers.ok && resAnalytics.ok) {
        setUsers(await resUsers.json());
        setAnalytics(await resAnalytics.json());
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdminData();
  }, []);

  const handleUpdateRole = async (userId, newRole) => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/users/${userId}/role`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ role: newRole })
      });
      if (res.ok) {
        fetchAdminData();
      } else {
        const errData = await res.json();
        alert(errData.detail || "Failed to update role");
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (!confirm(`Are you sure you want to permanently delete user ${username}?`)) return;
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/users/${userId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setUsers(prev => prev.filter(u => u.id !== userId));
        fetchAdminData();
      } else {
        const errData = await res.json();
        alert(errData.detail || "Failed to delete user");
      }
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px', color: 'var(--text-muted)' }}>
        Loading Admin Systems Portal...
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      <div>
        <h1>Platform Administration Control</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Manage user roles, audit active profiles, and monitor system resources.</p>
      </div>

      {/* Admin stats */}
      <div className="analytics-grid">
        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Registered Profiles</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{analytics?.total_users} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>users</span></h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)' }}>Active authorization DB</span>
          </div>
          <div style={{ background: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-emerald)', padding: '12px', borderRadius: '12px' }}>
            <Users size={24} />
          </div>
        </div>

        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Active HTTP Connections</span>
            <h2 style={{ fontSize: '2rem', margin: '4px 0' }}>{analytics?.active_connections} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>sessions</span></h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-teal)' }}>Concurrent server pollers</span>
          </div>
          <div style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--accent-teal)', padding: '12px', borderRadius: '12px' }}>
            <Cpu size={24} />
          </div>
        </div>

        <div className="glass-panel glass-card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>System Cluster Health</span>
            <h2 style={{ fontSize: '1.25rem', margin: '14px 0 8px 0', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Activity size={18} /> {analytics?.system_status}
            </h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Database size: {(analytics?.database_size_bytes / 1024).toFixed(1)} KB</span>
          </div>
          <div style={{ background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-indigo)', padding: '12px', borderRadius: '12px' }}>
            <Server size={24} />
          </div>
        </div>
      </div>

      {/* Main Split Grid */}
      <div className="dashboard-grid">
        
        {/* Left Side: Users list */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '20px', fontFamily: 'var(--font-header)' }}>User Directory & RBAC Audit</h3>

          <div className="custom-table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Current Role</th>
                  <th>Join Date</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id}>
                    <td style={{ fontWeight: '600' }}>{user.username}</td>
                    <td>{user.email}</td>
                    <td>
                      <select 
                        className="form-select" 
                        value={user.role} 
                        onChange={(e) => handleUpdateRole(user.id, e.target.value)}
                        style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                      >
                        <option>Recycling Facility Operator</option>
                        <option>Sustainability Manager</option>
                        <option>Textile Manufacturer</option>
                        <option>Administrator</option>
                      </select>
                    </td>
                    <td>{new Date(user.created_at).toLocaleDateString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button 
                        onClick={() => handleDeleteUser(user.id, user.username)}
                        style={{ background: 'transparent', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer' }}
                        title="Delete User"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Side: Security logs */}
        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-header)' }}>Role Matrix Guide</h3>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.03)', padding: '14px', borderRadius: '8px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)', marginBottom: '4px' }}>Recycling Operator</h4>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                Full access to waste registry batches, image uploads, AI analytics scoring, and circular recovery outputs.
              </p>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.03)', padding: '14px', borderRadius: '8px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--accent-teal)', marginBottom: '4px' }}>Sustainability Manager</h4>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                Access to macro environment performance graphs, carbon/water savings tracking, ESG milestones achievements, and PDF export reports.
              </p>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.01)', border: '1px solid rgba(255,255,255,0.03)', padding: '14px', borderRadius: '8px' }}>
              <h4 style={{ fontSize: '0.85rem', color: 'var(--accent-indigo)', marginBottom: '4px' }}>Textile Manufacturer</h4>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                Log factory-level offcuts, trace supply surplus, and audit plant carbon foot mitigation ratios.
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
