import React, { useState, useEffect } from 'react';
import { ShieldAlert, LogOut, Key, ArrowRight, UserCheck, Sparkles } from 'lucide-react';

// Dashboards
import RecyclerDashboard from './pages/RecyclerDashboard';
import SustainabilityDashboard from './pages/SustainabilityDashboard';
import ManufacturerDashboard from './pages/ManufacturerDashboard';
import AdminDashboard from './pages/AdminDashboard';

// Components
import NotificationCenter from './components/NotificationCenter';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [user, setUser] = useState(null);
  const [role, setRole] = useState(localStorage.getItem('role') || '');
  
  // Auth Form State
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [signupRole, setSignupRole] = useState('Recycling Facility Operator');
  const [error, setError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  const fetchProfile = async (authToken) => {
    try {
      const res = await fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      if (res.ok) {
        const profile = await res.json();
        setUser(profile);
        setRole(profile.role);
        localStorage.setItem('role', profile.role);
      } else {
        // Token expired/invalid
        handleLogout();
      }
    } catch (err) {
      console.error("Failed to fetch profile", err);
    }
  };

  useEffect(() => {
    if (token) {
      fetchProfile(token);
    }
  }, [token]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setAuthLoading(true);
    try {
      const res = await fetch('/api/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('role', data.role);
        setToken(data.access_token);
        setRole(data.role);
        setUsername('');
        setPassword('');
      } else {
        const errData = await res.json();
        setError(errData.detail || "Invalid username or password");
      }
    } catch (err) {
      setError("Server connection failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setAuthLoading(true);
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, role: signupRole })
      });

      if (res.ok) {
        setIsRegister(false);
        alert("Registration successful! Please login.");
      } else {
        const errData = await res.json();
        setError(errData.detail || "Registration failed");
      }
    } catch (err) {
      setError("Server connection failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleOAuthLogin = async (provider) => {
    setError('');
    setAuthLoading(true);
    try {
      const mockOAuthPayload = {
        email: `${provider.toLowerCase()}_demo_user@textilewaste.org`,
        name: `${provider} Sandbox User`
      };
      
      const res = await fetch('/api/auth/oauth-mock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockOAuthPayload)
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('role', data.role);
        setToken(data.access_token);
        setRole(data.role);
      } else {
        setError("OAuth simulation failed");
      }
    } catch (err) {
      setError("OAuth server connection failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    setToken('');
    setRole('');
    setUser(null);
  };

  // Helper sandbox override to inspect other role views instantly
  const handleSandboxRoleOverride = (newRole) => {
    setRole(newRole);
    localStorage.setItem('role', newRole);
  };

  const renderDashboard = () => {
    switch (role) {
      case 'Recycling Facility Operator':
        return <RecyclerDashboard />;
      case 'Sustainability Manager':
        return <SustainabilityDashboard />;
      case 'Textile Manufacturer':
        return <ManufacturerDashboard />;
      case 'Administrator':
        return <AdminDashboard />;
      default:
        return (
          <div style={{ textAlign: 'center', padding: '100px' }}>
            <h2>Unauthorized Role Access</h2>
            <p>Please contact an Administrator or select another preview role.</p>
          </div>
        );
    }
  };

  if (!token) {
    // RENDER LOGIN / REGISTER CARD
    return (
      <div className="auth-container">
        <div className="glass-panel auth-card animate-fade-in">
          <div style={{ textAlign: 'center', marginBottom: '30px' }}>
            <div style={{
              display: 'inline-flex',
              background: 'var(--grad-primary)',
              borderRadius: '12px',
              padding: '10px',
              color: '#042f2e',
              marginBottom: '12px'
            }}>
              <Sparkles size={28} />
            </div>
            <h2 style={{ fontFamily: 'var(--font-header)', fontSize: '1.6rem', fontWeight: '800' }}>
              {isRegister ? 'Create Platform Account' : 'Textile Waste Intelligence'}
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '6px' }}>
              {isRegister ? 'Join the circular textile economy network' : 'AI-powered material classification & ESG analytics'}
            </p>
          </div>

          {error && (
            <div style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              color: 'var(--accent-rose)',
              padding: '10px 14px',
              borderRadius: '8px',
              fontSize: '0.85rem',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <ShieldAlert size={16} />
              {error}
            </div>
          )}

          <form onSubmit={isRegister ? handleRegister : handleLogin}>
            <div className="form-group">
              <label>Username</label>
              <input 
                type="text" 
                className="form-input" 
                value={username} 
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. recycler"
                required
              />
            </div>

            {isRegister && (
              <div className="form-group">
                <label>Email Address</label>
                <input 
                  type="email" 
                  className="form-input" 
                  value={email} 
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="e.g. operator@facility.com"
                  required
                />
              </div>
            )}

            <div className="form-group" style={{ marginBottom: '20px' }}>
              <label>Password</label>
              <input 
                type="password" 
                className="form-input" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)}
                placeholder="e.g. recycler123"
                required
              />
            </div>

            {isRegister && (
              <div className="form-group">
                <label>Default Workspace Role</label>
                <select className="form-select" value={signupRole} onChange={(e) => setSignupRole(e.target.value)}>
                  <option>Recycling Facility Operator</option>
                  <option>Sustainability Manager</option>
                  <option>Textile Manufacturer</option>
                  <option>Administrator</option>
                </select>
              </div>
            )}

            <button type="submit" className="btn btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: '10px' }} disabled={authLoading}>
              {isRegister ? 'Register Account' : 'Authenticate Credentials'} <ArrowRight size={18} />
            </button>
          </form>

          {/* Social Sign-in Mockup */}
          <div style={{ marginTop: '24px', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '20px' }}>
            <span style={{ display: 'block', textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px' }}>
              OR LOGIN INSTANTLY VIA
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <button onClick={() => handleOAuthLogin('Google')} className="btn btn-secondary" style={{ justifyContent: 'center', fontSize: '0.8rem', gap: '6px' }}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                  <path d="M12.24 10.285V13.4h6.887c-.275 1.565-1.88 4.604-6.887 4.604-4.33 0-7.859-3.579-7.859-8s3.53-8 7.859-8c2.46 0 4.105 1.025 5.047 1.926l2.427-2.334C18.155.955 15.42 0 12.24 0 5.58 0 0 5.37 0 12s5.58 12 12.24 12c6.96 0 11.57-4.89 11.57-11.79 0-.795-.085-1.4-.195-1.925H12.24z"/>
                </svg> Google
              </button>
              <button onClick={() => handleOAuthLogin('GitHub')} className="btn btn-secondary" style={{ justifyContent: 'center', fontSize: '0.8rem', gap: '6px' }}>
                <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                </svg> GitHub
              </button>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '20px', fontSize: '0.85rem' }}>
            <a 
              href="#" 
              onClick={(e) => { e.preventDefault(); setIsRegister(!isRegister); setError(''); }}
              style={{ color: 'var(--accent-emerald)', textDecoration: 'none' }}
            >
              {isRegister ? 'Already have an account? Sign In' : 'New operator? Register workspace'}
            </a>
          </div>
          
          {/* Quick Demo Info */}
          {!isRegister && (
            <div style={{ marginTop: '20px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              <span style={{ fontWeight: '700', color: 'var(--text-primary)', display: 'block', marginBottom: '4px' }}>Demo Sandbox Logins:</span>
              recycler / recycler123 (Operator)<br />
              sustainability / sustainability123 (Manager)<br />
              manufacturer / manufacturer123 (Manufacturer)
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-wrapper">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-logo">
          🌱 <span>Waste Intelligence</span>
        </div>

        {/* Sandbox Role Switcher Panel */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', padding: '4px 12px', borderRadius: '10px' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <UserCheck size={14} color="var(--accent-teal)" /> SANDBOX VIEW:
          </span>
          <select 
            className="form-select" 
            value={role} 
            onChange={(e) => handleSandboxRoleOverride(e.target.value)}
            style={{ padding: '4px 8px', fontSize: '0.75rem', background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontWeight: '700' }}
          >
            <option style={{ background: 'var(--bg-secondary)' }}>Recycling Facility Operator</option>
            <option style={{ background: 'var(--bg-secondary)' }}>Sustainability Manager</option>
            <option style={{ background: 'var(--bg-secondary)' }}>Textile Manufacturer</option>
            <option style={{ background: 'var(--bg-secondary)' }}>Administrator</option>
          </select>
        </div>

        <div className="nav-actions">
          <NotificationCenter />
          
          <div className="nav-user">
            <span style={{ fontSize: '0.85rem', fontWeight: '600' }}>@{user?.username || 'user'}</span>
            <span className="role-tag">{role.split(' ')[0]}</span>
          </div>

          <button onClick={handleLogout} className="btn btn-secondary" style={{ padding: '8px 12px' }} title="Log out">
            <LogOut size={16} />
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="dashboard-content">
        {renderDashboard()}
      </main>
    </div>
  );
}
