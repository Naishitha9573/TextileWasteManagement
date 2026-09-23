import React, { useState, useEffect, useRef } from 'react';
import { Bell, Check, AlertTriangle, CheckCircle, Info, Sparkles } from 'lucide-react';

export default function NotificationCenter() {
  const [notifications, setNotifications] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const fetchNotifications = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;
      const res = await fetch('/api/notifications', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setNotifications(data);
      }
    } catch (err) {
      console.error("Failed to fetch notifications:", err);
    }
  };

  useEffect(() => {
    fetchNotifications();
    // Poll notifications every 10 seconds for real-time feel
    const interval = setInterval(fetchNotifications, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const markAsRead = async (id) => {
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/api/notifications/${id}/read`, {
        method: 'PUT',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        setNotifications(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n));
      }
    } catch (err) {
      console.error(err);
    }
  };

  const getIcon = (type) => {
    switch (type) {
      case 'warning':
        return <AlertTriangle className="text-rose-500" style={{ color: 'var(--accent-rose)', minWidth: '18px' }} size={18} />;
      case 'milestone':
        return <Sparkles className="text-amber-500" style={{ color: 'var(--accent-amber)', minWidth: '18px' }} size={18} />;
      case 'collection':
        return <Info className="text-indigo-500" style={{ color: 'var(--accent-indigo)', minWidth: '18px' }} size={18} />;
      case 'opportunity':
        return <CheckCircle className="text-emerald-500" style={{ color: 'var(--accent-emerald)', minWidth: '18px' }} size={18} />;
      default:
        return <Info className="text-gray-400" size={18} />;
    }
  };

  const unreadCount = notifications.filter(n => !n.is_read).length;

  return (
    <div style={{ position: 'relative' }} ref={dropdownRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '50%',
          width: '40px',
          height: '40px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          position: 'relative',
          transition: 'var(--transition-smooth)'
        }}
        className="nav-btn-glow"
      >
        <Bell size={20} color="var(--text-primary)" />
        {unreadCount > 0 && (
          <span style={{
            position: 'absolute',
            top: '-2px',
            right: '-2px',
            background: 'var(--grad-danger)',
            color: 'white',
            borderRadius: '50%',
            fontSize: '10px',
            fontWeight: '700',
            width: '18px',
            height: '18px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            {unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="glass-panel" style={{
          position: 'absolute',
          top: '50px',
          right: 0,
          width: '360px',
          maxHeight: '450px',
          overflowY: 'auto',
          zIndex: 1000,
          padding: '16px',
          borderWidth: '1px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '8px' }}>
            <h4 style={{ margin: 0, fontFamily: 'var(--font-header)', fontSize: '1rem' }}>Alerts & Notifications</h4>
            {unreadCount > 0 && (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{unreadCount} unread</span>
            )}
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {notifications.length === 0 ? (
              <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                No notifications at this time.
              </div>
            ) : (
              notifications.map((notif) => (
                <div 
                  key={notif.id}
                  style={{
                    display: 'flex',
                    gap: '12px',
                    padding: '12px',
                    borderRadius: '8px',
                    background: notif.is_read ? 'rgba(255,255,255,0.01)' : 'rgba(255,255,255,0.03)',
                    borderLeft: notif.is_read ? '2px solid transparent' : '2px solid var(--accent-emerald)',
                    transition: 'var(--transition-smooth)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                    {getIcon(notif.type)}
                  </div>
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    <p style={{ margin: 0, fontSize: '0.85rem', color: notif.is_read ? 'var(--text-secondary)' : 'var(--text-primary)', lineHeight: '1.3' }}>
                      {notif.message}
                    </p>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      {new Date(notif.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  {!notif.is_read && (
                    <button 
                      onClick={() => markAsRead(notif.id)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--text-secondary)',
                        cursor: 'pointer',
                        padding: '2px',
                        alignSelf: 'flex-start'
                      }}
                      title="Mark as read"
                    >
                      <Check size={14} />
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
