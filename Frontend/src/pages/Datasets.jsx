import React, { useEffect, useState } from 'react';
import { Image } from 'lucide-react';
export default function Datasets(){
  const [datasets, setDatasets] = useState([]);
  useEffect(()=>{fetchDatasets()},[]);
  const fetchDatasets = async ()=>{
    try{
      const token = localStorage.getItem('token');
      const res = await fetch('/api/datasets',{ headers: {'Authorization': `Bearer ${token}`} });
      if(res.ok){ setDatasets(await res.json()); }
    }catch(err){ console.error(err); }
  }
  return (
    <div>
      <h2 style={{fontFamily:'var(--font-header)'}}>Datasets</h2>
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
        {datasets.map(ds=> (
          <div className="glass-panel" key={ds.name} style={{padding:12}}>
            <div style={{display:'flex',justifyContent:'space-between'}}>
              <div>
                <h4 style={{margin:0}}>{ds.name}</h4>
                <div style={{fontSize:'0.85rem',color:'var(--text-secondary)'}}>Categories: {ds.categories?.length || 0}</div>
                <div style={{fontSize:'0.85rem',color:'var(--text-secondary)'}}>Images: {ds.images || 0}</div>
                <div style={{fontSize:'0.85rem',color:'var(--text-secondary)'}}>Source: {ds.source_type || 'unknown'}</div>
                {ds.categories?.length > 0 && (
                  <div style={{marginTop:8,fontSize:'0.8rem',color:'var(--text-secondary)'}}>
                    {ds.categories.slice(0,6).join(', ')}{ds.categories.length > 6 ? ' …' : ''}
                  </div>
                )}
              </div>
              <div style={{alignSelf:'center'}}>
                <Image />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
