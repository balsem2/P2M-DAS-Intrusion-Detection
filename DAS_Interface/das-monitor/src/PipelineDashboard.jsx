import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ShieldAlert, CheckCircle, AlertTriangle, Activity, Map, ArrowLeft } from 'lucide-react';

// --- PALETTE DE COULEURS ---
const theme = {
  bgGlobal: '#0b111e',     
  bgCard: '#131c2e',       
  border: '#1e293b',       
  textMain: '#f8fafc',     
  textMuted: '#94a3b8',    
  accentBlue: '#00d2ff',   
  accentOrange: '#ff9f00', 
  green: '#10b981',        
  orange: '#f59e0b',       
  red: '#ef4444',          
};

// =========================================================================
// TOPOLOGIE RÉELLE DE LA FIBRE
// =========================================================================
const FIBER_REAL_VERTICES = [
  { x: 740, y: 428 },
  { x: 970, y: 428 },
  { x: 970, y: 159 },
  { x: 192, y: 159 },
  { x: 180, y: 167 },
  { x: 58, y: 167 },
  { x: 58, y: 177 },
  { x: 87, y: 177 },
  { x: 87, y: 209 },
  { x: 99, y: 209 },
  { x: 99, y: 250 },
  { x: 126, y: 250 },
  { x: 126, y: 314 },
  { x: 198, y: 314 },
];

const generate166Segments = (vertices, totalPoints = 166) => {
  let segments = [];
  let totalLength = 0;

  for (let i = 0; i < vertices.length - 1; i++) {
    const dx = vertices[i+1].x - vertices[i].x;
    const dy = vertices[i+1].y - vertices[i].y;
    const len = Math.sqrt(dx*dx + dy*dy);
    segments.push({ from: vertices[i], to: vertices[i+1], length: len, dx, dy });
    totalLength += len;
  }

  const step = totalLength / (totalPoints - 1);
  const points = [];

  for (let i = 0; i < totalPoints; i++) {
    const targetDist = i * step;
    let currentDist = 0;
    let pt = { x: vertices[0].x, y: vertices[0].y };

    for (let seg of segments) {
      if (currentDist + seg.length >= targetDist || seg === segments[segments.length - 1]) {
        const ratio = seg.length === 0 ? 0 : (targetDist - currentDist) / seg.length;
        pt = {
          x: seg.from.x + seg.dx * ratio,
          y: seg.from.y + seg.dy * ratio
        };
        break;
      }
      currentDist += seg.length;
    }

    points.push({
      id: i + 1,
      name: `Segment ${i + 1} - Mètres ${i * 10} à ${(i + 1) * 10}`,
      status: 'green',
      eventName: 'regular',
      x: pt.x,
      y: pt.y
    });
  }
  return points;
};

const svgPathD = FIBER_REAL_VERTICES.map((v, idx) => 
  `${idx === 0 ? 'M' : 'L'} ${v.x} ${v.y}`
).join(' ');

export default function PipelineDashboard() {
  const [view, setView] = useState('auth'); 
  const [activeSegment, setActiveSegment] = useState(null);
  const [segments, setSegments] = useState([]);
  const [detailedData, setDetailedData] = useState(null);

  const pageStyle = {
    backgroundColor: theme.bgGlobal,
    minHeight: '100vh',
    color: theme.textMain,
    fontFamily: 'Segoe UI, Roboto, sans-serif',
    padding: '24px',
    boxSizing: 'border-box'
  };

  // 1. Rafraîchissement automatique de la carte (Vue Synoptique)
  useEffect(() => {
    // On initialise d'abord la structure si vide
    if (segments.length === 0) {
      setSegments(generate166Segments(FIBER_REAL_VERTICES, 166));
    }

    const fetchSegments = () => {
      fetch('http://localhost:8000/api/segments')
        .then(res => res.json())
        .then(data => {
          const updated = generate166Segments(FIBER_REAL_VERTICES, 166).map(seg => {
            const backend = data.find(d => d.id === seg.id);
            return backend ? { ...seg, status: backend.status, eventName: backend.event_name } : seg; 
          });
          setSegments(updated);
        })
        .catch(err => console.error("Erreur API segments:", err));
    };

    fetchSegments(); // Appel initial immédiat
    const interval = setInterval(fetchSegments, 1000); // Polling toutes les secondes
    return () => clearInterval(interval);
  }, []);

  // 2. Rafraîchissement automatique des détails d'un segment (Vue détaillée 10m + LSTM)
  useEffect(() => {
    if (view !== 'segment' || !activeSegment?.id) return;

    const fetchDetails = () => {
      fetch(`http://localhost:8000/api/segment/${activeSegment.id}`)
        .then(res => res.json())
        .then(data => setDetailedData(data))
        .catch(err => console.error("Erreur API détails:", err));
    };

    fetchDetails();
    const interval = setInterval(fetchDetails, 1000);
    return () => clearInterval(interval);
  }, [view, activeSegment?.id]);

  // Synchronisation dynamique de l'état live du segment sélectionné
  const liveSegment = segments.find(s => s.id === activeSegment?.id) || activeSegment;

  // ====================== VUE AUTH ======================
  if (view === 'auth') {
    return (
      <div style={{ ...pageStyle, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ backgroundColor: theme.bgCard, padding: '40px', borderRadius: '12px', width: '360px', border: `1px solid ${theme.border}`, textAlign: 'center', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}>
          <Activity size={48} color={theme.accentBlue} style={{ marginBottom: '16px' }} />
          <h1 style={{ margin: '0 0 24px 0', fontSize: '24px', fontWeight: 'bold' }}>DAS Pipeline Monitor</h1>
          <input type="text" placeholder="Identifiant" style={{ width: '100%', padding: '12px', marginBottom: '16px', borderRadius: '6px', backgroundColor: theme.bgGlobal, border: `1px solid ${theme.border}`, color: '#fff', boxSizing: 'border-box' }} />
          <input type="password" placeholder="Mot de passe" style={{ width: '100%', padding: '12px', marginBottom: '24px', borderRadius: '6px', backgroundColor: theme.bgGlobal, border: `1px solid ${theme.border}`, color: '#fff', boxSizing: 'border-box' }} />
          <button onClick={() => setView('map')} style={{ width: '100%', padding: '12px', backgroundColor: theme.accentBlue, color: '#000', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', fontSize: '16px' }}>
            Se Connecter
          </button>
        </div>
      </div>
    );
  }

  // ====================== VUE CARTE ======================
  if (view === 'map') {
    const displaySegments = segments.length > 0 ? segments : generate166Segments(FIBER_REAL_VERTICES, 166);

    return (
      <div style={pageStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: theme.bgCard, padding: '16px 24px', borderRadius: '8px', marginBottom: '24px', border: `1px solid ${theme.border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Map color={theme.accentBlue} />
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold' }}>Topologie Réelle de la Fibre DAS (1663 mètres)</h1>
          </div>
          <span style={{ fontSize: '13px', backgroundColor: '#1e293b', padding: '6px 12px', borderRadius: '20px', color: theme.accentBlue }}>
            🟢 {displaySegments.filter(s => s.status === 'green').length} OK | 
            🟡 {displaySegments.filter(s => s.status === 'orange').length} Activités | 
            🔴 {displaySegments.filter(s => s.status === 'red').length} Alertes Crêtes
          </span>
        </div>

        <div style={{ backgroundColor: theme.bgCard, borderRadius: '12px', padding: '20px', border: `1px solid ${theme.border}` }}>
          <p style={{ color: theme.textMuted, margin: '0 0 16px 0', fontSize: '14px' }}>
            Résolution spatiale : <strong>10 mètres par point de mesure</strong>. Cliquez sur un nœud pour ouvrir l'analyse prescriptive.
          </p>
          
          <div style={{ width: '100%', overflow: 'hidden', backgroundColor: '#090d16', borderRadius: '8px', padding: '10px' }}>
            <svg viewBox="0 0 1000 600" width="100%" height="100%" style={{ display: 'block' }}>
              <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              <path d={svgPathD} stroke={theme.accentBlue} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)" />
              <path d={svgPathD} stroke={theme.accentBlue} strokeWidth="12" fill="none" opacity="0.08" strokeLinecap="round" strokeLinejoin="round" />

              {displaySegments.map((seg) => {
                const isAlert = seg.status !== 'green';
                const color = seg.status === 'green' ? theme.green : seg.status === 'orange' ? theme.orange : theme.red;
                const radius = isAlert ? 7 : 4;

                return (
                  <g key={seg.id} onClick={() => { setActiveSegment(seg); setView('segment'); }} style={{ cursor: 'pointer' }}>
                    {isAlert && (
                      <circle cx={seg.x} cy={seg.y} r={16} fill={color} opacity="0.3">
                        <animate attributeName="r" values="10;22;10" dur="1.2s" repeatCount="indefinite" />
                        <animate attributeName="opacity" values="0.6;0;0.6" dur="1.2s" repeatCount="indefinite" />
                      </circle>
                    )}
                    <circle 
                      cx={seg.x} 
                      cy={seg.y} 
                      r={radius} 
                      fill={color} 
                      stroke={isAlert ? '#fff' : 'none'} 
                      strokeWidth={isAlert ? 1.5 : 0}
                    />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      </div>
    );
  }

  // ====================== VUE SEGMENT ======================
  if (view === 'segment' && liveSegment) {
    const statusColor = liveSegment.status === 'green' ? theme.green : 
                        liveSegment.status === 'orange' ? theme.orange : theme.red;

    return (
      <div style={pageStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: theme.bgCard, padding: '16px 24px', borderRadius: '8px', marginBottom: '24px', border: `1px solid ${theme.border}` }}>
          <button 
            onClick={() => { setView('map'); setDetailedData(null); }} 
            style={{ background: 'none', border: 'none', color: theme.accentBlue, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}
          >
            <ArrowLeft size={18} /> Retour au synoptique cartographique
          </button>
          <h2 style={{ margin: 0, fontSize: '18px' }}>{liveSegment.name}</h2>
        </div>

        {/* Bloc d'Événement CNN mis à jour dynamiquement */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', backgroundColor: theme.bgCard, padding: '24px', borderRadius: '12px', marginBottom: '24px', border: `1px solid ${statusColor}`, boxShadow: `inset 0 0 15px ${statusColor}20` }}>
          {liveSegment.status === 'red' && <ShieldAlert size={44} color={theme.red} />}
          {liveSegment.status === 'orange' && <AlertTriangle size={44} color={theme.orange} />}
          {liveSegment.status === 'green' && <CheckCircle size={44} color={theme.green} />}
          
          <div>
            <h3 style={{ margin: '0 0 6px 0', fontSize: '22px', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Classification Événementielle (CNN 2D) : <span style={{ color: statusColor }}>{liveSegment.eventName || 'Inconnu'}</span>
            </h3>
            <p style={{ margin: 0, color: theme.textMuted, fontSize: '14px' }}>
              Niveau de sécurité : <span style={{ color: statusColor, fontWeight: 'bold' }}>
                {liveSegment.status === 'red' ? '⚠️ CRITIQUE (Menace infrastructure détectée)' : 
                 liveSegment.status === 'orange' ? '⚠️ EN COURS D\'ANALYSE (Événement anonyme/suspect)' : 
                 '✅ SÉCURISÉ (Aucun signal d\'intrusion)'}
              </span>
            </p>
          </div>
        </div>

        {/* Section Graphiques Glissants Réel vs LSTM */}
        <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity color={theme.accentBlue} size={20} /> Spectrogramme Spatial - Signal Brut .H5 vs Horizon Temporel LSTM
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
          {Array.from({ length: 10 }).map((_, idx) => {
            // Lecture des clés exactes générées par notre API fastapi ("meter_1" à "meter_10")
            const chartData = detailedData?.charts?.[`meter_${idx + 1}`] || [];

            return (
              <div key={idx} style={{ backgroundColor: theme.bgCard, padding: '16px', borderRadius: '8px', border: `1px solid ${theme.border}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span style={{ fontSize: '13px', fontWeight: 'bold', color: theme.textMuted }}>
                    Sous-canal : Mètre {idx + 1}
                  </span>
                  <span style={{ fontSize: '11px', color: theme.accentBlue, fontWeight: '500' }}>Flux continu</span>
                </div>

                <div style={{ height: '140px' }}>
                  {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="time" hide />
                        <YAxis hide domain={['auto', 'auto']} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', color: '#fff', fontSize: '11px' }} />
                        <Line type="monotone" dataKey="reel" stroke={theme.accentBlue} strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="predit" stroke={theme.accentOrange} strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', fontSize: '12px', color: theme.textMuted, fontStyle: 'italic' }}>
                      Synchro signaux en cours...
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  return null;
}