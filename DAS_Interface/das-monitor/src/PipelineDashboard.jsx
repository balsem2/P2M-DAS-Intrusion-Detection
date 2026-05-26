import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ShieldAlert, CheckCircle, AlertTriangle, Activity, Map, ArrowLeft, LogOut } from 'lucide-react';

// --- PALETTE DE COULEURS INDUSTRIAL SCADA ---
const theme = {
  bgGlobal: '#0b111e',     
  bgCard: '#131c2e',       
  border: '#1e293b',       
  textMain: '#f8fafc',     
  textMuted: '#94a3b8',    
  accentBlue: '#00d2ff',   // Bleu ciel (Signal réel)
  accentOrange: '#ff9f00', // Orange néon (Prédiction LSTM)
  green: '#10b981',        
  orange: '#f59e0b',       
  red: '#ef4444',          
};

// =========================================================================
// 1. TOPOLOGIE RÉELLE DE LA FIBRE (Inspirée de la ligne verte de la Fig 1)
// =========================================================================
// Modifiez ces coordonnées (X entre 50 et 950, Y entre 50 et 550) pour ajuster les virages de la carte
const FIBER_REAL_VERTICES = [
  { x: 740, y: 428 }, // Départ bas-droite
  { x: 970, y: 428 },   // Vers coin bas-droite
  { x: 970, y: 159 },   // Montée verticale côté droit
  { x: 192, y: 159 },   // Ligne haute vers la gauche
  { x: 180, y: 167 },   
  { x: 58, y: 167 },
  { x: 58, y: 177 },
  { x: 87, y: 177 },  // Extrémité gauche haute
  { x: 87, y: 209 },
  { x: 99, y: 209 },
  { x: 99, y: 250 },
  { x: 126, y: 250 },
  { x: 126, y: 314 },
  { x: 198, y: 314 },   // Arrivée vers le DAS
     // Retour au départ (fermeture)
];

// =========================================================================
// 2. ALGORITHME DE GÉNÉRATION AUTOMATIQUE DES 166 POINTS (1 point = 10m)
// =========================================================================
const generate166Segments = (vertices, totalPoints = 166) => {
  let segments = [];
  let totalLength = 0;

  // Calcul des distances de chaque tronçon
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

    // Simulation d'alertes à des endroits précis de la fibre réelle :
    let status = 'green';
    if (i === 35) status = 'orange';  // Alerte travaux au mètre 350 (Segment 36)
    if (i === 112) status = 'red';    // Alerte fuite critique au mètre 1120 (Segment 113)

    points.push({
      id: i + 1,
      name: `Segment ${i + 1} - Mètres ${i * 10} à ${(i + 1) * 10}`,
      status,
      x: pt.x,
      y: pt.y
    });
  }
  return points;
};

const PIPELINE_SEGMENTS = generate166Segments(FIBER_REAL_VERTICES, 166);

// Génération de la chaîne de caractères du chemin SVG pour la ligne continue bleu ciel
const svgPathD = FIBER_REAL_VERTICES.map((v, idx) => `${idx === 0 ? 'M' : 'L'} ${v.x} ${v.y}`).join(' ');

const generateChartData = () => {
  return Array.from({ length: 20 }, (_, i) => ({
    time: i,
    reel: Math.random() * 0.4 + 0.3,
    predit: Math.random() * 0.4 + 0.3 + (Math.random() * 0.06 - 0.03),
  }));
};

export default function PipelineDashboard() {
  const [view, setView] = useState('auth'); 
  const [activeSegment, setActiveSegment] = useState(null);

  const pageStyle = {
    backgroundColor: theme.bgGlobal,
    minHeight: '100vh',
    color: theme.textMain,
    fontFamily: 'Segoe UI, Roboto, sans-serif',
    padding: '24px',
    boxSizing: 'border-box'
  };

  // ==========================================
  // VUE 1 : AUTHENTIFICATION
  // ==========================================
  if (view === 'auth') {
    return (
      <div style={{ ...pageStyle, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ backgroundColor: theme.bgCard, padding: '40px', borderRadius: '12px', width: '360px', border: `1px solid ${theme.border}`, textAlign: 'center', boxShadow: '0 10px 25px rgba(0,0,0,0.5)' }}>
          <Activity size={48} color={theme.accentBlue} style={{ marginBottom: '16px' }} />
          <h1 style={{ margin: '0 0 24px 0', fontSize: '24px', fontWeight: 'bold' }}>DAS Pipeline Monitor</h1>
          <input type="text" placeholder="Identifiant" style={{ width: '100%', padding: '12px', marginBottom: '16px', borderRadius: '6px', backgroundColor: theme.bgGlobal, border: `1px solid ${theme.border}`, color: '#fff', boxSizing: 'border-box' }} />
          <input type="password" placeholder="Mot de passe" style={{ width: '100%', padding: '12px', marginBottom: '24px', borderRadius: '6px', backgroundColor: theme.bgGlobal, border: `1px solid ${theme.border}`, color: '#fff', boxSizing: 'border-box' }} />
          <button onClick={() => setView('map')} style={{ width: '100%', padding: '12px', backgroundColor: theme.accentBlue, color: '#000', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', fontSize: '16px' }}>Se Connecter</button>
        </div>
      </div>
    );
  }

  // ==========================================
  // VUE 2 : SYNOPTIQUE RÉEL (166 POINTS SUR TRACÉ)
  // ==========================================
  if (view === 'map') {
    return (
      <div style={pageStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: theme.bgCard, padding: '16px 24px', borderRadius: '8px', marginBottom: '24px', border: `1px solid ${theme.border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Map color={theme.accentBlue} />
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold' }}>Topologie Réelle de la Fibre DAS (1663 mètres)</h1>
          </div>
          <span style={{ fontSize: '13px', backgroundColor: '#1e293b', padding: '6px 12px', borderRadius: '20px', color: theme.accentBlue }}>
            🟢 164 Normaux | 🟡 1 Critique | 🔴 1 Danger
          </span>
        </div>

        <div style={{ backgroundColor: theme.bgCard, borderRadius: '12px', padding: '1px', border: `1px solid ${theme.border}` }}>
          <p style={{ color: theme.textMuted, margin: '0 0 0 0', fontSize: '14px' }}>
            Résolution spatiale : **10 mètres par point**. Survolez ou cliquez sur un capteur pour inspecter ses 10 diagrammes.
          </p>
          
          <div style={{ width: '100%', overflow: 'hidden', backgroundColor: '#090d16', borderRadius: '8px', padding: '10px' }}>
            <svg viewBox="0 0 1000 600" width="100%" height="100%" style={{ display: 'block' }}>
              <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* LIGNE BLEU CIEL DE LA FIBRE RÉELLE */}
              <path d={svgPathD} stroke={theme.accentBlue} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow)" />
              <path d={svgPathD} stroke={theme.accentBlue} strokeWidth="12" fill="none" opacity="0.08" strokeLinecap="round" strokeLinejoin="round" />

              {/* RENDU DES 166 POINTS INTERACTIFS ACCROCHÉS AU TRACÉ */}
              {PIPELINE_SEGMENTS.map((seg) => {
                const isAlert = seg.status !== 'green';
                const color = seg.status === 'green' ? theme.green : seg.status === 'orange' ? theme.orange : theme.red;
                const radius = isAlert ? 7 : 4; // Plus gros si alerte pour rester visible

                return (
                  <g key={seg.id} onClick={() => { setActiveSegment(seg); setView('segment'); }} style={{ cursor: 'pointer' }}>
                    {isAlert && (
                      <circle cx={seg.x} cy={seg.y} r={16} fill={color} opacity="0.3">
                        <animate attributeName="r" values="10;20;10" dur="1.5s" repeatCount="indefinite" />
                        <animate attributeName="opacity" values="0.5;0;0.5" dur="1.5s" repeatCount="indefinite" />
                      </circle>
                    )}
                    <circle 
                      cx={seg.x} 
                      cy={seg.y} 
                      r={radius} 
                      fill={color} 
                      stroke={isAlert ? '#fff' : 'none'} 
                      strokeWidth={isAlert ? 1.5 : 0}
                      style={{ transition: 'r 0.2s' }}
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

  // ==========================================
  // VUE 3 : INTERFACE DU SEGMENT SÉLECTIONNÉ (10 MÈTRES)
  // ==========================================
  if (view === 'segment' && activeSegment) {
    const statusColor = activeSegment.status === 'green' ? theme.green : activeSegment.status === 'orange' ? theme.orange : theme.red;
    
    return (
      <div style={pageStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: theme.bgCard, padding: '16px 24px', borderRadius: '8px', marginBottom: '24px', border: `1px solid ${theme.border}` }}>
          <button onClick={() => setView('map')} style={{ background: 'none', border: 'none', color: theme.accentBlue, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
            <ArrowLeft size={18} /> Retour à la carte réelle
          </button>
          <h2 style={{ margin: 0, fontSize: '18px' }}>{activeSegment.name}</h2>
        </div>

        {/* EVENEMENT PROBABLE (CNN) & NIVEAU DE DANGER */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', backgroundColor: theme.bgCard, padding: '24px', borderRadius: '12px', marginBottom: '24px', border: `1px solid ${statusColor}`, boxShadow: `inset 0 0 15px ${statusColor}20` }}>
          {activeSegment.status === 'red' && <ShieldAlert size={44} color={theme.red} />}
          {activeSegment.status === 'orange' && <AlertTriangle size={44} color={theme.orange} />}
          {activeSegment.status === 'green' && <CheckCircle size={44} color={theme.green} />}
          
          <div>
            <h3 style={{ margin: '0 0 6px 0', fontSize: '20px', fontWeight: 'bold' }}>
              Événement Classifié (CNN) : {
                activeSegment.status === 'red' ? '🚨 CRITICAL : FUITE PROCHE DE LA CONDUITE' : 
                activeSegment.status === 'orange' ? '⚠️ ALERTE : EXCAVATION / MARCHE / ACTIVITÉ INTRUSIVE' : 
                '✅ STATUT NORMAL : UNIQUEMENT DU BRUIT DE FOND ACOUSTIQUE'
              }
            </h3>
            <p style={{ margin: 0, color: theme.textMuted, fontSize: '14px' }}>
              Criticité de la zone : <span style={{ color: statusColor, fontWeight: 'bold' }}>{activeSegment.status.toUpperCase()}</span>
            </p>
          </div>
        </div>

        {/* LES 10 DIAGRAMMES DE NIVEAU DU SIGNAL (LSTM) */}
        <h3 style={{ margin: '0 0 16px 0', fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity color={theme.accentBlue} size={20} /> Signal Instantané vs Prédiction Temporelle (LSTM sur 10 sous-sections d'un mètre)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
          {Array.from({ length: 10 }).map((_, idx) => (
            <div key={idx} style={{ backgroundColor: theme.bgCard, padding: '16px', borderRadius: '8px', border: `1px solid ${theme.border}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span style={{ fontSize: '13px', fontWeight: 'bold', color: theme.textMuted }}>Mètre {idx + 1}</span>
                <span style={{ fontSize: '11px', color: theme.accentBlue }}>Live & Horizon</span>
              </div>

              <div style={{ height: '140px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={generateChartData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="time" hide />
                    <YAxis hide domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', color: '#fff', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="reel" stroke={theme.accentBlue} strokeWidth={2} dot={false} name="Signal Actuel" />
                    <Line type="monotone" dataKey="predit" stroke={theme.accentOrange} strokeWidth={2} strokeDasharray="4 4" dot={false} name="Prédiction LSTM" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
}