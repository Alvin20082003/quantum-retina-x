// DigitalTwin.tsx - Interactive Eye Digital Twin - driven by real API results
import { useState, useEffect, useRef } from "react";

interface LesionInfo { detected: boolean; confidence: number; color: string; }
interface RealResult {
  grade: number; grade_label: string; confidence: number;
  tcav_scores: Record<string,number>;
  lesion_detections?: Record<string,LesionInfo>;
  dme_detected?: boolean; dme_confidence?: number;
  recommendation: string; risk_level: string;
}

interface Finding { id:string; label:string; color:string; glow:string; cx:number; cy:number; r:number; severity:string; desc:string; }
interface Layer  { id:string; label:string; depth:number; borderColor:string; desc:string; clinicalNote:string; findings:Finding[]; }

const SEVERITY_COLOR: Record<string,string> = { none:"#22c55e", low:"#84cc16", moderate:"#f59e0b", high:"#f97316", critical:"#ef4444" };
const SEVERITY_LABEL: Record<string,string> = { none:"Clear", low:"Low Risk", moderate:"Moderate", high:"High Risk", critical:"Critical" };
const LAYER_COLORS = ["#60a5fa","#818cf8","#38bdf8","#a855f7","#f59e0b","#f97316"];
const GRADE_COLORS = ["#22c55e","#f59e0b","#f97316","#ef4444","#dc2626"];

function buildLayers(result: RealResult | null): Layer[] {
  const grade = result?.grade ?? 2;
  const tcav = result?.tcav_scores ?? { microaneurysm:0.61, hemorrhage:0.74, exudate:0.52, edema:0.38 };
  const dme = result?.dme_detected ?? false;

  // Helper: score → severity
  const sev = (s:number): string => s>0.75?"high": s>0.5?"moderate": s>0.25?"low":"none";

  // Build retina findings from TCAV scores
  const retinaFindings: Finding[] = [];
  if ((tcav.microaneurysm??0) > 0.1) retinaFindings.push(
    { id:"ma1", label:"Microaneurysms", color:"#ef4444", glow:"#ef444430", cx:38, cy:42, r:5, severity:sev(tcav.microaneurysm??0), desc:`Score ${((tcav.microaneurysm??0)*100).toFixed(0)}% - Tiny balloon-like bulges in retinal vessel walls. Earliest sign of DR.` })
  if ((tcav.hemorrhage??0) > 0.1) retinaFindings.push(
    { id:"hem1", label:"Hemorrhages", color:"#dc2626", glow:"#dc262630", cx:52, cy:60, r:8, severity:sev(tcav.hemorrhage??0), desc:`Score ${((tcav.hemorrhage??0)*100).toFixed(0)}% - Blood leaking from damaged retinal vessels. Indicates vessel wall breakdown.` })
  if ((tcav.exudate??0) > 0.1) retinaFindings.push(
    { id:"ex1", label:"Hard Exudates", color:"#fbbf24", glow:"#fbbf2430", cx:62, cy:40, r:6, severity:sev(tcav.exudate??0), desc:`Score ${((tcav.exudate??0)*100).toFixed(0)}% - Fatty lipid deposits from leaking blood vessels. Yellow-white patches on retina.` })

  // Macula findings from DME + edema TCAV
  const maculaFindings: Finding[] = [];
  const edemaScore = tcav.edema ?? 0;
  if (dme || edemaScore > 0.3) maculaFindings.push(
    { id:"dme1", label:"Macular Oedema", color:"#3b82f6", glow:"#3b82f630", cx:50, cy:50, r:16, severity:dme?"high":"moderate", desc:`${dme?"DETECTED - ":"Risk elevated - "}Fluid accumulation in macula. Causes blurry central vision. Edema score: ${(edemaScore*100).toFixed(0)}%.` })
  if (grade >= 2 && (tcav.exudate??0) > 0.4) maculaFindings.push(
    { id:"ex2", label:"Circinate Exudates", color:"#fbbf24", glow:"#fbbf2430", cx:50, cy:50, r:22, severity:"moderate", desc:"Ring of hard exudates around the fovea. High risk to central sharp vision." })

  // Grade-based clinical notes
  const gradeNotes = [
    "No diabetic retinopathy detected. Retina appears healthy.",
    "Mild NPDR - a few microaneurysms detected. Early stage, monitor closely.",
    "Moderate NPDR - multiple lesion types present. Treatment planning recommended.",
    "Severe NPDR - extensive damage. Immediate ophthalmologist referral required.",
    "Proliferative DR - new vessel growth detected. Urgent specialist referral.",
  ];

  return [
    { id:"cornea",   label:"Cornea",      depth:0, borderColor:"#60a5fa",
      desc:"Transparent outer surface of the eye. Clear in most DR patients.",
      clinicalNote:"No DR lesions at corneal level. Surface integrity normal.",
      findings:[] },
    { id:"lens",     label:"Lens",        depth:1, borderColor:"#818cf8",
      desc:"Crystalline lens focuses light onto the retina. Chronic hyperglycaemia can accelerate cataract.",
      clinicalNote: grade >= 2 ? "Mild lens opacity possible with chronic hyperglycaemia. Monitor." : "Lens appears clear.",
      findings: grade >= 3 ? [{ id:"opacity", label:"Lens Opacity", color:"#818cf8", glow:"#818cf820", cx:50, cy:48, r:14, severity:"low", desc:"Early cataract formation linked to diabetes duration." }] : [] },
    { id:"vitreous", label:"Vitreous",    depth:2, borderColor:"#38bdf8",
      desc:"Gel-filled chamber. In proliferative DR, new vessels can bleed into vitreous.",
      clinicalNote: grade >= 4 ? "VITREOUS HAEMORRHAGE RISK - new vessel growth detected." : "Vitreous clear. No haemorrhage at this stage.",
      findings: grade >= 4 ? [{ id:"vh", label:"NVE Risk", color:"#f97316", glow:"#f9731630", cx:50, cy:50, r:12, severity:"critical", desc:"New vessel proliferation (NVE) may bleed into vitreous - Grade 4 risk." }] : [] },
    { id:"retina",   label:"Retina",      depth:3, borderColor:"#a855f7",
      desc:"Light-sensitive neural tissue. Primary site of all DR lesions.",
      clinicalNote: gradeNotes[grade],
      findings: retinaFindings },
    { id:"macula",   label:"Macula",      depth:4, borderColor:"#f59e0b",
      desc:"Central 5mm responsible for sharp vision and colour. DME occurs here.",
      clinicalNote: dme ? "⚠ DME DETECTED - macular oedema present. Vision-threatening." : edemaScore > 0.3 ? "Elevated DME risk. Oedema score above threshold." : "Macula appears normal. No oedema detected.",
      findings: maculaFindings },
    { id:"optic",    label:"Optic Disc",  depth:5, borderColor:"#f97316",
      desc:"Exit point of the optic nerve. In PDR, new vessel growth (NVD) appears here.",
      clinicalNote: grade >= 4 ? "⚠ NVD RISK - neovascularisation at disc possible." : "No NVD detected. Cup-to-disc ratio normal.",
      findings: grade >= 4 ? [{ id:"nvd", label:"NVD", color:"#ef4444", glow:"#ef444430", cx:50, cy:50, r:16, severity:"critical", desc:"New vessels at disc - hallmark of Proliferative DR." }] : [{ id:"disc", label:"Optic Disc", color:"#f97316", glow:"#f9731620", cx:50, cy:50, r:18, severity:"none", desc:"Disc margin sharp. No NVD." }] },
  ];
}

const CSS = `
  @keyframes dtPulse   { 0%,100%{opacity:0.5;transform:scale(1)} 50%{opacity:1;transform:scale(1.1)} }
  @keyframes dtFloat   { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
  @keyframes dtSlideIn { from{opacity:0;transform:translateX(16px)} to{opacity:1;transform:none} }
  @keyframes dtScan    { 0%{transform:translateY(-120%);opacity:0.8} 100%{transform:translateY(220%);opacity:0} }
  @keyframes dtReveal  { from{opacity:0;transform:scale(0.6)} to{opacity:1;transform:scale(1)} }
  .dt-layer:hover { transform: translateX(4px); transition: transform 0.15s; }
`;

export default function DigitalTwin({ result }: { result: RealResult | null }) {
  const layers = buildLayers(result);
  const [activeLayer,  setActiveLayer]  = useState(result ? (result.grade >= 4 ? 5 : result.grade >= 2 ? 3 : 3) : 3);
  const [activeFinding,setActiveFinding] = useState<Finding|null>(null);
  const [scanning,     setScanning]      = useState(false);
  const [revealed,     setRevealed]      = useState<Set<number>>(new Set([0,1,2,3,4,5]));
  const scanRef = useRef<number>(0);
  const layer   = layers[activeLayer];
  const grade   = result?.grade ?? 2;
  const gradeColor = GRADE_COLORS[grade] ?? "#f59e0b";

  function runScan() {
    setScanning(true); setRevealed(new Set()); setActiveFinding(null);
    let i = 0;
    const tick = () => {
      setRevealed(p => { const n=new Set(p); n.add(i); return n; });
      setActiveLayer(i); i++;
      if (i < layers.length) scanRef.current = window.setTimeout(tick, 550);
      else { setScanning(false); setActiveLayer(grade >= 4 ? 5 : 3); }
    };
    scanRef.current = window.setTimeout(tick, 150);
  }
  useEffect(() => () => clearTimeout(scanRef.current), []);

  function EyeSVG() {
    const radii = [96, 80, 64, 48, 28, 14];
    return (
      <svg viewBox="0 0 200 200" style={{width:"100%",height:"100%",overflow:"visible"}}>
        <defs>
          <radialGradient id="dtEyeBg" cx="50%" cy="45%" r="55%"><stop offset="0%" stopColor="#2d1060"/><stop offset="70%" stopColor="#1a0533"/><stop offset="100%" stopColor="#07070f"/></radialGradient>
          <radialGradient id="dtIris"  cx="38%" cy="35%" r="58%"><stop offset="0%" stopColor="#ede9fe"/><stop offset="30%" stopColor="#c4b5fd"/><stop offset="65%" stopColor="#7c3aed"/><stop offset="100%" stopColor="#3b0764"/></radialGradient>
          <filter id="dtGlow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        {radii.map((r,i) => {
          const active = activeLayer===i, rev = revealed.has(i);
          return <g key={i} onClick={()=>{setActiveLayer(i);setActiveFinding(null);}} style={{cursor:"pointer"}}>
            <circle cx="100" cy="100" r={r} fill={i===3?"url(#dtIris)":i===0?"url(#dtEyeBg)":"none"} stroke={rev?LAYER_COLORS[i]:"#ffffff08"} strokeWidth={active?2.5:1} opacity={rev?1:0.15} style={{transition:"all 0.4s",filter:active?"url(#dtGlow)":"none"}}/>
            {active&&<circle cx="100" cy="100" r={r} fill="none" stroke={LAYER_COLORS[i]} strokeWidth={1} opacity={0.4} style={{animation:"dtPulse 2s ease-in-out infinite"}}/>}
          </g>;
        })}
        {revealed.has(3)&&[0,30,60,90,120,150,180,210,240,270,300,330].map(d=>(
          <line key={d} x1="100" y1="100" x2={100+46*Math.cos(d*Math.PI/180)} y2={100+46*Math.sin(d*Math.PI/180)} stroke="#a855f718" strokeWidth={0.5}/>
        ))}
        {revealed.has(3)&&<><circle cx="100" cy="100" r={11} fill="#000" opacity={0.92}/><circle cx="94" cy="94" r={3} fill="#ffffff55"/><circle cx="97" cy="97" r={1.2} fill="#ffffff35"/></>}
        {layer.findings.map(f=>{
          const sx=f.cx*2, sy=f.cy*2, sr=Math.max(f.r*1.6,5);
          const isAct=activeFinding?.id===f.id;
          return <g key={f.id} onClick={e=>{e.stopPropagation();setActiveFinding(isAct?null:f);}} style={{cursor:"pointer"}}>
            <circle cx={sx} cy={sy} r={sr+4} fill={f.glow}/>
            <circle cx={sx} cy={sy} r={sr+(isAct?3:0)} fill={f.color} opacity={0.85} filter="url(#dtGlow)" style={{animation:`dtPulse ${1.4+f.r/8}s ease-in-out infinite`,transition:"r 0.2s"}}/>
            {isAct&&<circle cx={sx} cy={sy} r={sr+8} fill="none" stroke={f.color} strokeWidth={1} strokeDasharray="4,3" opacity={0.7}/>}
          </g>;
        })}
        {scanning&&<line x1="4" y1="100" x2="196" y2="100" stroke="#a855f7" strokeWidth={1.5} opacity={0.7} style={{animation:"dtScan 0.55s linear infinite"}}/>}
        {radii.map((r,i)=>{
          if(!revealed.has(i)) return null;
          const ang=-55+i*20, lx=100+(r+11)*Math.cos(ang*Math.PI/180), ly=100+(r+11)*Math.sin(ang*Math.PI/180);
          return <text key={i} x={lx} y={ly} fill={LAYER_COLORS[i]} fontSize={5} textAnchor="middle" opacity={0.75}>{layers[i].label}</text>;
        })}
      </svg>
    );
  }

  const PAT_EXPLAIN = [
    "Your cornea - the clear front window of your eye - is unaffected by diabetes.",
    grade>=2 ? "High blood sugar over time can make the lens slightly cloudy. This is early cataract formation and is treatable." : "Your lens looks clear. No diabetes-related changes here yet.",
    grade>=4 ? "In advanced diabetes, fragile new vessels can bleed into this gel-filled space causing sudden dark spots or vision loss." : "The gel inside your eye is clear - good news. We will keep monitoring this.",
    grade===0 ? "Your retina looks healthy - no diabetes damage detected." : grade===1 ? "There are a few tiny red dots (microaneurysms) on your retina. These are the first sign of diabetes affecting your eye." : grade===2 ? "Your retina has moderate damage - red dots, some bleeding, and yellow deposits from leaking vessels. This needs treatment to stop getting worse." : grade===3 ? "Your retina has extensive damage. Many blood vessels are leaking. Immediate referral to a specialist is needed." : "New abnormal vessels are growing on your retina. This is the most severe stage - urgent treatment can save your sight.",
    result?.dme_detected ? "The centre of your vision - the macula - has fluid swelling. This blurs your central vision for reading and faces. Injections or laser can treat this effectively." : (result?.tcav_scores?.edema??0)>0.3 ? "There are early signs of fluid building near your macula. Catching this now means treatment will be more effective." : "Your macular area looks stable. No fluid swelling detected.",
    grade>=4 ? "Abnormal new vessels may be growing at your optic nerve. This is very serious - urgent laser treatment is needed." : "Your optic nerve - the cable connecting your eye to your brain - appears healthy.",
  ];

  return (
    <div style={{padding:"24px 28px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:18}}>
      <style>{CSS}</style>

      {/* Header */}
      <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",flexWrap:"wrap",gap:14}}>
        <div>
          <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>PHASE 2 · DIGITAL TWIN</div>
          <h2 style={{fontSize:26,fontWeight:900,color:"#fff",letterSpacing:-0.5,margin:0}}>Eye Digital Twin</h2>
          <p style={{fontSize:13,color:"#9b89c4",marginTop:4,maxWidth:520}}>
            {result ? `Showing real findings from your scan - Grade ${result.grade} ${result.grade_label}` : "Upload an image to see real AI findings - showing demo data until then."}</p>
        </div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          {!result && <div style={{fontSize:11,padding:"5px 12px",borderRadius:999,background:"#f59e0b20",color:"#f59e0b",border:"1px solid #f59e0b30",fontWeight:600}}>⚠ Demo Mode</div>}
          {result  && <div style={{fontSize:11,padding:"5px 12px",borderRadius:999,background:"#22c55e20",color:"#22c55e",border:"1px solid #22c55e30",fontWeight:600}}>✓ Live Scan Data</div>}
          <button onClick={runScan} disabled={scanning}
            style={{background:scanning?"#1a1040":"linear-gradient(135deg,#7c3aed,#a855f7)",color:"#fff",border:"none",padding:"10px 24px",borderRadius:999,fontSize:13,fontWeight:700,cursor:scanning?"not-allowed":"pointer",opacity:scanning?0.6:1,boxShadow:scanning?"none":"0 6px 20px #7c3aed40",transition:"all 0.2s"}}>
            {scanning ? "⟳ Scanning..." : "⚡ Run Scan"}
          </button>
        </div>
      </div>

      {/* 3-col layout */}
      <div className="ql-twin-grid" style={{display:"grid",gridTemplateColumns:"1fr 200px 1fr",gap:20,alignItems:"start"}}>

        {/* Eye diagram */}
        <div style={{background:"#0a0a1a",border:"1px solid #a855f720",borderRadius:18,padding:18,display:"flex",flexDirection:"column",alignItems:"center",gap:14}}>
          <div style={{fontSize:10,fontWeight:700,color:"#9b89c4",letterSpacing:"1px",textTransform:"uppercase"}}>Retinal Cross-Section</div>
          <div style={{width:"100%",maxWidth:260,aspectRatio:"1/1",animation:"dtFloat 4s ease-in-out infinite"}}><EyeSVG/></div>
          <div style={{fontSize:10,color:"#7c6ea0",textAlign:"center"}}>Click any ring · Coloured dots = AI findings</div>
          <div style={{background:"linear-gradient(135deg,#1a0533,#2d1060)",border:`1px solid ${gradeColor}30`,borderRadius:12,padding:"12px 18px",width:"100%",textAlign:"center"}}>
            <div style={{fontSize:10,color:"#9b89c4",marginBottom:3}}>AI Diagnosis {result?"(Live)":"(Demo)"}</div>
            <div style={{fontSize:18,fontWeight:900,color:gradeColor}}>Grade {grade} - {result?.grade_label ?? "Moderate NPDR"}</div>
            <div style={{fontSize:11,color:"#c4b5fd",marginTop:2}}>Confidence: {result ? (result.confidence*100).toFixed(1) : "81.6"}% · Risk: {result?.risk_level ?? "HIGH"}</div>
            {result?.dme_detected && <div style={{fontSize:11,color:"#ef4444",marginTop:4,fontWeight:700}}>⚠ DME Detected</div>}
          </div>
        </div>

        {/* Layer selector */}
        <div style={{display:"flex",flexDirection:"column",gap:5}}>
          <div style={{fontSize:10,fontWeight:700,color:"#9b89c4",letterSpacing:"1px",textTransform:"uppercase",marginBottom:4}}>Zoom Layer</div>
          {layers.map((l,i)=>{
            const isAct=activeLayer===i, rev=revealed.has(i);
            const sevs=l.findings.length===0?"none":l.findings.reduce((acc,f)=>{const o=["none","low","moderate","high","critical"];return o.indexOf(f.severity)>o.indexOf(acc)?f.severity:acc;},"none");
            return <div key={l.id} className="dt-layer" onClick={()=>{setActiveLayer(i);setActiveFinding(null);}}
              style={{background:isAct?"#1a1040":"#111128",border:`1.5px solid ${isAct?LAYER_COLORS[i]:"#ffffff10"}`,borderRadius:10,padding:"9px 12px",cursor:"pointer",opacity:rev?1:0.3,transition:"all 0.2s"}}>
              <div style={{display:"flex",alignItems:"center",gap:8}}>
                <div style={{width:8,height:8,borderRadius:"50%",background:LAYER_COLORS[i],boxShadow:`0 0 5px ${LAYER_COLORS[i]}`,flexShrink:0}}/>
                <span style={{fontSize:12,fontWeight:isAct?700:400,color:isAct?"#fff":"#c4b5fd",flex:1}}>{l.label}</span>
                <div style={{width:7,height:7,borderRadius:"50%",background:SEVERITY_COLOR[sevs]}}/>
              </div>
              {isAct&&<div style={{fontSize:10,color:"#9b89c4",marginTop:4}}>{l.findings.length===0?"No findings":`${l.findings.length} finding${l.findings.length>1?"s":""}`}</div>}
            </div>;
          })}
          <div style={{marginTop:6,padding:"9px 12px",borderRadius:10,background:"#111128",border:"1px solid #ffffff08"}}>
            <div style={{fontSize:9,color:"#7c6ea0",marginBottom:5,fontWeight:700}}>SEVERITY</div>
            {Object.entries(SEVERITY_COLOR).map(([s,c])=><div key={s} style={{display:"flex",alignItems:"center",gap:6,marginBottom:3}}><div style={{width:6,height:6,borderRadius:"50%",background:c}}/><span style={{fontSize:10,color:"#9b89c4"}}>{SEVERITY_LABEL[s]}</span></div>)}
          </div>
        </div>

        {/* Detail panel */}
        <div style={{display:"flex",flexDirection:"column",gap:12,animation:"dtSlideIn 0.3s ease"}} key={layer.id}>
          <div style={{background:"#0a0a1a",border:`1.5px solid ${layer.borderColor}30`,borderRadius:16,padding:20}}>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12}}>
              <div style={{width:11,height:11,borderRadius:"50%",background:layer.borderColor,boxShadow:`0 0 10px ${layer.borderColor}`}}/>
              <span style={{fontSize:15,fontWeight:800,color:"#fff"}}>{layer.label}</span>
              <span style={{marginLeft:"auto",fontSize:10,padding:"2px 9px",borderRadius:999,background:`${layer.borderColor}20`,color:layer.borderColor,fontWeight:700}}>Layer {activeLayer+1}/6</span>
            </div>
            <p style={{fontSize:12,color:"#9b89c4",lineHeight:1.65,marginBottom:12}}>{layer.desc}</p>
            <div style={{background:"#111128",border:`1px solid ${layer.borderColor}20`,borderRadius:9,padding:"9px 13px"}}>
              <div style={{fontSize:9,fontWeight:700,color:layer.borderColor,letterSpacing:"1px",textTransform:"uppercase",marginBottom:5}}>AI Clinical Note</div>
              <div style={{fontSize:12,color:"#c4b5fd",lineHeight:1.6}}>{layer.clinicalNote}</div>
            </div>
          </div>

          {layer.findings.length>0 && (
            <div style={{background:"#0a0a1a",border:"1px solid #ffffff10",borderRadius:16,padding:20}}>
              <div style={{fontSize:11,fontWeight:700,color:"#c4b5fd",marginBottom:12}}>Findings ({layer.findings.length})</div>
              {layer.findings.map(f=>(
                <div key={f.id} onClick={()=>setActiveFinding(activeFinding?.id===f.id?null:f)}
                  style={{background:activeFinding?.id===f.id?"#1a1040":"#111128",border:`1px solid ${activeFinding?.id===f.id?f.color+"60":"#ffffff10"}`,borderRadius:9,padding:"9px 13px",cursor:"pointer",marginBottom:8,transition:"all 0.15s"}}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}>
                    <div style={{width:9,height:9,borderRadius:"50%",background:f.color,boxShadow:`0 0 6px ${f.color}`}}/>
                    <span style={{fontSize:12,fontWeight:600,color:"#e2d9f3",flex:1}}>{f.label}</span>
                    <span style={{fontSize:9,padding:"2px 7px",borderRadius:999,background:`${SEVERITY_COLOR[f.severity]}20`,color:SEVERITY_COLOR[f.severity],fontWeight:700}}>{SEVERITY_LABEL[f.severity]}</span>
                  </div>
                  {activeFinding?.id===f.id&&<p style={{fontSize:11,color:"#9b89c4",lineHeight:1.6,margin:"8px 0 0"}}>{f.desc}</p>}
                </div>
              ))}
            </div>
          )}
          {layer.findings.length===0&&<div style={{background:"#0a1a0a",border:"1px solid #22c55e20",borderRadius:12,padding:"13px 17px",display:"flex",alignItems:"center",gap:10}}><div style={{fontSize:20}}>✓</div><div><div style={{fontSize:12,fontWeight:700,color:"#22c55e"}}>No findings</div><div style={{fontSize:11,color:"#9b89c4",marginTop:2}}>Layer appears normal.</div></div></div>}

          <div style={{background:"linear-gradient(135deg,#0d0d1a,#1a0533)",border:"1px solid #a855f720",borderRadius:12,padding:"14px 18px"}}>
            <div style={{fontSize:9,fontWeight:700,color:"#a855f7",letterSpacing:"1px",textTransform:"uppercase",marginBottom:7}}>What this means for you</div>
            <p style={{fontSize:12,color:"#c4b5fd",lineHeight:1.65,margin:0}}>{PAT_EXPLAIN[activeLayer]}</p>
          </div>
        </div>
      </div>

      {/* Bottom strip */}
      <div style={{background:"#0a0a1a",border:"1px solid #ffffff08",borderRadius:14,padding:"12px 20px",display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:6}}>
        {layers.map((l,i)=>{
          const sev=l.findings.length===0?"none":l.findings.reduce((a,f)=>{const o=["none","low","moderate","high","critical"];return o.indexOf(f.severity)>o.indexOf(a)?f.severity:a;},"none");
          const sc=SEVERITY_COLOR[sev];
          return <div key={l.id} onClick={()=>{setActiveLayer(i);setActiveFinding(null);}}
            style={{textAlign:"center",padding:"8px 6px",borderRadius:9,background:activeLayer===i?"#1a1040":"transparent",border:`1px solid ${activeLayer===i?LAYER_COLORS[i]:"#ffffff08"}`,cursor:"pointer",transition:"all 0.2s"}}>
            <div style={{width:9,height:9,borderRadius:"50%",background:sc,boxShadow:`0 0 5px ${sc}`,margin:"0 auto 5px"}}/>
            <div style={{fontSize:10,fontWeight:600,color:"#e2d9f3"}}>{l.label}</div>
            <div style={{fontSize:9,color:sc,marginTop:1}}>{SEVERITY_LABEL[sev]}</div>
          </div>;
        })}
      </div>
    </div>
  );
}
