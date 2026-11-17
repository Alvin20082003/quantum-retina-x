import { useState, useRef, useCallback } from "react";
import DigitalTwin from "./DigitalTwin";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Upload, Activity, FileText, Eye, Brain, Settings, Info, Download, AlertTriangle, CheckCircle, Shield, Zap, User, ChevronDown, Droplets } from "lucide-react";

const API = "/api";

interface LesionInfo { detected: boolean; confidence: number; color: string; }
interface PipelineStage {
  label: string; params?: string; shape?: string; output_dim?: number;
  cls_norm?: number; values?: number[]; logits?: number[]; kept_fraction?: number;
  probability?: number; threshold?: number; detected?: boolean;
}
interface HistBin { bin: number; count: number; }
interface PreprocessingStats {
  raw_mean: number; raw_std: number; norm_mean: number; norm_std: number;
  raw_hist: HistBin[]; norm_hist: HistBin[];
}
interface PipelineInfo {
  input: PipelineStage; preprocessing: PreprocessingStats; backbone: PipelineStage;
  reducer: PipelineStage; vqc: PipelineStage; head: PipelineStage;
  fundus_mask: PipelineStage; dme: PipelineStage;
}
interface PredResult {
  grade: number; grade_label: string; confidence: number;
  probabilities: number[]; original_b64: string; heatmap_b64: string;
  redfree_b64?: string; mask_viz_b64?: string;
  lesion_map_b64?: string; lesion_detections?: Record<string, LesionInfo>;
  concept_heatmaps_b64?: Record<string, string>;
  marked_fraction?: number;
  tcav_scores: Record<string,number>; recommendation: string;
  risk_level: string; early_detection: boolean; model_type: string;
  inference_time: number; dr_type?: string; dme_detected?: boolean; dme_confidence?: number;
  pipeline?: PipelineInfo;
}

const GRADE_COLORS = ["#10b981","#f59e0b","#f97316","#ef4444","#dc2626"];
const RISK_COLORS: Record<string,string> = { LOW:"#10b981", MODERATE:"#f59e0b", HIGH:"#f97316", "VERY HIGH":"#ef4444", CRITICAL:"#dc2626" };

const ABLATION = [
  { config: "VQC Only (B)", accuracy: 54.6, f1: 0.237, auc: 0.646, kappa: 0.289, g1: 0.0, color: "#ef4444" },
  { config: "ViT Only (A)", accuracy: 78.4, f1: 0.638, auc: 0.936, kappa: 0.881, g1: 73.2, color: "#3b82f6" },
  { config: "ViT+VQC (C)", accuracy: 81.6, f1: 0.659, auc: 0.894, kappa: 0.871, g1: 55.4, color: "#8b5cf6" },
];

const TRAINING_STRATEGY = [
  { strategy: "Save on kappa", accuracy: 81.6, kappa: 0.871, g1: 55.4, g3: 41.4 },
  { strategy: "G1 weight x3, save on G1", accuracy: 74.6, kappa: 0.762, g1: 78.6, g3: 0.0 },
  { strategy: "G1 weight x3, gated save", accuracy: 81.5, kappa: 0.872, g1: 66.1, g3: 27.6 },
];

const CROSS_DATASET = [
  { dataset: "APTOS (India)", role: "Trained", vit_acc: 78.4, vqc_acc: 81.6, vit_auc: 0.936, vqc_auc: 0.894, vit_g1: 73.2, vqc_g1: 55.4 },
  { dataset: "Messidor (France)", role: "Tested (unseen)", vit_acc: 59.4, vqc_acc: 58.3, vit_auc: 0.760, vqc_auc: 0.577, vit_g1: 3.3, vqc_g1: 5.6 },
  { dataset: "IDRiD (India)", role: "Tested (unseen)", vit_acc: 42.9, vqc_acc: 43.5, vit_auc: 0.786, vqc_auc: 0.657, vit_g1: 31.8, vqc_g1: 50.0 },
  { dataset: "EyePACS (USA)", role: "Tested (unseen)", vit_acc: 64.2, vqc_acc: 63.0, vit_auc: 0.724, vqc_auc: 0.603, vit_g1: 18.1, vqc_g1: 19.0 },
];

const TCAV_REF: Record<string, Record<number, number>> = {
  hemorrhage: { 0: 0.0, 1: 0.2, 2: 0.9, 3: 1.0, 4: 1.0 },
  exudate: { 0: 0.0, 1: 0.67, 2: 1.0, 3: 1.0, 4: 1.0 },
  edema: { 0: 0.0, 1: 0.7, 2: 1.0, 3: 1.0, 4: 1.0 },
};

type Page = "dashboard"|"upload"|"pipeline"|"reports"|"explainability"|"insights"|"settings"|"about"|"twin";
const SIDEBAR: {icon:any;label:string;page:Page}[] = [
  {icon:Activity,label:"Dashboard",page:"dashboard"},{icon:Upload,label:"Upload Image",page:"upload"},
  {icon:Zap,label:"Pipeline",page:"pipeline"},
  {icon:FileText,label:"Reports",page:"reports"},{icon:Eye,label:"Explainability",page:"explainability"},
  {icon:Eye,label:"Digital Twin",page:"twin"},{icon:Brain,label:"Model Insights",page:"insights"},{icon:Settings,label:"Settings",page:"settings"},
  {icon:Info,label:"About",page:"about"},
];

function Card({title,children,icon}:{title:string;children:React.ReactNode;icon?:React.ReactNode}) {
  return <div style={{background:"#111128",border:"1px solid #ffffff10",padding:20}}><div className="text-sm font-semibold mb-4 flex items-center gap-2 text-purple-200">{icon}{title}</div>{children}</div>;
}
function StatBox({label,value,sub,color}:{label:string;value:string;sub?:string;color?:string}) {
  return <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}><div style={{color:"#9b89c4"}}>{label}</div><div style={{color:color||"#e2d9f3"}}>{value}</div>{sub&&<div style={{color:"#7c6ea0"}}>{sub}</div>}</div>;
}

function LesionPanel({lesions}:{lesions?:Record<string,LesionInfo>}) {
  const entries = Object.entries(lesions||{});
  return (
    <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
      <div style={{color:"#c4b5fd"}}><Eye size={14} className="text-purple-400"/>Lesion detection</div>
      <div className="space-y-2">{entries.length===0&&<div className="text-xs text-gray-400">No lesion detectors loaded.</div>}
        {entries.map(([name,info])=>(
        <div key={name} className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span style={{background:info.color}}/>
            <span className="text-gray-700">{name}</span>
          </div>
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${info.detected?"bg-green-500/20 text-green-400":"bg-slate-700 text-gray-400"}`}>
            {info.detected?`Detected (${info.confidence}%)`:"Not detected"}
          </span>
        </div>
      ))}</div>
    </div>
  );
}

export default function App() {
  const [page,setPage]=useState<Page>("dashboard");
  const [result,setResult]=useState<PredResult|null>(null);
  const [loading,setLoading]=useState(false);
  const [step,setStep]=useState(0);
  const [preview,setPreview]=useState<string|null>(null);
  const [modelType,setModelType]=useState<"hybrid"|"vit">("hybrid");
  const [dragOver,setDragOver]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const [heatmapOpacity,setHeatmapOpacity]=useState(85);
  const [scanHistory,setScanHistory]=useState<PredResult[]>([]);
  const [selectedScan,setSelectedScan]=useState<PredResult|null>(null);
  const [selectedStage,setSelectedStage]=useState<string|null>(null);
  const fileRef=useRef<HTMLInputElement>(null);
  const steps=["Reading 196 patches","Quantum circuit (16D Hilbert)","Generating Grad-CAM","Computing TCAV scores"];

  const handleFile=useCallback(async(file:File,destPage:Page="dashboard")=>{
    setError(null);setResult(null);setLoading(true);setStep(0);setPage(destPage);setSelectedScan(null);
    const reader=new FileReader();reader.onload=(e)=>setPreview(e.target?.result as string);reader.readAsDataURL(file);
    const iv=setInterval(()=>setStep(s=>Math.min(s+1,3)),1200);
    const form=new FormData();form.append("file",file);
    try{const res=await fetch(`${API}/predict?model_type=${modelType}`,{method:"POST",body:form});
      if(!res.ok)throw new Error(`Server error ${res.status}`);
      const data:PredResult=await res.json();clearInterval(iv);setResult(data);setLoading(false);
      setScanHistory(prev=>[data,...prev].slice(0,20));
    }catch(e:any){clearInterval(iv);setError(e.message);setLoading(false);}
  },[modelType]);

  const onDrop=useCallback((e:React.DragEvent,destPage:Page="dashboard")=>{e.preventDefault();setDragOver(false);if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0],destPage);},[handleFile]);

  const downloadHeatmap=()=>{if(!r)return;const link=document.createElement("a");link.href=`data:image/png;base64,${r.heatmap_b64}`;link.download=`quantumretina_heatmap_grade${r.grade}.png`;link.click();};
  const downloadReport=()=>{if(!r)return;const text=`QuantumRetina-X Diagnostic Report\n${"=".repeat(40)}\nDate: ${new Date().toLocaleString()}\nModel: ${r.model_type==="hybrid"?"ViT + VQC (Quantum)":"ViT Only"}\n\nDiabetic Retinopathy: ${r.grade_label}\nGrade: ${r.grade} | Confidence: ${(r.confidence*100).toFixed(1)}%\n\nDiabetic Macular Edema: ${r.dme_detected?"DETECTED":"Not detected"}\nDME Confidence: ${r.dme_confidence||0}%\n\nRisk Level: ${r.risk_level}\n\nTCAV Concept Scores:\n${Object.entries(r.tcav_scores).map(([k,v])=>`  ${k}: ${Number(v).toFixed(2)}`).join("\n")}\n\nRecommendation: ${r.recommendation}\nInference Time: ${r.inference_time}s\n${"=".repeat(40)}\nGenerated by QuantumRetina-X v1.0`;
    const blob=new Blob([text],{type:"text/plain"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`quantumretina_report.txt`;link.click();};

  const r = selectedScan || result;
  const probData = r ? r.probabilities.map((p,i)=>({name:`G${i}`,value:Math.round(p*100),grade:i})) : [];
  const tcavData = r ? Object.entries(r.tcav_scores).map(([k,v])=>({name:k.charAt(0).toUpperCase()+k.slice(1),value:Number((v as number).toFixed(2))})) : [];

  const renderUploadZone=(destPage:Page="dashboard")=>(
    <div style={{background:dragOver?"#1a1040":"#111128",border:dragOver?"2px dashed #a855f7":"2px dashed #ffffff20",borderRadius:20,padding:48,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:320,cursor:"pointer"}}
      onDragOver={e=>{e.preventDefault();setDragOver(true)}} onDragLeave={()=>setDragOver(false)} onDrop={e=>onDrop(e,destPage)} onClick={()=>fileRef.current?.click()}>
      <Upload size={48} className="text-blue-500 mb-4"/>
      <p style={{fontSize:18,fontWeight:700,color:"#e2d9f3",marginBottom:4}}>Drop retinal fundus image here</p>
      <p style={{fontSize:14,color:"#9b89c4",marginBottom:20}}>or click to browse - PNG, JPG supported</p>
      <button className="px-5 py-2 bg-purple-600 rounded-lg text-sm font-medium hover:bg-purple-500 transition">Select Image</button>
      <input ref={fileRef} type="file" className="hidden" accept="image/*" onChange={e=>{if(e.target.files?.[0])handleFile(e.target.files[0],destPage)}}/>
    </div>
  );

  const renderDashboard=()=>(
    <div style={{gridTemplateColumns:r?"1fr 340px":"1fr"}}>
      <div className="space-y-4">
        {!r&&!loading&&(
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            {/* Welcome banner */}
            <div style={{background:"linear-gradient(135deg,#1a0533,#2d1060)",border:"1px solid #a855f730",borderRadius:20,padding:"32px 36px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:24}}>
              <div>
                <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:10}}>QUANTUMRETINA-X · AI DIAGNOSTIC PLATFORM</div>
                <h2 style={{fontSize:28,fontWeight:900,color:"#fff",letterSpacing:-0.5,margin:0,marginBottom:10}}>Welcome, Dr. Admin</h2>
                <p style={{fontSize:15,color:"#c4b5fd",lineHeight:1.6,maxWidth:480,margin:0}}>Upload a retinal fundus image to get instant DR grading, DME detection, Grad-CAM heatmaps, and TCAV lesion analysis - all in one pass.</p>
              </div>
              <button onClick={()=>{setPage("upload");}} style={{background:"#a855f7",color:"#fff",border:"none",padding:"14px 32px",borderRadius:999,fontSize:15,fontWeight:700,cursor:"pointer",whiteSpace:"nowrap",boxShadow:"0 8px 24px #a855f740"}}>
                + New Scan
              </button>
            </div>
            {/* Mock result preview */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16}}>
              {/* Mock diagnosis card */}
              <div style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:16,padding:20}}>
                <div style={{fontSize:11,fontWeight:700,letterSpacing:"1px",textTransform:"uppercase",color:"#9b89c4",marginBottom:14}}>Example Diagnosis</div>
                <div style={{display:"flex",alignItems:"baseline",gap:10,marginBottom:6}}>
                  <span style={{fontSize:38,fontWeight:900,color:"#f59e0b"}}>G2</span>
                  <span style={{fontSize:16,fontWeight:700,color:"#fff"}}>Moderate NPDR</span>
                </div>
                <div style={{fontSize:13,color:"#9b89c4",marginBottom:14}}>Confidence: 81.6%</div>
                <div style={{height:5,borderRadius:3,background:"#1f1540",overflow:"hidden",marginBottom:14}}>
                  <div style={{height:"100%",width:"81.6%",borderRadius:3,background:"linear-gradient(90deg,#22c55e,#f59e0b,#ef4444)"}}/>
                </div>
                {[["G0","No DR","#22c55e"],["G1","Mild","#84cc16"],["G2","Moderate","#f59e0b"],["G3","Severe","#f97316"],["G4","PDR","#ef4444"]].map(([g,l,c])=>(
                  <div key={g} style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
                    <span style={{width:8,height:8,borderRadius:"50%",background:g==="G2"?c:"#ffffff20",display:"inline-block"}}/>
                    <span style={{fontSize:12,color:g==="G2"?"#fff":"#666",fontWeight:g==="G2"?700:400}}>{g} - {l}</span>
                    {g==="G2"&&<span style={{marginLeft:"auto",fontSize:11,color:c,fontWeight:700}}>DETECTED</span>}
                  </div>
                ))}
              </div>
              {/* Mock heatmap card */}
              <div style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:16,padding:20}}>
                <div style={{fontSize:11,fontWeight:700,letterSpacing:"1px",textTransform:"uppercase",color:"#9b89c4",marginBottom:14}}>Grad-CAM Heatmap</div>
                <div style={{borderRadius:12,overflow:"hidden",marginBottom:14,position:"relative",height:140,background:"#0a0a1a",display:"flex",alignItems:"center",justifyContent:"center"}}>
                  <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse at 55% 45%, #ef444460 0%, #f59e0b40 30%, #22c55e20 60%, transparent 80%)",borderRadius:12}}/>
                  <div style={{position:"absolute",inset:0,background:"radial-gradient(ellipse at 35% 60%, #a855f730 0%, transparent 50%)",borderRadius:12}}/>
                  <span style={{position:"relative",fontSize:12,color:"#9b89c4"}}>Heatmap preview</span>
                </div>
                <div style={{fontSize:12,color:"#9b89c4",lineHeight:1.6}}>Percentile-stretched Grad-CAM showing which retinal regions the model focused on. Hot = high attention.</div>
              </div>
              {/* Mock TCAV + lesion card */}
              <div style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:16,padding:20}}>
                <div style={{fontSize:11,fontWeight:700,letterSpacing:"1px",textTransform:"uppercase",color:"#9b89c4",marginBottom:14}}>TCAV Lesion Scores</div>
                {[["Hemorrhages","#ef4444",0.82],["Exudates","#22c55e",0.74],["Microaneurysms","#a855f7",0.61],["Macular Swelling","#3b82f6",0.38]].map(([name,color,score])=>(
                  <div key={name as string} style={{marginBottom:12}}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
                      <div style={{display:"flex",alignItems:"center",gap:7}}>
                        <span style={{width:8,height:8,borderRadius:"50%",background:color as string,display:"inline-block"}}/>
                        <span style={{fontSize:12,color:"#c4b5fd"}}>{name}</span>
                      </div>
                      <span style={{fontSize:12,fontWeight:700,color:color as string}}>{Math.round((score as number)*100)}%</span>
                    </div>
                    <div style={{height:5,borderRadius:3,background:"#1f1540"}}>
                      <div style={{height:"100%",width:`${(score as number)*100}%`,borderRadius:3,background:color as string,opacity:0.8}}/>
                    </div>
                  </div>
                ))}
                <div style={{marginTop:14,padding:"10px 14px",borderRadius:10,background:"#0a0a1a",border:"1px solid #3b82f620"}}>
                  <div style={{fontSize:11,color:"#60a5fa",fontWeight:600,marginBottom:3}}>DME: Not Detected</div>
                  <div style={{fontSize:11,color:"#666"}}>Confidence: 91.7%</div>
                </div>
              </div>
            </div>
            {/* Stats strip */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12}}>
              {[["81.6%","Model Accuracy","#a855f7"],["0.871","Quadratic Kappa","#c4b5fd"],["3,125","Trainable Params","#7c3aed"],["89.1%","DME Accuracy","#a855f7"]].map(([v,l,c])=>(
                <div key={l} style={{background:"#111128",border:"1px solid #ffffff08",borderRadius:14,padding:"18px 20px"}}>
                  <div style={{fontSize:26,fontWeight:900,color:c as string,letterSpacing:-0.5}}>{v}</div>
                  <div style={{fontSize:12,color:"#9b89c4",marginTop:4}}>{l}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {loading&&(
          <div style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:20,padding:48,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:350}}>
            {preview&&<img src={preview} className="w-40 h-40 rounded-xl object-cover mb-6 opacity-60"/>}
            <div className="w-full max-w-md">{steps.map((s,i)=>(
              <div key={i} className={`flex items-center gap-3 py-2 text-sm ${i<=step?"text-purple-300":"text-gray-300"}`}>
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${i<step?"bg-blue-600 text-white":i===step?"bg-blue-100 text-blue-600 animate-pulse":"bg-slate-800 text-gray-300"}`}>{i<step?"?":i+1}</div>{s}
              </div>))}</div>
          </div>)}
        {r&&(<>
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div className="grid grid-cols-3 gap-4">
              <div><div className="text-sm text-gray-500 mb-2 text-center">Original retinal image</div><img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/></div>
              <div><div className="text-sm text-gray-500 mb-2 text-center">Grad-CAM heatmap (AI focus)</div>
                <div className="relative"><img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/><img src={`data:image/png;base64,${r.heatmap_b64}`} style={{opacity:heatmapOpacity/100}}/></div></div>
              <div><div className="text-sm text-gray-500 mb-2 text-center">Lesion-type map</div>
                <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full rounded-lg"/></div>
            </div>
            <div className="flex items-center gap-3 mt-3 px-2"><span style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Heatmap</span><input type="range" min={0} max={100} value={heatmapOpacity} onChange={e=>setHeatmapOpacity(Number(e.target.value))} className="flex-1 accent-blue-500"/><span className="text-xs text-gray-500 w-10 text-right">{heatmapOpacity}%</span></div>
            <div className="flex justify-center flex-wrap gap-x-5 gap-y-1 mt-2 text-xs text-gray-500">
              {Object.entries(r.lesion_detections||{}).map(([name,info])=>(
                <span key={name} className="flex items-center gap-1"><span style={{background:info.color}}/> {name}</span>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <LesionPanel lesions={r.lesion_detections}/>
            <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
              <div style={{color:"#c4b5fd"}}>TCAV concept scores</div>
              {tcavData.map(t=>(<div key={t.name} className="mb-2">
                <div className="flex justify-between text-xs mb-1"><span className="text-gray-700">{t.name} impact</span><span className="text-gray-800 font-medium">{t.value}</span></div>
                <div className="h-2 rounded-full bg-slate-800"><div style={{width:`${t.value*100}%`}}/></div>
              </div>))}
            </div>
          </div>
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div style={{color:"#c4b5fd"}}>Grade probability distribution</div>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={probData}><XAxis dataKey="name" tick={{fill:"#64748b",fontSize:11}}/><YAxis tick={{fill:"#64748b",fontSize:11}} domain={[0,100]}/>
                <Tooltip contentStyle={{background:"#111128",border:"1px solid #a855f730",borderRadius:8,fontSize:12,color:"#c4b5fd"}}/>
                <Bar dataKey="value" radius={[4,4,0,0]}>{probData.map((d,i)=><Cell key={i} fill={d.grade===r.grade?GRADE_COLORS[r.grade]:"#e2e8f0"}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>)}
      </div>
      {r&&(
        <div className="space-y-4">
          {/* DR Diagnosis */}
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Diagnosis summary</div>
            <div className="flex items-center gap-3 mb-1">
              <div style={{background:"#1e293b"}}><Activity size={20} className="text-gray-700"/></div>
              <div><div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Diabetic Retinopathy (DR)</div><div style={{color:GRADE_COLORS[r.grade]}}>Grade {r.grade}</div></div>
              <span style={{background:GRADE_COLORS[r.grade]+"20",color:GRADE_COLORS[r.grade]}}>{r.dr_type||"NPDR"}</span>
            </div>
            <div className="flex justify-between items-center mt-2"><span style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Confidence</span><span style={{color:GRADE_COLORS[r.grade]}}>{(r.confidence*100).toFixed(1)}%</span></div>
            <div className="h-1.5 rounded-full bg-slate-800 mt-1"><div style={{width:`${r.confidence*100}%`,background:GRADE_COLORS[r.grade]}}/></div>
          </div>

          {/* DME Detection */}
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div className="flex items-center gap-3">
              <div style={{background:"#1e293b"}}><Droplets size={20} className="text-blue-400"/></div>
              <div>
                <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Diabetic Macular Edema (DME)</div>
                <div className={`text-xl font-bold ${r.dme_detected?"text-red-400":"text-green-400"}`}>{r.dme_detected?"Detected":"Not detected"}</div>
              </div>
              <span className={`ml-auto text-[10px] px-2 py-1 rounded-full font-medium ${r.dme_detected?"bg-red-500/20 text-red-400":"bg-green-500/20 text-green-400"}`}>{r.dme_detected?"Present":"Absent"}</span>
            </div>
            <div className="flex justify-between items-center mt-3"><span style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Confidence</span><span className={`text-lg font-bold ${r.dme_detected?"text-red-400":"text-green-400"}`}>{r.dme_confidence}%</span></div>
          </div>

          {/* Early Detection Alert */}
          {r.early_detection&&(
            <div style={{background:"#f59e0b15",border:"1px solid #f59e0b40"}}>
              <AlertTriangle size={20} className="text-amber-400"/>
              <div><div className="text-sm font-medium text-amber-300">Early detection alert</div><div className="text-xs text-amber-400/70">Mild DR detected ? early intervention can prevent blindness</div></div>
            </div>
          )}

          {/* Risk Assessment */}
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div style={{color:"#c4b5fd"}}>Risk assessment</div>
            <span style={{background:RISK_COLORS[r.risk_level]+"20",color:RISK_COLORS[r.risk_level]}}>{r.risk_level}</span>
            <div className="text-xs text-gray-500 leading-relaxed mt-3">{r.recommendation}</div>
          </div>

          {/* AI Explanation */}
          <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
            <div className="text-sm font-medium mb-2 flex items-center gap-2"><Zap size={14} className="text-purple-400"/> AI explanation (XAI)</div>
            <div className="text-xs text-gray-400 mb-2">Reason for prediction:</div>
            <div className="text-xs text-gray-500 leading-relaxed space-y-1.5">
              {Object.entries(r.tcav_scores).map(([concept,score])=>(
                <div key={concept} className="flex items-start gap-2"><CheckCircle size={12} className="text-green-400 mt-0.5 shrink-0"/>
                  <span>{concept.charAt(0).toUpperCase()+concept.slice(1)} pattern {Number(score)>0.7?"strongly detected":Number(score)>0.4?"moderately detected":"weakly detected"} (score: {Number(score).toFixed(2)})</span>
                </div>))}
              {r.dme_detected&&<div className="flex items-start gap-2"><CheckCircle size={12} className="text-green-400 mt-0.5 shrink-0"/><span>Macular swelling indicators suggest presence of DME</span></div>}
              <div className="flex items-start gap-2"><CheckCircle size={12} className="text-green-400 mt-0.5 shrink-0"/><span>Overall lesion pattern matches {r.grade_label} profile</span></div>
            </div>
          </div>

          <div style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="flex justify-between"><span>Model</span><span className="text-gray-700">{r.model_type==="hybrid"?"ViT + VQC (Quantum)":"ViT Only"}</span></div>
            <div className="flex justify-between mt-1"><span>Inference</span><span className="text-gray-700">{r.inference_time}s</span></div>
          </div>
        </div>
      )}
    </div>
  );

  const PipelineNode=({title,spec,children,accent,onClick,selected}:{title:string;spec?:string;children?:React.ReactNode;accent?:string;onClick?:()=>void;selected?:boolean})=>(
    <div onClick={onClick} className={`rounded-xl p-3 flex-1 min-w-[150px] transition-all ${onClick?"cursor-pointer hover:brightness-125":""}`}
      style={{background:selected?"#1a1040":"#111128",border:`1.5px solid ${selected?"#a855f7":(accent||"#ffffff15")}`,boxShadow:selected?"0 0 0 2px #a855f750, 0 8px 32px #a855f720":"none",borderRadius:16,padding:"16px 18px"}}>
      <div className="text-xs font-semibold text-gray-800 flex items-center justify-between">{title}{onClick&&<span className="text-[9px] text-purple-400">click â“˜</span>}</div>
      {spec&&<div style={{fontSize:10,color:"#9b89c4",marginBottom:8,lineHeight:1.4}}>{spec}</div>}
      {children}
    </div>
  );
  const PipelineArrow=()=>(<div className="flex items-center justify-center text-purple-400 px-1 shrink-0">â†’</div>);
  const Formula=({children}:{children:React.ReactNode})=>(
    <div style={{background:"#1e3a8a",border:"1px solid #2d2050"}}>{children}</div>
  );
  const MiniHist=({data,color}:{data:HistBin[];color:string})=>(
    <ResponsiveContainer width="100%" height={70}>
      <BarChart data={data.map(d=>({name:d.bin,value:d.count}))}>
        <Bar dataKey="value" radius={[2,2,0,0]} fill={color}/>
      </BarChart>
    </ResponsiveContainer>
  );
  const ValueChips=({values}:{values?:number[]})=>(
    <div className="flex flex-wrap gap-1">{(values||[]).map((v,i)=>(
      <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-gray-700 font-mono">{v.toFixed(2)}</span>
    ))}</div>
  );

  const PIPELINE_STAGES=["Input image","Preprocessing","ViT Backbone","Reducer (768â†’4)","4-Qubit VQC","Head + Output"];
  const renderPipelineLoading=()=>(
    <div style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
      <div className="flex items-center gap-2 mb-5">
        {preview&&<img src={preview} className="w-14 h-14 rounded-lg object-cover"/>}
        <div className="text-sm text-gray-700">Passing your image through the real model, live...</div>
      </div>
      <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
        {PIPELINE_STAGES.map((label,i)=>{
          const lit = i<=step+1;
          return (<div key={label} className="flex items-stretch">
            <div className={`rounded-xl p-3 flex-1 min-w-[130px] transition-all duration-500 ${lit?"":"opacity-30"}`}
              style={{background:lit?"#eff6ff":"#ffffff",border:`1px solid ${lit?"#7c3aed":"#1e293b"}`}}>
              <div className={`text-xs font-semibold ${lit?"text-blue-700":"text-gray-400"}`}>{label}</div>
              {lit&&i<=step+1&&i>0&&<div className="mt-2 flex gap-1">{[0,1,2].map(d=><span key={d} className="w-1 h-1 rounded-full bg-purple-400 animate-pulse"/>)}</div>}
              {lit&&i===0&&<CheckCircle size={14} className="text-purple-300 mt-1"/>}
            </div>
            {i<PIPELINE_STAGES.length-1&&<div className="flex items-center justify-center text-purple-400 px-1">â†’</div>}
          </div>);
        })}
      </div>
    </div>
  );

  const renderPipeline=()=>{
    const p=r?.pipeline;
    return (<div style={{padding:"32px 36px",display:"flex",flexDirection:"column",gap:28,background:"#070711",minHeight:"100%"}}>
      <div>
        <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:8}}>MODEL PIPELINE</div>
        <h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:0}}>Forward pass visualiser</h2>
        <p style={{fontSize:14,color:"#9b89c4",marginTop:8}}>Drop an image below to watch it pass through the real model, stage by stage.</p>
      </div>
      {loading?renderPipelineLoading():!r||!p?(
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          {/* Static pipeline diagram */}
          <div style={{background:"#0f0f1f",border:"1px solid #a855f720",borderRadius:20,padding:28}}>
            <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#9b89c4",marginBottom:20}}>FORWARD PASS - Upload an image above to see real values at each stage</div>
            <div className="ql-pipeline-row" style={{display:"flex",alignItems:"stretch",gap:4,overflowX:"auto",paddingBottom:8}}>
              {[
                {title:"Input",spec:"224 × 224 × 3",detail:"Fundus photo",color:"#a855f7"},
                {title:"Preprocessing",spec:"Resize + Normalize",detail:"ImageNet μ/σ",color:"#eab308"},
                {title:"ViT Backbone",spec:"86.4M params, frozen",detail:"[CLS] → 768-dim",color:"#3b82f6"},
                {title:"Reducer",spec:"Linear 768 → 4",detail:"Compression bridge",color:"#10b981"},
                {title:"VQC",spec:"4-qubit circuit",detail:"2 layers, CNOT ring",color:"#f59e0b"},
                {title:"Head",spec:"Linear 4 → 5",detail:"5-class output",color:"#10b981"},
                {title:"Output",spec:"DR Grade G0-G4",detail:"+ DME + XAI",color:"#a855f7"},
              ].map((node,i,arr)=>(
                <div key={node.title} style={{display:"flex",alignItems:"stretch"}}>
                  <div style={{background:"#111128",border:`1.5px solid ${node.color}30`,borderRadius:14,padding:"16px 18px",minWidth:140,flex:1,display:"flex",flexDirection:"column",gap:4}}>
                    <div style={{fontSize:12,fontWeight:700,color:"#e2d9f3"}}>{node.title}</div>
                    <div style={{fontSize:10,color:"#9b89c4",lineHeight:1.4}}>{node.spec}</div>
                    <div style={{marginTop:"auto",paddingTop:10}}>
                      <div style={{height:3,borderRadius:2,background:`${node.color}40`,overflow:"hidden"}}>
                        <div style={{height:"100%",width:"60%",background:node.color,borderRadius:2,opacity:0.6}}/>
                      </div>
                      <div style={{fontSize:10,color:node.color,marginTop:5,fontWeight:600}}>{node.detail}</div>
                    </div>
                  </div>
                  {i<arr.length-1&&<div style={{display:"flex",alignItems:"center",color:"#a855f7",padding:"0 8px",flexShrink:0,fontSize:18,opacity:0.5}}>→</div>}
                </div>
              ))}
            </div>
          </div>

          {/* Explainability branch */}
          <div style={{background:"#0f0f1f",border:"1px solid #ef444420",borderRadius:20,padding:28}}>
            <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#f87171",marginBottom:20}}>↳ EXPLAINABILITY BRANCH (Grad-CAM) - spatial heatmap of model attention</div>
            <div style={{display:"flex",alignItems:"stretch",gap:4}}>
              {[
                {title:"Fundus Mask",spec:"Otsu + ellipse fit",detail:"18% shrink inward",color:"#ef4444"},
                {title:"Grad-CAM Backprop",spec:"Gradients → last ViT layer",detail:"Backward pass",color:"#ef4444"},
                {title:"Heatmap",spec:"Percentile-stretched",detail:"60-99.5th pct, masked",color:"#ef4444"},
              ].map((node,i,arr)=>(
                <div key={node.title} style={{display:"flex",alignItems:"stretch",flex:1}}>
                  <div style={{background:"#111128",border:`1.5px solid ${node.color}25`,borderRadius:14,padding:"16px 18px",flex:1}}>
                    <div style={{fontSize:12,fontWeight:700,color:"#e2d9f3",marginBottom:4}}>{node.title}</div>
                    <div style={{fontSize:10,color:"#9b89c4",lineHeight:1.4,marginBottom:8}}>{node.spec}</div>
                    <div style={{fontSize:10,color:node.color,fontWeight:600}}>{node.detail}</div>
                  </div>
                  {i<arr.length-1&&<div style={{display:"flex",alignItems:"center",color:"#ef4444",padding:"0 8px",flexShrink:0,fontSize:18,opacity:0.5}}>→</div>}
                </div>
              ))}
            </div>
          </div>

          {/* TCAV branch */}
          <div style={{background:"#0f0f1f",border:"1px solid #3b82f420",borderRadius:20,padding:28}}>
            <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#60a5fa",marginBottom:20}}>↳ CONCEPT BRANCH (TCAV) - why that grade was predicted</div>
            <div style={{display:"flex",alignItems:"stretch",gap:4}}>
              {[
                {title:"Concept Scoring",spec:"4 probes on [CLS]",detail:"Microaneurysm / Hemorrhage / Exudate / Edema",color:"#3b82f6"},
                {title:"Per-Concept Maps",spec:"4 separate backprops",detail:"One spatial map per lesion type",color:"#3b82f6"},
                {title:"Multi-Color Merge",spec:"Winner-take-all",detail:"Threshold 0.3, 4 colors",color:"#3b82f6"},
              ].map((node,i,arr)=>(
                <div key={node.title} style={{display:"flex",alignItems:"stretch",flex:1}}>
                  <div style={{background:"#111128",border:`1.5px solid ${node.color}25`,borderRadius:14,padding:"16px 18px",flex:1}}>
                    <div style={{fontSize:12,fontWeight:700,color:"#e2d9f3",marginBottom:4}}>{node.title}</div>
                    <div style={{fontSize:10,color:"#9b89c4",lineHeight:1.4,marginBottom:8}}>{node.spec}</div>
                    <div style={{fontSize:10,color:node.color,fontWeight:600}}>{node.detail}</div>
                  </div>
                  {i<arr.length-1&&<div style={{display:"flex",alignItems:"center",color:"#3b82f6",padding:"0 8px",flexShrink:0,fontSize:18,opacity:0.5}}>→</div>}
                </div>
              ))}
            </div>
          </div>

          {/* DME branch */}
          <div style={{background:"#0f0f1f",border:"1px solid #10b98120",borderRadius:20,padding:28}}>
            <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#34d399",marginBottom:20}}>↳ DME BRANCH - parallel model for macular edema detection</div>
            <div style={{display:"flex",alignItems:"stretch",gap:4}}>
              {[
                {title:"Same Image",spec:"Identical preprocessing",detail:"No extra input needed",color:"#10b981"},
                {title:"2nd ViT+VQC Model",spec:"Separately trained",detail:"Messidor + IDRiD data",color:"#10b981"},
                {title:"DME Output",spec:"Binary: Present / Absent",detail:"Sigmoid threshold 0.5",color:"#10b981"},
              ].map((node,i,arr)=>(
                <div key={node.title} style={{display:"flex",alignItems:"stretch",flex:1}}>
                  <div style={{background:"#111128",border:`1.5px solid ${node.color}25`,borderRadius:14,padding:"16px 18px",flex:1}}>
                    <div style={{fontSize:12,fontWeight:700,color:"#e2d9f3",marginBottom:4}}>{node.title}</div>
                    <div style={{fontSize:10,color:"#9b89c4",lineHeight:1.4,marginBottom:8}}>{node.spec}</div>
                    <div style={{fontSize:10,color:node.color,fontWeight:600}}>{node.detail}</div>
                  </div>
                  {i<arr.length-1&&<div style={{display:"flex",alignItems:"center",color:"#10b981",padding:"0 8px",flexShrink:0,fontSize:18,opacity:0.5}}>→</div>}
                </div>
              ))}
            </div>
          </div>

          {/* CTA to upload */}
          <div style={{textAlign:"center",padding:"20px 0 8px"}}>
            <p style={{fontSize:14,color:"#9b89c4",marginBottom:16}}>Upload a fundus image to see real values flowing through each stage above</p>
            <button onClick={()=>setPage("upload")} style={{background:"#7c3aed",color:"#fff",border:"none",padding:"13px 32px",borderRadius:999,fontSize:14,fontWeight:700,cursor:"pointer",boxShadow:"0 8px 24px #7c3aed40"}}>
              + Upload Image to Activate Pipeline
            </button>
          </div>
        </div>
      ):(<>
        <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
          <div className="text-xs text-gray-400 uppercase tracking-wider mb-3">Forward pass â€” click any stage to see what actually happens there</div>
          <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
            <PipelineNode title="Input" spec={p.input.shape} selected={selectedStage==="input"} onClick={()=>setSelectedStage(selectedStage==="input"?null:"input")}>
              <img src={`data:image/png;base64,${r.original_b64}`} className="w-full h-16 object-cover rounded"/>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Preprocessing" spec="Resize + normalize" accent="#eab308" selected={selectedStage==="preprocessing"} onClick={()=>setSelectedStage(selectedStage==="preprocessing"?null:"preprocessing")}>
              <div className="text-[10px] text-gray-500">Î¼={p.preprocessing.raw_mean}â†’{p.preprocessing.norm_mean}</div>
              <div className="text-[10px] text-gray-500">Ïƒ={p.preprocessing.raw_std}â†’{p.preprocessing.norm_std}</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="ViT Backbone" spec={`${p.backbone.params} params, frozen`} accent="#3b82f6" selected={selectedStage==="backbone"} onClick={()=>setSelectedStage(selectedStage==="backbone"?null:"backbone")}>
              <div className="text-[10px] text-gray-500">[CLS] dim {p.backbone.output_dim}</div>
              <div className="text-[10px] text-gray-500">â€-featuresâ€- = {p.backbone.cls_norm}</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Reducer" spec={p.reducer.label} accent="#10b981" selected={selectedStage==="reducer"} onClick={()=>setSelectedStage(selectedStage==="reducer"?null:"reducer")}>
              <ValueChips values={p.reducer.values}/>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="VQC" spec={p.vqc.label} accent="#f59e0b" selected={selectedStage==="vqc"} onClick={()=>setSelectedStage(selectedStage==="vqc"?null:"vqc")}>
              <ValueChips values={p.vqc.values}/>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Head" spec={p.head.label} accent="#10b981" selected={selectedStage==="head"} onClick={()=>setSelectedStage(selectedStage==="head"?null:"head")}>
              <div className="text-[10px] text-gray-500">{p.head.logits?.length} class probs</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Output" accent={GRADE_COLORS[r.grade]} selected={selectedStage==="output"} onClick={()=>setSelectedStage(selectedStage==="output"?null:"output")}>
              <div style={{color:GRADE_COLORS[r.grade]}}>{r.grade_label}</div>
              <div className="text-[10px] text-gray-500">{(r.confidence*100).toFixed(1)}% confidence</div>
            </PipelineNode>
          </div>

          {selectedStage&&(
            <div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #a855f720"}}>
              {selectedStage==="input"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Input: the raw image tensor</div>
                <div className="grid grid-cols-2 gap-4">
                  <img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/>
                  <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>
                    <Formula>x âˆˆ â„^(224 Ã- 224 Ã- 3), x[i,j,c] âˆˆ [0, 255]</Formula>
                    <p>The uploaded photo is resized to 224Ã-224 with 3 color channels (R, G, B) â€” a plain integer pixel array. Nothing model-specific has happened yet; this is exactly what you uploaded, reformatted for the network.</p>
                  </div>
                </div>
              </div>)}
              {selectedStage==="preprocessing"&&p.preprocessing&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Preprocessing: cleaning the pixels before the model sees them</div>
                <Formula>x&apos; = (x / 255 âˆ’ Î¼) / Ïƒ,  Î¼ = [0.485, 0.456, 0.406],  Ïƒ = [0.229, 0.224, 0.225]</Formula>
                <p style={{fontSize:13,color:"#9b89c4",marginBottom:12,lineHeight:1.7}}>Every pixel is scaled to [0,1] then normalized using the same per-channel mean/std the ViT backbone was pretrained on (ImageNet statistics) â€” this isn&apos;t a cosmetic step, it matches the input distribution the frozen backbone actually expects. No cropping or CLAHE enhancement is applied here (this app currently sends the plain resized+normalized image to the model â€” CLAHE only exists in the separate red-free display view, not what the model sees).</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-[10px] text-gray-400 mb-1">Raw pixel distribution (0-255) â€” Î¼={p.preprocessing.raw_mean}, Ïƒ={p.preprocessing.raw_std}</div>
                    <MiniHist data={p.preprocessing.raw_hist} color="#eab308"/>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-400 mb-1">Normalized distribution (fed to model) â€” Î¼={p.preprocessing.norm_mean}, Ïƒ={p.preprocessing.norm_std}</div>
                    <MiniHist data={p.preprocessing.norm_hist} color="#a855f7"/>
                  </div>
                </div>
              </div>)}
              {selectedStage==="backbone"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>ViT Backbone: split into patches, read with attention</div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="relative">
                    <img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/>
                    <div style={{
                      backgroundImage:"linear-gradient(to right, rgba(59,130,246,0.5) 1px, transparent 1px), linear-gradient(to bottom, rgba(59,130,246,0.5) 1px, transparent 1px)",
                      backgroundSize:"calc(100%/14) calc(100%/14)"}}/>
                  </div>
                  <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>
                    <Formula>Attention(Q,K,V) = softmax(QK^T / âˆšd) V</Formula>
                    <p>The image is cut into a real 14Ã-14 grid of 16Ã-16-pixel patches (196 patches total) â€” shown overlaid on the actual image, not illustrative.</p>
                    <p>Each patch becomes a vector; 12 self-attention layers let every patch "look at" every other patch using the formula above, so a lesion's relationship to the optic disc or macula is captured. The whole 86.4M-parameter backbone is frozen â€” it never updates during training here.</p>
                    <p>Only the [CLS] token â€” a single 768-number summary of the entire image â€” is passed onward. Its magnitude for this image: <span className="text-gray-800 font-mono">â€-hâ€- = {p.backbone.cls_norm}</span></p>
                  </div>
                </div>
              </div>)}
              {selectedStage==="reducer"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Reducer: 768 numbers compressed to 4</div>
                <Formula>y = WÂ·h + b,  W âˆˆ â„^(4 Ã- 768), h âˆˆ â„^768 â†’ y âˆˆ â„^4</Formula>
                <div style={{fontSize:13,color:"#9b89c4",marginBottom:12,lineHeight:1.7}}>A single trainable linear layer compresses the 768-number [CLS] summary down to 4 numbers â€” the only "bridge" between the classical and quantum parts. This is a 192Ã- compression â€” a real bottleneck (see the audit report for what this costs).</div>
                <div className="space-y-1.5 max-w-md">{(p.reducer.values||[]).map((v,i)=>(
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 w-4">x{i}</span>
                    <div style={{flex:1,height:6,borderRadius:3,background:"#1f1540",position:"relative",overflow:"hidden"}}>
                      <div style={{width:`${Math.min(Math.abs(v)/3*100,100)}%`,marginLeft:v<0?"auto":0,marginRight:v<0?0:"auto",background:v<0?"#ef4444":"#10b981"}}/>
                    </div>
                    <span className="text-[10px] font-mono text-gray-700 w-12 text-right">{v.toFixed(3)}</span>
                  </div>
                ))}</div>
              </div>)}
              {selectedStage==="vqc"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>4-Qubit VQC: the actual quantum measurement</div>
                <Formula>|Ïˆáµ¢âŸ© = R_Y(yáµ¢)|0âŸ© &nbsp;â†’&nbsp; Uâ‚‚Uâ‚|ÏˆâŸ© &nbsp;â†’&nbsp; âŸ¨Záµ¢âŸ© = âŸ¨Ïˆ|Záµ¢|ÏˆâŸ©</Formula>
                <div style={{fontSize:13,color:"#9b89c4",marginBottom:12,lineHeight:1.7}}>Each of the 4 reducer numbers rotates one qubit (angle encoding, R_Y). Two trainable layers of rotations + CNOT entangling gates (Uâ‚, Uâ‚‚) mix them together in a 16-dimensional Hilbert space. Each qubit is then measured (Pauli-Z expectation âŸ¨Záµ¢âŸ©, range -1 to +1) â€” these are the real measured values for this image:</div>
                <div className="space-y-1.5 max-w-md">{(p.vqc.values||[]).map((v,i)=>(
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 w-8">qubit {i}</span>
                    <div style={{flex:1,height:6,borderRadius:3,background:"#1f1540",position:"relative",overflow:"hidden"}}>
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600"/>
                      <div style={{width:`${Math.abs(v)/2*100}%`,left:v<0?`${50-Math.abs(v)/2*100}%`:"50%",background:"#f59e0b"}}/>
                    </div>
                    <span className="text-[10px] font-mono text-gray-700 w-12 text-right">{v.toFixed(3)}</span>
                  </div>
                ))}</div>
              </div>)}
              {selectedStage==="head"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Head: 4 quantum outputs â†’ 5 grade probabilities</div>
                <Formula>z = WÂ·âŸ¨ZâŸ© + b &nbsp;â†’&nbsp; p_c = e^(z_c) / Î£â±¼ e^(z_j)</Formula>
                <div style={{fontSize:13,color:"#9b89c4",marginBottom:12,lineHeight:1.7}}>A final linear layer maps the 4 quantum measurements to 5 logits, then softmax turns them into a probability for each DR grade:</div>
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart data={probData}><XAxis dataKey="name" tick={{fill:"#64748b",fontSize:11}}/><YAxis tick={{fill:"#64748b",fontSize:11}} domain={[0,100]}/>
                    <Tooltip contentStyle={{background:"#111128",border:"1px solid #a855f730",borderRadius:8,fontSize:12,color:"#c4b5fd"}}/>
                    <Bar dataKey="value" radius={[4,4,0,0]}>{probData.map((d,i)=><Cell key={i} fill={d.grade===r.grade?GRADE_COLORS[r.grade]:"#e2e8f0"}/>)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>)}
              {selectedStage==="output"&&(<div>
                <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Output: final diagnosis</div>
                <div className="text-xs text-gray-500 space-y-1">
                  <p><span style={{color:GRADE_COLORS[r.grade]}}>{r.grade_label}</span> â€” {(r.confidence*100).toFixed(1)}% confidence, risk level {r.risk_level}.</p>
                  <p>{r.recommendation}</p>
                  <p>Total inference time: {r.inference_time}s.</p>
                </div>
              </div>)}
            </div>
          )}
        </div>

        {/* Branch A: Grad-CAM explainability pipeline */}
        <div style={{background:"#ffffff",border:"1px solid #ef444440"}}>
          <div className="text-xs text-red-300 uppercase tracking-wider mb-3">â†³ Explainability branch (Grad-CAM) â€” WHERE it looked. Click any stage.</div>
          <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
            <PipelineNode title="Fundus Mask" spec={p.fundus_mask?.label} accent="#ef4444" selected={selectedStage==="fundus_mask"} onClick={()=>setSelectedStage(selectedStage==="fundus_mask"?null:"fundus_mask")}>
              <img src={`data:image/png;base64,${r.mask_viz_b64||r.original_b64}`} className="w-full h-16 object-cover rounded"/>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Grad-CAM Backprop" spec="Gradients â†’ last ViT layer" accent="#ef4444" selected={selectedStage==="gradcam_backprop"} onClick={()=>setSelectedStage(selectedStage==="gradcam_backprop"?null:"gradcam_backprop")}>
              <div className="text-[10px] text-gray-500">Second, backward pass</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Heatmap" spec="Percentile-stretched, masked" accent="#ef4444" selected={selectedStage==="heatmap_render"} onClick={()=>setSelectedStage(selectedStage==="heatmap_render"?null:"heatmap_render")}>
              <img src={`data:image/png;base64,${r.heatmap_b64}`} className="w-full h-16 object-cover rounded"/>
            </PipelineNode>
          </div>
          {selectedStage==="fundus_mask"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #ef444420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Fundus Mask: isolating the real eye from the black background</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.mask_viz_b64||r.original_b64}`} className="w-full rounded-lg"/>
              <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>
                <p>The black corners around a fundus photo aren&apos;t retina â€” but the ViT&apos;s LayerNorm blows up activations on these low-variance patches, producing a false &quot;hot ring&quot; with no clinical meaning. This step fits an ellipse to the real eye (Otsu threshold + 18% shrink) so every later stage only looks inside it.</p>
                <p>Red-tinted area (above) = excluded. Untouched area = the {p.fundus_mask?.kept_fraction}% of the frame that Grad-CAM and the lesion maps are allowed to mark.</p>
              </div>
            </div>
          </div>)}
          {selectedStage==="gradcam_backprop"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #ef444420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Grad-CAM Backprop: a second, backward pass through the network</div>
            <Formula>Î±_k = (1/Z) Î£áµ¢â±¼ âˆ‚y/âˆ‚A_ij^k &nbsp;â†’&nbsp; L = ReLU(Î£_k Î±_k A^k)</Formula>
            <p style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Unlike the forward pass above (which only computes a prediction), this stage runs a second pass in reverse: the gradient of the predicted grade&apos;s score is traced backward to the ViT&apos;s last encoder layer, reshaped into the 14Ã-14 patch grid. Patches that mattered most for the prediction get the highest weight Î±_k â€” this is what turns into the heatmap on the right.</p>
          </div>)}
          {selectedStage==="heatmap_render"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #ef444420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Heatmap: the raw Grad-CAM signal, stretched and masked for readability</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.heatmap_b64}`} className="w-full rounded-lg"/>
              <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>
                <p>The raw Grad-CAM signal is percentile-stretched (60thâ€“99.5th percentile) so genuine hotspots stand out instead of washing into uniform color, then hard-masked to the fundus mask from the first stage so nothing is ever marked outside the eye.</p>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch B: TCAV concept pipeline */}
        <div style={{background:"#ffffff",border:"1px solid #3b82f640"}}>
          <div className="text-xs text-blue-300 uppercase tracking-wider mb-3">â†³ Concept branch (TCAV) â€” WHY that grade. Click any stage.</div>
          <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
            <PipelineNode title="Concept Scoring" spec="4 trained probes on [CLS]" accent="#3b82f6" selected={selectedStage==="concept_scoring"} onClick={()=>setSelectedStage(selectedStage==="concept_scoring"?null:"concept_scoring")}>
              {Object.entries(r.tcav_scores).slice(0,2).map(([k,v])=>(
                <div key={k} className="text-[10px] text-gray-500 flex justify-between"><span className="capitalize">{k}</span><span>{(v*100).toFixed(0)}%</span></div>))}
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Per-Concept Maps" spec="4 separate backprops" accent="#3b82f6" selected={selectedStage==="concept_maps"} onClick={()=>setSelectedStage(selectedStage==="concept_maps"?null:"concept_maps")}>
              <div className="grid grid-cols-2 gap-0.5">{Object.values(r.concept_heatmaps_b64||{}).slice(0,4).map((b64,i)=>(
                <img key={i} src={`data:image/png;base64,${b64}`} className="w-full h-7 object-cover rounded-sm"/>))}</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="Multi-Color Merge" spec="Winner-take-all, threshold 0.3" accent="#3b82f6" selected={selectedStage==="multicolor_merge"} onClick={()=>setSelectedStage(selectedStage==="multicolor_merge"?null:"multicolor_merge")}>
              <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full h-16 object-cover rounded"/>
            </PipelineNode>
          </div>
          {selectedStage==="concept_scoring"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #3b82f420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Concept Scoring: the same [CLS] feature, scored against 4 clinical concepts</div>
            <Formula>P(concept | h) = Ïƒ(wáµ€h + b)</Formula>
            <p className="text-xs text-gray-500 mb-2">The same 768-number ViT feature used for grading is reused here, but scored against 4 separate logistic-regression probes, each trained on real lesion patches cropped from expert-annotated masks (microaneurysm, hemorrhage, hard exudate, macular edema).</p>
            {Object.entries(r.tcav_scores).map(([k,v])=>(<div key={k} className="mb-1.5">
              <div className="flex justify-between text-[10px] text-gray-500"><span className="capitalize">{k}</span><span>{(v*100).toFixed(0)}%</span></div>
              <div style={{height:5,borderRadius:3,background:"#1f1540"}}><div style={{height:"100%",borderRadius:3,background:"#3b82f6",width:`${v*100}%`}}/></div>
            </div>))}
          </div>)}
          {selectedStage==="concept_maps"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #3b82f420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Per-Concept Maps: where each lesion type is, individually, before merging</div>
            <p style={{fontSize:13,color:"#9b89c4",marginBottom:12,lineHeight:1.7}}>Each concept gets its own Grad-CAM-style backward pass â€” the concept probe&apos;s own decision score is backpropagated instead of the DR-grade class score, producing 4 independent spatial maps.</p>
            <div className="grid grid-cols-4 gap-2">{Object.entries(r.concept_heatmaps_b64||{}).map(([k,b64])=>(
              <div key={k}><img src={`data:image/png;base64,${b64}`} className="w-full rounded-lg mb-1"/><div className="text-[10px] text-gray-500 text-center capitalize">{k}</div></div>))}</div>
          </div>)}
          {selectedStage==="multicolor_merge"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #3b82f420"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>Multi-Color Merge: 4 maps combined into one, each lesion type in its own color</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full rounded-lg"/>
              <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>
                <p>Each pixel takes the color of whichever concept scored highest there, as long as that score clears 0.3 relative to that concept&apos;s own peak (not 0.5 â€” a stricter cutoff was tested and only marked each concept&apos;s single hottest pixel, missing real lesion area).</p>
                <div className="flex flex-wrap gap-x-3 gap-y-1">{Object.entries(r.lesion_detections||{}).map(([name,info])=>(
                  <span key={name} className="flex items-center gap-1 text-[10px]"><span style={{background:info.color}}/>{name}</span>))}</div>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch C: DME parallel model */}
        <div style={{background:"#ffffff",border:"1px solid #10b98140"}}>
          <div className="text-xs text-emerald-300 uppercase tracking-wider mb-3">â†³ DME branch (parallel model) â€” a second disease, click to see how</div>
          <div className="flex items-stretch gap-1 overflow-x-auto pb-2">
            <PipelineNode title="Same Image" spec="Identical preprocessing" accent="#10b981" selected={selectedStage==="dme_input"} onClick={()=>setSelectedStage(selectedStage==="dme_input"?null:"dme_input")}>
              <img src={`data:image/png;base64,${r.original_b64}`} className="w-full h-16 object-cover rounded"/>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="2nd ViT+VQC Model" spec={p.dme?.label} accent="#10b981" selected={selectedStage==="dme_model"} onClick={()=>setSelectedStage(selectedStage==="dme_model"?null:"dme_model")}>
              <div className="text-[10px] text-gray-500">Separately trained weights</div>
            </PipelineNode>
            <PipelineArrow/>
            <PipelineNode title="DME Output" accent={r.dme_detected?"#ef4444":"#10b981"} selected={selectedStage==="dme_output"} onClick={()=>setSelectedStage(selectedStage==="dme_output"?null:"dme_output")}>
              <div style={{color:r.dme_detected?"#ef4444":"#10b981"}}>{r.dme_detected?"Detected":"Not detected"}</div>
              <div className="text-[10px] text-gray-500">{r.dme_confidence}% confidence</div>
            </PipelineNode>
          </div>
          {selectedStage==="dme_model"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #10b98120"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>2nd ViT+VQC Model: same architecture, different job</div>
            <p style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>This is a completely separate set of trained weights running the exact same ViT â†’ Reducer â†’ 4-qubit VQC architecture shown at the top of this page â€” the only structural difference is the final head outputs 1 number (DME present/absent) instead of 5 (DR grade). It was trained on a combined Messidor + IDRiD dataset, since neither alone had enough positive macular-edema cases on its own.</p>
          </div>)}
          {selectedStage==="dme_output"&&(<div style={{marginTop:20,paddingTop:20,borderTop:"1px solid #10b98120"}}>
            <div style={{fontSize:14,fontWeight:700,color:"#fff",marginBottom:10}}>DME Output: sigmoid probability against a 0.5 threshold</div>
            <Formula>p = 1 / (1 + e^(âˆ’z)) &nbsp;â†’&nbsp; detected if p â‰¥ 0.5</Formula>
            <div className="max-w-md">
              <div className="flex justify-between text-[10px] text-gray-500 mb-1"><span>0%</span><span>Raw probability: {((p.dme?.probability||0)*100).toFixed(1)}%</span><span>100%</span></div>
              <div className="h-3 rounded-full bg-slate-800 relative overflow-hidden">
                <div style={{width:`${(p.dme?.probability||0)*100}%`,background:r.dme_detected?"#ef4444":"#10b981"}}/>
                <div style={{left:"50%"}}/>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch D: Red-free classical view */}
        <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
          <div className="text-xs text-amber-300 uppercase tracking-wider mb-2">â†³ Red-free view â€” classical enhancement, not part of the AI</div>
          <div className="grid grid-cols-2 gap-4">
            <img src={`data:image/png;base64,${r.redfree_b64||r.original_b64}`} className="w-full rounded-lg"/>
            <div className="text-xs text-gray-500 space-y-2 flex flex-col justify-center">
              <p>This is a plain, non-AI image-processing view â€” the green color channel extracted and contrast-enhanced (CLAHE) â€” the same style of view retinal specialists traditionally read hemorrhages and microaneurysms from as high-contrast dark spots.</p>
              <p>It is computed for reference only and is <span className="text-gray-700">not</span> what any of the models above actually see â€” the ViT backbone is fed the plain resized + normalized RGB image from the Preprocessing stage.</p>
            </div>
          </div>
        </div>

        <div style={{background:"#111128",border:"1px solid #ffffff10",padding:16}}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-medium">Input vs. output comparison</div>
            {r.marked_fraction!==undefined&&<div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Difference: <span className="text-purple-300 font-semibold">{r.marked_fraction}%</span> of visible retina marked</div>}
          </div>
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div><div className="text-xs text-gray-500 mb-2 text-center">Input (unmodified)</div>
              <img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/></div>
            <div><div className="text-xs text-gray-500 mb-2 text-center">Output (lesion-marked)</div>
              <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full rounded-lg"/></div>
          </div>
          <div className="text-xs text-gray-400 mb-2">What changed:</div>
          <div className="space-y-1">
            {Object.entries(r.lesion_detections||{}).filter(([,info])=>info.detected).map(([name,info])=>(
              <div key={name} className="flex items-center gap-2 text-xs">
                <span style={{background:info.color}}/>
                <span className="text-gray-700">{name} marked</span>
                <span className="text-gray-400">({info.confidence}% confidence)</span>
              </div>
            ))}
            {Object.values(r.lesion_detections||{}).every(info=>!info.detected)&&
              <div className="text-xs text-gray-400">No lesions crossed the detection threshold - output matches input with no markings.</div>}
          </div>
        </div>
        <button onClick={()=>{setResult(null);setSelectedScan(null);setPreview(null)}}
          className="px-4 py-2 text-xs rounded-lg bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 transition">
          Drop another image
        </button>
      </>)}
    </div>);
  };

  const renderReports=()=>(<div style={{padding:"32px 36px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:16}}><div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>SCAN HISTORY</div><h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:"0 0 4px"}}>Reports</h2>
    <p style={{fontSize:12,color:"#9b89c4",marginBottom:12}}>Click any scan to view full results</p>
    {scanHistory.length===0?<div style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:16,padding:48,textAlign:"center",color:"#9b89c4",fontSize:14}}>No scans yet. Upload a retinal image to get started.</div>
    :<div className="space-y-2">{scanHistory.map((scan,i)=>(
      <div key={i} onClick={()=>{setSelectedScan(scan);setResult(scan);setPage("dashboard")}} style={{background:"#111128",border:"1px solid #ffffff10",borderRadius:14,padding:16,display:"flex",alignItems:"center",justifyContent:"space-between",cursor:"pointer",transition:"all 0.2s"}} onMouseEnter={e=>(e.currentTarget.style.borderColor="#a855f740")} onMouseLeave={e=>(e.currentTarget.style.borderColor="#ffffff10")}>
        <div className="flex items-center gap-4"><img src={`data:image/png;base64,${scan.original_b64}`} className="w-14 h-14 rounded-lg object-cover"/>
          <div><div style={{fontSize:14,fontWeight:700,color:GRADE_COLORS[scan.grade]}}>{scan.grade_label}{scan.dme_detected?" + DME":""}</div>
            <div style={{fontSize:13,color:"#9b89c4",lineHeight:1.7}}>Confidence: {(scan.confidence*100).toFixed(1)}% | {scan.model_type==="hybrid"?"Quantum":"Classical"}</div></div></div>
        <div className="text-xs text-gray-400">{scan.inference_time}s</div>
      </div>))}</div>}</div>);

  const renderExplainability=()=>(<div style={{padding:"32px 36px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:16}}><div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>XAI</div><h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:"0 0 4px"}}>Explainability</h2>
    <p style={{fontSize:14,color:"#9b89c4",marginBottom:8}}>QuantumRetina-X uses two methods so doctors can trust the AI:</p>
    <div className="grid grid-cols-2 gap-4">
      <Card title="Grad-CAM ? WHERE the AI looked" icon={<Eye size={14} className="text-red-400"/>}>
        <p style={{fontSize:12,color:"#9b89c4",lineHeight:1.7,marginBottom:12}}>Creates a heatmap overlaid on the retina. Red = AI focused here. Blue = AI ignored this region. For diseased eyes, red should land on lesions.</p>
        <p style={{fontSize:12,color:"#c4b5fd"}}>Upload an image to see live Grad-CAM heatmaps.</p>
      </Card>
      <Card title="TCAV ? WHY the AI chose that grade" icon={<Brain size={14} className="text-purple-400"/>}>
        <p style={{fontSize:12,color:"#9b89c4",lineHeight:1.7,marginBottom:12}}>Measures which clinical concepts drove the diagnosis:</p>
        <div style={{display:"flex",flexDirection:"column",gap:6,marginBottom:12,fontSize:12,color:"#c4b5fd"}}>
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-400"/>Hemorrhage ? blood leaking from vessels</div>
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-amber-400"/>Exudate ? fatty deposits</div>
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-400"/>Edema ? fluid swelling (also indicates DME)</div>
        </div>
        <div style={{fontSize:12,fontWeight:600,color:"#e2d9f3",marginBottom:8}}>Scores increase with severity:</div>
        {Object.entries(TCAV_REF).map(([concept,scores])=>(<div key={concept} className="mb-3">
          <div style={{fontSize:11,color:"#9b89c4",marginBottom:4,textTransform:"capitalize"}}>{concept}</div>
          <div className="flex gap-1">{Object.entries(scores).map(([g,s])=>(
            <div key={g} className="flex-1 text-center"><div style={{height:48,borderRadius:4,position:"relative",background:"#1f1540",display:"flex",alignItems:"flex-end"}}><div style={{height:`${Number(s)*100}%`,background:Number(s)>0.5?"#8b5cf6":"#e2e8f0"}}/></div><span className="text-[10px] text-gray-400">G{g}</span></div>
          ))}</div></div>))}
      </Card>
    </div></div>);

  const renderInsights=()=>(<div style={{padding:"32px 36px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:16}}><div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>PERFORMANCE</div><h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:"0 0 4px"}}>Model Insights</h2>
    <div className="ql-stats-grid grid grid-cols-4 gap-4">
      <StatBox label="Best accuracy" value="81.6%" sub="ViT+VQC on APTOS" color="#8b5cf6"/>
      <StatBox label="Best kappa" value="0.881" sub="ViT baseline" color="#3b82f6"/>
      <StatBox label="G1-sensitivity" value="78.6%" sub="Early detection" color="#f59e0b"/>
      <StatBox label="Datasets" value="4" sub="3 countries" color="#10b981"/>
    </div>
    <div className="grid grid-cols-2 gap-4">
      <Card title="Ablation study" icon={<Activity size={14} className="text-purple-400"/>}>
        {ABLATION.map(row=>(<div key={row.config} style={{background:"#111128",border:"1px solid #ffffff10"}}>
          <span style={{fontSize:12,fontWeight:600,color:row.color}}>{row.config}</span>
          <div style={{fontSize:11}}><span style={{color:"#9b89c4"}}>Acc: <span style={{color:"#fff"}}>{row.accuracy}%</span></span><span className="text-gray-500">F1: <span className="text-emerald-300">{row.f1}</span></span><span className="text-gray-500">AUC: <span className="text-sky-300">{row.auc}</span></span><span className="text-gray-500">Îº: <span className="text-gray-800">{row.kappa}</span></span><span style={{color:"#9b89c4"}}>G1: <span style={{color:"#fff"}}>{row.g1}%</span></span></div>
        </div>))}</Card>
      <Card title="Cross-dataset generalisation" icon={<Shield size={14} className="text-purple-400"/>}>
        {CROSS_DATASET.map(row=>(<div key={row.dataset} style={{background:"#111128",border:"1px solid #ffffff10"}}>
          <div className="flex items-center justify-between mb-1"><span style={{fontSize:12,fontWeight:600,color:"#e2d9f3"}}>{row.dataset}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${row.role==="Trained"?"bg-purple-500/20 text-purple-300":"bg-slate-700 text-gray-500"}`}>{row.role}</span></div>
          <div className="flex gap-3 text-xs"><span className="text-blue-400">ViT:{row.vit_acc}% / AUC:{row.vit_auc}</span><span className="text-purple-400">VQC:{row.vqc_acc}% / AUC:{row.vqc_auc}</span><span className="text-amber-400">G1:{row.vqc_g1}%</span></div>
        </div>))}</Card>
    </div></div>);

  const renderSettings=()=>(<div style={{padding:"32px 36px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:16}}><div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>CONFIGURATION</div><h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:"0 0 4px"}}>Settings</h2>
    <Card title="Model configuration" icon={<Settings size={14} className="text-purple-400"/>}>
      <div className="space-y-4">
        <div><label style={{fontSize:12,color:"#9b89c4",display:"block",marginBottom:6}}>Default model</label>
          <select value={modelType} onChange={e=>setModelType(e.target.value as any)} style={{background:"#1a1040",color:"#e2d9f3",fontSize:13,padding:"10px 14px",borderRadius:8,border:"1.5px solid #a855f740",outline:"none",width:"100%",cursor:"pointer"}}>
            <option value="hybrid">ViT + VQC (Quantum)</option><option value="vit">ViT Only (Classical)</option></select></div>
        <div><label style={{fontSize:12,color:"#9b89c4",display:"block",marginBottom:6}}>Heatmap opacity</label>
          <input type="range" min={0} max={100} value={heatmapOpacity} onChange={e=>setHeatmapOpacity(Number(e.target.value))} className="w-full accent-blue-500"/><span style={{fontSize:12,color:"#9b89c4"}}>{heatmapOpacity}%</span></div>
      </div></Card>
    <Card title="System information" icon={<Info size={14} className="text-purple-400"/>}>
      <div style={{display:"flex",flexDirection:"column",gap:0}}>{[["Backend","FastAPI + PyTorch + CUDA"],["Backbone","ViT-Base/16"],["Quantum","4-qubit VQC (PennyLane)"],["XAI","Grad-CAM + TCAV"],["Detection","DR (5-grade) + DME"],["Training","APTOS 2019"],["Validation","Messidor, IDRiD, EyePACS"]].map(([k,v])=>(
        <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:12,color:"#9b89c4",padding:"6px 0",borderBottom:"1px solid #ffffff05"}}><span>{k}</span><span style={{color:"#e2d9f3",fontWeight:500}}>{v}</span></div>))}</div></Card></div>);

  const renderAbout=()=>(<div style={{padding:"32px 36px",background:"#070711",minHeight:"100%",display:"flex",flexDirection:"column",gap:16}}><div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:4}}>PROJECT</div><h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:"0 0 4px"}}>About QuantumRetina-X</h2>
    <Card title="Project overview" icon={<Zap size={14} className="text-purple-400"/>}>
      <p style={{fontSize:14,color:"#c4b5fd",lineHeight:1.7,marginBottom:12}}>QuantumRetina-X is an explainable hybrid quantum-classical Vision Transformer for <strong className="text-purple-300">early detection</strong> of Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME).</p>
      <p style={{fontSize:14,color:"#9b89c4",lineHeight:1.7}}>It combines ViT with a 4-qubit VQC in a 16D Hilbert space, plus dual XAI (Grad-CAM + TCAV) for clinically interpretable diagnostics.</p></Card>
    <Card title="6 research contributions" icon={<CheckCircle size={14} className="text-green-400"/>}>
      <div style={{display:"flex",flexDirection:"column",gap:10,fontSize:13,color:"#9b89c4"}}>{["First ViT + VQC for DR and DME early detection","Quantum latent space for micro-lesion detection","Dual XAI (Grad-CAM + TCAV) in quantum medical model","Four-dataset cross-validation (3 countries)","Low-data quantum learning analysis","IoMT-ready lightweight architecture"].map((c,i)=>(
        <div key={i} className="flex items-start gap-2"><span className="text-purple-400 font-bold">{i+1}.</span><span>{c}</span></div>))}</div></Card>
    <div style={{textAlign:"center",fontSize:12,color:"#7c6ea0",marginTop:16}}>QuantumRetina-X v1.0 ? 2026</div></div>);

  return (
    <div className="flex h-screen overflow-hidden" style={{background:"#070711"}}>
      <div className="ql-sidebar w-[220px] flex flex-col justify-between py-4 px-3 flex-shrink-0" style={{background:"#0a0a1a",borderRight:"1px solid #ffffff10"}}>
        <div>
          <div className="flex items-center gap-2 px-2 mb-6">
            <div style={{width:28,height:28,borderRadius:"50%",background:"radial-gradient(circle at 35% 30%,#c4b5fd,#7c3aed,#3b0764)",boxShadow:"0 0 10px #a855f760",flexShrink:0}}/>
            <span className="ql-logo-text" style={{fontWeight:800,fontSize:15,color:"#fff",letterSpacing:-0.3}}>Quantum<span style={{color:"#a855f7"}}>Retina-X</span></span>
          </div>
          {SIDEBAR.map(item=>(<div key={item.label} onClick={()=>{setPage(item.page);setSelectedScan(null)}}
            style={{display:"flex",alignItems:"center",gap:10,padding:"10px 12px",borderRadius:10,marginBottom:2,cursor:"pointer",fontSize:13,fontWeight:500,transition:"all 0.15s",
              background:page===item.page?"rgba(168,85,247,0.15)":"transparent",
              color:page===item.page?"#e2d9f3":"rgba(196,181,253,0.5)",
              border:page===item.page?"1px solid rgba(168,85,247,0.3)":"1px solid transparent"}}>
            <item.icon size={16}/><span className="ql-sidebar-label">{item.label}</span></div>))}
        </div>
        <div>
          <div style={{display:"flex",alignItems:"center",gap:10,padding:"10px 12px",borderRadius:10,marginBottom:8,background:"#111128",border:"1px solid #ffffff10"}}>
            <div style={{width:32,height:32,borderRadius:"50%",background:"rgba(168,85,247,0.2)",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
              <User size={15} style={{color:"#c4b5fd"}}/>
            </div>
            <div className="ql-user-info" style={{flex:1,minWidth:0}}>
              <div style={{fontSize:12,fontWeight:600,color:"#e2d9f3"}}>Dr. Admin</div>
              <div style={{fontSize:10,color:"#7c6ea0"}}>Ophthalmologist</div>
            </div>
            <ChevronDown size={12} style={{color:"#7c6ea0",flexShrink:0}}/>
          </div>
          <div className="ql-brand-card" style={{padding:"12px 14px",borderRadius:10,background:"linear-gradient(135deg,#1a0533,#2d1060)",border:"1px solid #a855f720"}}>
            <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:4}}>
              <Zap size={12} style={{color:"#a855f7"}}/>
              <span style={{fontSize:12,fontWeight:600,color:"#c4b5fd"}}>Quantum + AI</span>
            </div>
            <p style={{fontSize:11,color:"#9b89c4",lineHeight:1.5,margin:0}}>Smarter. Earlier. Explainable.<br/>Better Vision for All.</p>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto flex flex-col" style={{background:"#070711"}}>
        <div style={{background:"#0a0a1a",borderBottom:"1px solid #a855f720",padding:"14px 24px",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <CheckCircle size={20} style={{color:"#a855f7"}}/>
            <div>
              <div style={{fontSize:15,fontWeight:700,color:"#ffffff"}}>Analysis Results</div>
              <div style={{fontSize:12,color:"#9b89c4",marginTop:2}}>AI-powered retinal image analysis with explainability</div>
            </div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:24}}>
            <div>
              <div style={{fontSize:11,color:"#9b89c4",marginBottom:2}}>Scan Date</div>
              <div style={{fontSize:13,fontWeight:600,color:"#e2d9f3"}}>{new Date().toLocaleDateString()}</div>
            </div>
            <div>
              <div style={{fontSize:11,color:"#9b89c4",marginBottom:2}}>Model</div>
              <div style={{fontSize:13,fontWeight:600,color:"#e2d9f3"}}>QuantumRetina-X v1.0</div>
            </div>
            <select value={modelType} onChange={e=>setModelType(e.target.value as any)}
              style={{background:"#1a1040",color:"#e2d9f3",fontSize:12,padding:"8px 14px",borderRadius:8,border:"1.5px solid #a855f740",outline:"none",cursor:"pointer",fontWeight:500}}>
              <option value="hybrid">ViT + VQC (Quantum)</option>
              <option value="vit">ViT Only</option>
            </select>
          </div>
        </div>
        {page==="dashboard"&&renderDashboard()}
        {page==="upload"&&(
    <div style={{padding:"36px 36px",background:"#070711",minHeight:"100%"}}>
      <div style={{marginBottom:28}}>
        <div style={{fontSize:11,fontWeight:700,letterSpacing:"1.2px",textTransform:"uppercase",color:"#a855f7",marginBottom:8}}>NEW SCAN</div>
        <h2 style={{fontSize:28,fontWeight:800,color:"#fff",letterSpacing:-0.5,margin:0}}>Upload Fundus Image</h2>
        <p style={{fontSize:14,color:"#9b89c4",marginTop:8}}>Upload a retinal fundus photograph to run DR grading, DME detection, and XAI analysis.</p>
      </div>
      {renderUploadZone()}
      <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16,marginTop:24}}>
        {[["📷","PNG or JPEG","224×224 or larger, colour fundus photo"],["⚡","~3 seconds","Full pipeline: DR + DME + Grad-CAM + TCAV"],["🔒","Local only","Image processed on-device, not stored"]].map(([icon,title,desc])=>(
          <div key={title} style={{background:"#111128",border:"1px solid #ffffff08",borderRadius:14,padding:"20px 22px"}}>
            <div style={{fontSize:28,marginBottom:10}}>{icon}</div>
            <div style={{fontSize:14,fontWeight:700,color:"#e2d9f3",marginBottom:4}}>{title}</div>
            <div style={{fontSize:12,color:"#9b89c4",lineHeight:1.5}}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  )}
        {page==="pipeline"&&renderPipeline()}
        {page==="reports"&&renderReports()}
        {page==="explainability"&&renderExplainability()}
        {page==="twin"&&<DigitalTwin result={r}/>}
        {page==="insights"&&renderInsights()}
        {page==="settings"&&renderSettings()}
        {page==="about"&&renderAbout()}
        {r&&page==="dashboard"&&(
          <div style={{background:"#0a0a1a",borderTop:"1px solid #ffffff10",padding:"12px 24px",display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0}}>
            <div style={{display:"flex",alignItems:"center",gap:32}}>
              <div><div style={{fontSize:11,color:"#9b89c4",marginBottom:2}}>Prediction</div><div style={{fontSize:13,fontWeight:700,color:GRADE_COLORS[r.grade]}}>{r.grade_label}{r.dme_detected?" + DME":""}</div></div>
              <div><div style={{fontSize:11,color:"#9b89c4",marginBottom:2}}>Confidence</div><div style={{fontSize:13,fontWeight:700,color:"#e2d9f3"}}>{(r.confidence*100).toFixed(1)}%</div></div>
              <div><div style={{fontSize:11,color:"#9b89c4",marginBottom:2}}>Status</div><div style={{fontSize:13,fontWeight:600,color:"#f59e0b",display:"flex",alignItems:"center",gap:4}}><AlertTriangle size={12}/> Requires clinical review</div></div>
            </div>
            <div style={{display:"flex",gap:8}}>
              <button onClick={()=>{setResult(null);setSelectedScan(null);setPreview(null)}} style={{padding:"8px 16px",fontSize:12,borderRadius:8,background:"#111128",border:"1px solid #ffffff15",color:"#c4b5fd",cursor:"pointer"}}>New scan</button>
              <button onClick={downloadHeatmap} style={{padding:"8px 16px",fontSize:12,borderRadius:8,background:"#111128",border:"1px solid #ffffff15",color:"#c4b5fd",cursor:"pointer",display:"flex",alignItems:"center",gap:6}}><Download size={13}/> Heatmap</button>
              <button onClick={downloadReport} style={{padding:"8px 16px",fontSize:12,borderRadius:8,background:"#7c3aed",border:"none",color:"#fff",cursor:"pointer",display:"flex",alignItems:"center",gap:6}}><FileText size={13}/> Report</button>
            </div>
          </div>)}
        {error&&<div style={{background:"#2d0a0a",border:"1px solid #ef444440",borderRadius:10,padding:"10px 16px",margin:"8px 20px",fontSize:13,color:"#fca5a5"}}>Error: {error}</div>}
      </div>
    </div>
  );
}

