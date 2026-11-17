import { useState, useRef, useCallback } from "react";
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

type Page = "dashboard"|"upload"|"pipeline"|"reports"|"explainability"|"insights"|"settings"|"about";
const SIDEBAR: {icon:any;label:string;page:Page}[] = [
  {icon:Activity,label:"Dashboard",page:"dashboard"},{icon:Upload,label:"Upload Image",page:"upload"},
  {icon:Zap,label:"Pipeline",page:"pipeline"},
  {icon:FileText,label:"Reports",page:"reports"},{icon:Eye,label:"Explainability",page:"explainability"},
  {icon:Brain,label:"Model Insights",page:"insights"},{icon:Settings,label:"Settings",page:"settings"},
  {icon:Info,label:"About",page:"about"},
];

function Card({title,children,icon}:{title:string;children:React.ReactNode;icon?:React.ReactNode}) {
  return <div className="rounded-xl p-5" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}><div className="text-sm font-medium mb-4 flex items-center gap-2">{icon}{title}</div>{children}</div>;
}
function StatBox({label,value,sub,color}:{label:string;value:string;sub?:string;color?:string}) {
  return <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}><div className="text-xs text-gray-500 mb-1">{label}</div><div className="text-2xl font-bold" style={{color:color||"#1e293b"}}>{value}</div>{sub&&<div className="text-xs text-gray-400 mt-1">{sub}</div>}</div>;
}

function LesionPanel({lesions}:{lesions?:Record<string,LesionInfo>}) {
  const entries = Object.entries(lesions||{});
  return (
    <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
      <div className="text-sm font-medium mb-3 flex items-center gap-2"><Eye size={14} className="text-purple-400"/>Lesion detection</div>
      <div className="space-y-2">{entries.length===0&&<div className="text-xs text-gray-400">No lesion detectors loaded.</div>}
        {entries.map(([name,info])=>(
        <div key={name} className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{background:info.color}}/>
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
    <div className="rounded-xl p-8 flex flex-col items-center justify-center min-h-[350px] cursor-pointer transition-all"
      style={{background:dragOver?"#eff6ff":"#ffffff",border:dragOver?"2px dashed #3b82f6":"2px dashed #cbd5e1"}}
      onDragOver={e=>{e.preventDefault();setDragOver(true)}} onDragLeave={()=>setDragOver(false)} onDrop={e=>onDrop(e,destPage)} onClick={()=>fileRef.current?.click()}>
      <Upload size={48} className="text-blue-500 mb-4"/>
      <p className="text-lg font-medium mb-1">Drop retinal fundus image here</p>
      <p className="text-sm text-gray-500 mb-4">or click to browse ? PNG, JPG supported</p>
      <button className="px-5 py-2 bg-purple-600 rounded-lg text-sm font-medium hover:bg-purple-500 transition">Select Image</button>
      <input ref={fileRef} type="file" className="hidden" accept="image/*" onChange={e=>{if(e.target.files?.[0])handleFile(e.target.files[0],destPage)}}/>
    </div>
  );

  const renderDashboard=()=>(
    <div className="p-5 grid gap-4" style={{gridTemplateColumns:r?"1fr 340px":"1fr"}}>
      <div className="space-y-4">
        {!r&&!loading&&renderUploadZone()}
        {loading&&(
          <div className="rounded-xl p-8 flex flex-col items-center justify-center min-h-[350px]" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            {preview&&<img src={preview} className="w-40 h-40 rounded-xl object-cover mb-6 opacity-60"/>}
            <div className="w-full max-w-md">{steps.map((s,i)=>(
              <div key={i} className={`flex items-center gap-3 py-2 text-sm ${i<=step?"text-purple-300":"text-gray-300"}`}>
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${i<step?"bg-blue-600 text-white":i===step?"bg-blue-100 text-blue-600 animate-pulse":"bg-slate-800 text-gray-300"}`}>{i<step?"?":i+1}</div>{s}
              </div>))}</div>
          </div>)}
        {r&&(<>
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="grid grid-cols-3 gap-4">
              <div><div className="text-sm text-gray-500 mb-2 text-center">Original retinal image</div><img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/></div>
              <div><div className="text-sm text-gray-500 mb-2 text-center">Grad-CAM heatmap (AI focus)</div>
                <div className="relative"><img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/><img src={`data:image/png;base64,${r.heatmap_b64}`} className="w-full rounded-lg absolute inset-0" style={{opacity:heatmapOpacity/100}}/></div></div>
              <div><div className="text-sm text-gray-500 mb-2 text-center">Lesion-type map</div>
                <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full rounded-lg"/></div>
            </div>
            <div className="flex items-center gap-3 mt-3 px-2"><span className="text-xs text-gray-500">Heatmap</span><input type="range" min={0} max={100} value={heatmapOpacity} onChange={e=>setHeatmapOpacity(Number(e.target.value))} className="flex-1 accent-blue-500"/><span className="text-xs text-gray-500 w-10 text-right">{heatmapOpacity}%</span></div>
            <div className="flex justify-center flex-wrap gap-x-5 gap-y-1 mt-2 text-xs text-gray-500">
              {Object.entries(r.lesion_detections||{}).map(([name,info])=>(
                <span key={name} className="flex items-center gap-1"><span className="w-3 h-3 rounded-full inline-block" style={{background:info.color}}/> {name}</span>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <LesionPanel lesions={r.lesion_detections}/>
            <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
              <div className="text-sm font-medium mb-3">TCAV concept scores</div>
              {tcavData.map(t=>(<div key={t.name} className="mb-2">
                <div className="flex justify-between text-xs mb-1"><span className="text-gray-700">{t.name} impact</span><span className="text-gray-800 font-medium">{t.value}</span></div>
                <div className="h-2 rounded-full bg-slate-800"><div className="h-full rounded-full bg-purple-500 transition-all" style={{width:`${t.value*100}%`}}/></div>
              </div>))}
            </div>
          </div>
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium mb-3">Grade probability distribution</div>
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={probData}><XAxis dataKey="name" tick={{fill:"#64748b",fontSize:11}}/><YAxis tick={{fill:"#64748b",fontSize:11}} domain={[0,100]}/>
                <Tooltip contentStyle={{background:"#ffffff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:12}}/>
                <Bar dataKey="value" radius={[4,4,0,0]}>{probData.map((d,i)=><Cell key={i} fill={d.grade===r.grade?GRADE_COLORS[r.grade]:"#e2e8f0"}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>)}
      </div>
      {r&&(
        <div className="space-y-4">
          {/* DR Diagnosis */}
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Diagnosis summary</div>
            <div className="flex items-center gap-3 mb-1">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{background:"#1e293b"}}><Activity size={20} className="text-gray-700"/></div>
              <div><div className="text-xs text-gray-500">Diabetic Retinopathy (DR)</div><div className="text-xl font-bold" style={{color:GRADE_COLORS[r.grade]}}>Grade {r.grade}</div></div>
              <span className="ml-auto text-[10px] px-2 py-1 rounded-full font-medium" style={{background:GRADE_COLORS[r.grade]+"20",color:GRADE_COLORS[r.grade]}}>{r.dr_type||"NPDR"}</span>
            </div>
            <div className="flex justify-between items-center mt-2"><span className="text-xs text-gray-500">Confidence</span><span className="text-xl font-bold" style={{color:GRADE_COLORS[r.grade]}}>{(r.confidence*100).toFixed(1)}%</span></div>
            <div className="h-1.5 rounded-full bg-slate-800 mt-1"><div className="h-full rounded-full" style={{width:`${r.confidence*100}%`,background:GRADE_COLORS[r.grade]}}/></div>
          </div>

          {/* DME Detection */}
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl flex items-center justify-center" style={{background:"#1e293b"}}><Droplets size={20} className="text-blue-400"/></div>
              <div>
                <div className="text-xs text-gray-500">Diabetic Macular Edema (DME)</div>
                <div className={`text-xl font-bold ${r.dme_detected?"text-red-400":"text-green-400"}`}>{r.dme_detected?"Detected":"Not detected"}</div>
              </div>
              <span className={`ml-auto text-[10px] px-2 py-1 rounded-full font-medium ${r.dme_detected?"bg-red-500/20 text-red-400":"bg-green-500/20 text-green-400"}`}>{r.dme_detected?"Present":"Absent"}</span>
            </div>
            <div className="flex justify-between items-center mt-3"><span className="text-xs text-gray-500">Confidence</span><span className={`text-lg font-bold ${r.dme_detected?"text-red-400":"text-green-400"}`}>{r.dme_confidence}%</span></div>
          </div>

          {/* Early Detection Alert */}
          {r.early_detection&&(
            <div className="rounded-xl p-3 flex items-center gap-3" style={{background:"#f59e0b15",border:"1px solid #f59e0b40"}}>
              <AlertTriangle size={20} className="text-amber-400"/>
              <div><div className="text-sm font-medium text-amber-300">Early detection alert</div><div className="text-xs text-amber-400/70">Mild DR detected ? early intervention can prevent blindness</div></div>
            </div>
          )}

          {/* Risk Assessment */}
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium mb-2">Risk assessment</div>
            <span className="text-xs px-3 py-1 rounded-full font-medium" style={{background:RISK_COLORS[r.risk_level]+"20",color:RISK_COLORS[r.risk_level]}}>{r.risk_level}</span>
            <div className="text-xs text-gray-500 leading-relaxed mt-3">{r.recommendation}</div>
          </div>

          {/* AI Explanation */}
          <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
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

          <div className="rounded-xl p-3 text-xs text-gray-400" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="flex justify-between"><span>Model</span><span className="text-gray-700">{r.model_type==="hybrid"?"ViT + VQC (Quantum)":"ViT Only"}</span></div>
            <div className="flex justify-between mt-1"><span>Inference</span><span className="text-gray-700">{r.inference_time}s</span></div>
          </div>
        </div>
      )}
    </div>
  );

  const PipelineNode=({title,spec,children,accent,onClick,selected}:{title:string;spec?:string;children?:React.ReactNode;accent?:string;onClick?:()=>void;selected?:boolean})=>(
    <div onClick={onClick} className={`rounded-xl p-3 flex-1 min-w-[150px] transition-all ${onClick?"cursor-pointer hover:brightness-125":""}`}
      style={{background:selected?"#eff6ff":"#ffffff",border:`1px solid ${selected?"#a855f7":(accent||"#1e293b")}`,boxShadow:selected?"0 0 0 2px #a855f740":"none"}}>
      <div className="text-xs font-semibold text-gray-800 flex items-center justify-between">{title}{onClick&&<span className="text-[9px] text-purple-400">click â“˜</span>}</div>
      {spec&&<div className="text-[10px] text-gray-400 mb-2">{spec}</div>}
      {children}
    </div>
  );
  const PipelineArrow=()=>(<div className="flex items-center justify-center text-purple-400 px-1 shrink-0">â†’</div>);
  const Formula=({children}:{children:React.ReactNode})=>(
    <div className="rounded-lg px-3 py-2 mb-2 font-mono text-xs text-blue-700 overflow-x-auto" style={{background:"#1e3a8a",border:"1px solid #2d2050"}}>{children}</div>
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
    <div className="rounded-xl p-6" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
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
    return (<div className="p-5 space-y-4">
      <h2 className="text-lg font-semibold mb-1">Model pipeline</h2>
      <p className="text-sm text-gray-500 mb-4">Drop an image below to watch it pass through the real model, stage by stage.</p>
      {loading?renderPipelineLoading():!r||!p?(
        renderUploadZone("pipeline")
      ):(<>
        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
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
              <div className="text-[10px] text-gray-500">â€–featuresâ€– = {p.backbone.cls_norm}</div>
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
              <div className="text-sm font-bold" style={{color:GRADE_COLORS[r.grade]}}>{r.grade_label}</div>
              <div className="text-[10px] text-gray-500">{(r.confidence*100).toFixed(1)}% confidence</div>
            </PipelineNode>
          </div>

          {selectedStage&&(
            <div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
              {selectedStage==="input"&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">Input: the raw image tensor</div>
                <div className="grid grid-cols-2 gap-4">
                  <img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/>
                  <div className="text-xs text-gray-500 space-y-2">
                    <Formula>x âˆˆ â„^(224 Ã— 224 Ã— 3), x[i,j,c] âˆˆ [0, 255]</Formula>
                    <p>The uploaded photo is resized to 224Ã—224 with 3 color channels (R, G, B) â€” a plain integer pixel array. Nothing model-specific has happened yet; this is exactly what you uploaded, reformatted for the network.</p>
                  </div>
                </div>
              </div>)}
              {selectedStage==="preprocessing"&&p.preprocessing&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">Preprocessing: cleaning the pixels before the model sees them</div>
                <Formula>x&apos; = (x / 255 âˆ’ Î¼) / Ïƒ,  Î¼ = [0.485, 0.456, 0.406],  Ïƒ = [0.229, 0.224, 0.225]</Formula>
                <p className="text-xs text-gray-500 mb-3">Every pixel is scaled to [0,1] then normalized using the same per-channel mean/std the ViT backbone was pretrained on (ImageNet statistics) â€” this isn&apos;t a cosmetic step, it matches the input distribution the frozen backbone actually expects. No cropping or CLAHE enhancement is applied here (this app currently sends the plain resized+normalized image to the model â€” CLAHE only exists in the separate red-free display view, not what the model sees).</p>
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
                <div className="text-sm font-medium text-gray-800 mb-2">ViT Backbone: split into patches, read with attention</div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="relative">
                    <img src={`data:image/png;base64,${r.original_b64}`} className="w-full rounded-lg"/>
                    <div className="absolute inset-0 rounded-lg" style={{
                      backgroundImage:"linear-gradient(to right, rgba(59,130,246,0.5) 1px, transparent 1px), linear-gradient(to bottom, rgba(59,130,246,0.5) 1px, transparent 1px)",
                      backgroundSize:"calc(100%/14) calc(100%/14)"}}/>
                  </div>
                  <div className="text-xs text-gray-500 space-y-2">
                    <Formula>Attention(Q,K,V) = softmax(QK^T / âˆšd) V</Formula>
                    <p>The image is cut into a real 14Ã—14 grid of 16Ã—16-pixel patches (196 patches total) â€” shown overlaid on the actual image, not illustrative.</p>
                    <p>Each patch becomes a vector; 12 self-attention layers let every patch "look at" every other patch using the formula above, so a lesion's relationship to the optic disc or macula is captured. The whole 86.4M-parameter backbone is frozen â€” it never updates during training here.</p>
                    <p>Only the [CLS] token â€” a single 768-number summary of the entire image â€” is passed onward. Its magnitude for this image: <span className="text-gray-800 font-mono">â€–hâ€– = {p.backbone.cls_norm}</span></p>
                  </div>
                </div>
              </div>)}
              {selectedStage==="reducer"&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">Reducer: 768 numbers compressed to 4</div>
                <Formula>y = WÂ·h + b,  W âˆˆ â„^(4 Ã— 768), h âˆˆ â„^768 â†’ y âˆˆ â„^4</Formula>
                <div className="text-xs text-gray-500 mb-3">A single trainable linear layer compresses the 768-number [CLS] summary down to 4 numbers â€” the only "bridge" between the classical and quantum parts. This is a 192Ã— compression â€” a real bottleneck (see the audit report for what this costs).</div>
                <div className="space-y-1.5 max-w-md">{(p.reducer.values||[]).map((v,i)=>(
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 w-4">x{i}</span>
                    <div className="flex-1 h-3 rounded-full bg-slate-800 relative overflow-hidden">
                      <div className="h-full rounded-full" style={{width:`${Math.min(Math.abs(v)/3*100,100)}%`,marginLeft:v<0?"auto":0,marginRight:v<0?0:"auto",background:v<0?"#ef4444":"#10b981"}}/>
                    </div>
                    <span className="text-[10px] font-mono text-gray-700 w-12 text-right">{v.toFixed(3)}</span>
                  </div>
                ))}</div>
              </div>)}
              {selectedStage==="vqc"&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">4-Qubit VQC: the actual quantum measurement</div>
                <Formula>|Ïˆáµ¢âŸ© = R_Y(yáµ¢)|0âŸ© &nbsp;â†’&nbsp; Uâ‚‚Uâ‚|ÏˆâŸ© &nbsp;â†’&nbsp; âŸ¨Záµ¢âŸ© = âŸ¨Ïˆ|Záµ¢|ÏˆâŸ©</Formula>
                <div className="text-xs text-gray-500 mb-3">Each of the 4 reducer numbers rotates one qubit (angle encoding, R_Y). Two trainable layers of rotations + CNOT entangling gates (Uâ‚, Uâ‚‚) mix them together in a 16-dimensional Hilbert space. Each qubit is then measured (Pauli-Z expectation âŸ¨Záµ¢âŸ©, range -1 to +1) â€” these are the real measured values for this image:</div>
                <div className="space-y-1.5 max-w-md">{(p.vqc.values||[]).map((v,i)=>(
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-400 w-8">qubit {i}</span>
                    <div className="flex-1 h-3 rounded-full bg-slate-800 relative overflow-hidden">
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600"/>
                      <div className="h-full rounded-full absolute" style={{width:`${Math.abs(v)/2*100}%`,left:v<0?`${50-Math.abs(v)/2*100}%`:"50%",background:"#f59e0b"}}/>
                    </div>
                    <span className="text-[10px] font-mono text-gray-700 w-12 text-right">{v.toFixed(3)}</span>
                  </div>
                ))}</div>
              </div>)}
              {selectedStage==="head"&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">Head: 4 quantum outputs â†’ 5 grade probabilities</div>
                <Formula>z = WÂ·âŸ¨ZâŸ© + b &nbsp;â†’&nbsp; p_c = e^(z_c) / Î£â±¼ e^(z_j)</Formula>
                <div className="text-xs text-gray-500 mb-3">A final linear layer maps the 4 quantum measurements to 5 logits, then softmax turns them into a probability for each DR grade:</div>
                <ResponsiveContainer width="100%" height={140}>
                  <BarChart data={probData}><XAxis dataKey="name" tick={{fill:"#64748b",fontSize:11}}/><YAxis tick={{fill:"#64748b",fontSize:11}} domain={[0,100]}/>
                    <Tooltip contentStyle={{background:"#ffffff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:12}}/>
                    <Bar dataKey="value" radius={[4,4,0,0]}>{probData.map((d,i)=><Cell key={i} fill={d.grade===r.grade?GRADE_COLORS[r.grade]:"#e2e8f0"}/>)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>)}
              {selectedStage==="output"&&(<div>
                <div className="text-sm font-medium text-gray-800 mb-2">Output: final diagnosis</div>
                <div className="text-xs text-gray-500 space-y-1">
                  <p><span className="text-gray-700 font-medium" style={{color:GRADE_COLORS[r.grade]}}>{r.grade_label}</span> â€” {(r.confidence*100).toFixed(1)}% confidence, risk level {r.risk_level}.</p>
                  <p>{r.recommendation}</p>
                  <p>Total inference time: {r.inference_time}s.</p>
                </div>
              </div>)}
            </div>
          )}
        </div>

        {/* Branch A: Grad-CAM explainability pipeline */}
        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #ef444440"}}>
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
          {selectedStage==="fundus_mask"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Fundus Mask: isolating the real eye from the black background</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.mask_viz_b64||r.original_b64}`} className="w-full rounded-lg"/>
              <div className="text-xs text-gray-500 space-y-2">
                <p>The black corners around a fundus photo aren&apos;t retina â€” but the ViT&apos;s LayerNorm blows up activations on these low-variance patches, producing a false &quot;hot ring&quot; with no clinical meaning. This step fits an ellipse to the real eye (Otsu threshold + 18% shrink) so every later stage only looks inside it.</p>
                <p>Red-tinted area (above) = excluded. Untouched area = the {p.fundus_mask?.kept_fraction}% of the frame that Grad-CAM and the lesion maps are allowed to mark.</p>
              </div>
            </div>
          </div>)}
          {selectedStage==="gradcam_backprop"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Grad-CAM Backprop: a second, backward pass through the network</div>
            <Formula>Î±_k = (1/Z) Î£áµ¢â±¼ âˆ‚y/âˆ‚A_ij^k &nbsp;â†’&nbsp; L = ReLU(Î£_k Î±_k A^k)</Formula>
            <p className="text-xs text-gray-500">Unlike the forward pass above (which only computes a prediction), this stage runs a second pass in reverse: the gradient of the predicted grade&apos;s score is traced backward to the ViT&apos;s last encoder layer, reshaped into the 14Ã—14 patch grid. Patches that mattered most for the prediction get the highest weight Î±_k â€” this is what turns into the heatmap on the right.</p>
          </div>)}
          {selectedStage==="heatmap_render"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Heatmap: the raw Grad-CAM signal, stretched and masked for readability</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.heatmap_b64}`} className="w-full rounded-lg"/>
              <div className="text-xs text-gray-500 space-y-2">
                <p>The raw Grad-CAM signal is percentile-stretched (60thâ€“99.5th percentile) so genuine hotspots stand out instead of washing into uniform color, then hard-masked to the fundus mask from the first stage so nothing is ever marked outside the eye.</p>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch B: TCAV concept pipeline */}
        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #3b82f640"}}>
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
          {selectedStage==="concept_scoring"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Concept Scoring: the same [CLS] feature, scored against 4 clinical concepts</div>
            <Formula>P(concept | h) = Ïƒ(wáµ€h + b)</Formula>
            <p className="text-xs text-gray-500 mb-2">The same 768-number ViT feature used for grading is reused here, but scored against 4 separate logistic-regression probes, each trained on real lesion patches cropped from expert-annotated masks (microaneurysm, hemorrhage, hard exudate, macular edema).</p>
            {Object.entries(r.tcav_scores).map(([k,v])=>(<div key={k} className="mb-1.5">
              <div className="flex justify-between text-[10px] text-gray-500"><span className="capitalize">{k}</span><span>{(v*100).toFixed(0)}%</span></div>
              <div className="h-1.5 rounded-full bg-slate-800"><div className="h-full rounded-full bg-blue-500" style={{width:`${v*100}%`}}/></div>
            </div>))}
          </div>)}
          {selectedStage==="concept_maps"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Per-Concept Maps: where each lesion type is, individually, before merging</div>
            <p className="text-xs text-gray-500 mb-3">Each concept gets its own Grad-CAM-style backward pass â€” the concept probe&apos;s own decision score is backpropagated instead of the DR-grade class score, producing 4 independent spatial maps.</p>
            <div className="grid grid-cols-4 gap-2">{Object.entries(r.concept_heatmaps_b64||{}).map(([k,b64])=>(
              <div key={k}><img src={`data:image/png;base64,${b64}`} className="w-full rounded-lg mb-1"/><div className="text-[10px] text-gray-500 text-center capitalize">{k}</div></div>))}</div>
          </div>)}
          {selectedStage==="multicolor_merge"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">Multi-Color Merge: 4 maps combined into one, each lesion type in its own color</div>
            <div className="grid grid-cols-2 gap-4">
              <img src={`data:image/png;base64,${r.lesion_map_b64||r.heatmap_b64}`} className="w-full rounded-lg"/>
              <div className="text-xs text-gray-500 space-y-2">
                <p>Each pixel takes the color of whichever concept scored highest there, as long as that score clears 0.3 relative to that concept&apos;s own peak (not 0.5 â€” a stricter cutoff was tested and only marked each concept&apos;s single hottest pixel, missing real lesion area).</p>
                <div className="flex flex-wrap gap-x-3 gap-y-1">{Object.entries(r.lesion_detections||{}).map(([name,info])=>(
                  <span key={name} className="flex items-center gap-1 text-[10px]"><span className="w-2.5 h-2.5 rounded-full inline-block" style={{background:info.color}}/>{name}</span>))}</div>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch C: DME parallel model */}
        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #10b98140"}}>
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
              <div className="text-sm font-bold" style={{color:r.dme_detected?"#ef4444":"#10b981"}}>{r.dme_detected?"Detected":"Not detected"}</div>
              <div className="text-[10px] text-gray-500">{r.dme_confidence}% confidence</div>
            </PipelineNode>
          </div>
          {selectedStage==="dme_model"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">2nd ViT+VQC Model: same architecture, different job</div>
            <p className="text-xs text-gray-500">This is a completely separate set of trained weights running the exact same ViT â†’ Reducer â†’ 4-qubit VQC architecture shown at the top of this page â€” the only structural difference is the final head outputs 1 number (DME present/absent) instead of 5 (DR grade). It was trained on a combined Messidor + IDRiD dataset, since neither alone had enough positive macular-edema cases on its own.</p>
          </div>)}
          {selectedStage==="dme_output"&&(<div className="mt-4 pt-4" style={{borderTop:"1px solid #e2e8f0"}}>
            <div className="text-sm font-medium text-gray-800 mb-2">DME Output: sigmoid probability against a 0.5 threshold</div>
            <Formula>p = 1 / (1 + e^(âˆ’z)) &nbsp;â†’&nbsp; detected if p â‰¥ 0.5</Formula>
            <div className="max-w-md">
              <div className="flex justify-between text-[10px] text-gray-500 mb-1"><span>0%</span><span>Raw probability: {((p.dme?.probability||0)*100).toFixed(1)}%</span><span>100%</span></div>
              <div className="h-3 rounded-full bg-slate-800 relative overflow-hidden">
                <div className="h-full rounded-full" style={{width:`${(p.dme?.probability||0)*100}%`,background:r.dme_detected?"#ef4444":"#10b981"}}/>
                <div className="absolute top-0 bottom-0 w-px bg-slate-400" style={{left:"50%"}}/>
              </div>
            </div>
          </div>)}
        </div>

        {/* Branch D: Red-free classical view */}
        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
          <div className="text-xs text-amber-300 uppercase tracking-wider mb-2">â†³ Red-free view â€” classical enhancement, not part of the AI</div>
          <div className="grid grid-cols-2 gap-4">
            <img src={`data:image/png;base64,${r.redfree_b64||r.original_b64}`} className="w-full rounded-lg"/>
            <div className="text-xs text-gray-500 space-y-2 flex flex-col justify-center">
              <p>This is a plain, non-AI image-processing view â€” the green color channel extracted and contrast-enhanced (CLAHE) â€” the same style of view retinal specialists traditionally read hemorrhages and microaneurysms from as high-contrast dark spots.</p>
              <p>It is computed for reference only and is <span className="text-gray-700">not</span> what any of the models above actually see â€” the ViT backbone is fed the plain resized + normalized RGB image from the Preprocessing stage.</p>
            </div>
          </div>
        </div>

        <div className="rounded-xl p-4" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-medium">Input vs. output comparison</div>
            {r.marked_fraction!==undefined&&<div className="text-xs text-gray-500">Difference: <span className="text-purple-300 font-semibold">{r.marked_fraction}%</span> of visible retina marked</div>}
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
                <span className="w-2.5 h-2.5 rounded-full inline-block" style={{background:info.color}}/>
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

  const renderReports=()=>(<div className="p-5 space-y-4"><h2 className="text-lg font-semibold mb-4">Scan history</h2>
    <p className="text-xs text-gray-500 mb-2">Click any scan to view full results</p>
    {scanHistory.length===0?<div className="text-sm text-gray-500 rounded-xl p-8 text-center" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>No scans yet. Upload a retinal image to get started.</div>
    :<div className="space-y-2">{scanHistory.map((scan,i)=>(
      <div key={i} onClick={()=>{setSelectedScan(scan);setResult(scan);setPage("dashboard")}} className="rounded-xl p-4 flex items-center justify-between cursor-pointer hover:bg-blue-50 transition" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
        <div className="flex items-center gap-4"><img src={`data:image/png;base64,${scan.original_b64}`} className="w-14 h-14 rounded-lg object-cover"/>
          <div><div className="text-sm font-medium" style={{color:GRADE_COLORS[scan.grade]}}>{scan.grade_label}{scan.dme_detected?" + DME":""}</div>
            <div className="text-xs text-gray-500">Confidence: {(scan.confidence*100).toFixed(1)}% | {scan.model_type==="hybrid"?"Quantum":"Classical"}</div></div></div>
        <div className="text-xs text-gray-400">{scan.inference_time}s</div>
      </div>))}</div>}</div>);

  const renderExplainability=()=>(<div className="p-5 space-y-4"><h2 className="text-lg font-semibold mb-2">How the AI explains its diagnosis</h2>
    <p className="text-sm text-gray-500 mb-4">QuantumRetina-X uses two methods so doctors can trust the AI:</p>
    <div className="grid grid-cols-2 gap-4">
      <Card title="Grad-CAM ? WHERE the AI looked" icon={<Eye size={14} className="text-red-400"/>}>
        <p className="text-xs text-gray-500 leading-relaxed mb-3">Creates a heatmap overlaid on the retina. Red = AI focused here. Blue = AI ignored this region. For diseased eyes, red should land on lesions.</p>
        <p className="text-xs text-purple-300">Upload an image to see live Grad-CAM heatmaps.</p>
      </Card>
      <Card title="TCAV ? WHY the AI chose that grade" icon={<Brain size={14} className="text-purple-400"/>}>
        <p className="text-xs text-gray-500 leading-relaxed mb-3">Measures which clinical concepts drove the diagnosis:</p>
        <div className="space-y-1 text-xs mb-3">
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-400"/>Hemorrhage ? blood leaking from vessels</div>
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-amber-400"/>Exudate ? fatty deposits</div>
          <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-400"/>Edema ? fluid swelling (also indicates DME)</div>
        </div>
        <div className="text-xs font-medium text-gray-700 mb-2">Scores increase with severity:</div>
        {Object.entries(TCAV_REF).map(([concept,scores])=>(<div key={concept} className="mb-3">
          <div className="text-xs text-gray-500 mb-1 capitalize">{concept}</div>
          <div className="flex gap-1">{Object.entries(scores).map(([g,s])=>(
            <div key={g} className="flex-1 text-center"><div className="h-12 rounded relative bg-slate-800 flex items-end"><div className="w-full rounded" style={{height:`${Number(s)*100}%`,background:Number(s)>0.5?"#8b5cf6":"#e2e8f0"}}/></div><span className="text-[10px] text-gray-400">G{g}</span></div>
          ))}</div></div>))}
      </Card>
    </div></div>);

  const renderInsights=()=>(<div className="p-5 space-y-4"><h2 className="text-lg font-semibold mb-2">Model performance</h2>
    <div className="grid grid-cols-4 gap-4">
      <StatBox label="Best accuracy" value="81.6%" sub="ViT+VQC on APTOS" color="#8b5cf6"/>
      <StatBox label="Best kappa" value="0.881" sub="ViT baseline" color="#3b82f6"/>
      <StatBox label="G1-sensitivity" value="78.6%" sub="Early detection" color="#f59e0b"/>
      <StatBox label="Datasets" value="4" sub="3 countries" color="#10b981"/>
    </div>
    <div className="grid grid-cols-2 gap-4">
      <Card title="Ablation study" icon={<Activity size={14} className="text-purple-400"/>}>
        {ABLATION.map(row=>(<div key={row.config} className="rounded-lg p-3 flex items-center justify-between mb-2" style={{background:"#1e3a8a"}}>
          <span className="text-xs font-medium" style={{color:row.color}}>{row.config}</span>
          <div className="flex gap-3 text-xs flex-wrap"><span className="text-gray-500">Acc: <span className="text-gray-800">{row.accuracy}%</span></span><span className="text-gray-500">F1: <span className="text-emerald-300">{row.f1}</span></span><span className="text-gray-500">AUC: <span className="text-sky-300">{row.auc}</span></span><span className="text-gray-500">Îº: <span className="text-gray-800">{row.kappa}</span></span><span className="text-gray-500">G1: <span className="text-gray-800">{row.g1}%</span></span></div>
        </div>))}</Card>
      <Card title="Cross-dataset generalisation" icon={<Shield size={14} className="text-purple-400"/>}>
        {CROSS_DATASET.map(row=>(<div key={row.dataset} className="rounded-lg p-3 mb-2" style={{background:"#1e3a8a"}}>
          <div className="flex items-center justify-between mb-1"><span className="text-xs font-medium text-gray-700">{row.dataset}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full ${row.role==="Trained"?"bg-purple-500/20 text-purple-300":"bg-slate-700 text-gray-500"}`}>{row.role}</span></div>
          <div className="flex gap-3 text-xs"><span className="text-blue-400">ViT:{row.vit_acc}% / AUC:{row.vit_auc}</span><span className="text-purple-400">VQC:{row.vqc_acc}% / AUC:{row.vqc_auc}</span><span className="text-amber-400">G1:{row.vqc_g1}%</span></div>
        </div>))}</Card>
    </div></div>);

  const renderSettings=()=>(<div className="p-5 space-y-4"><h2 className="text-lg font-semibold mb-4">Settings</h2>
    <Card title="Model configuration" icon={<Settings size={14} className="text-purple-400"/>}>
      <div className="space-y-4">
        <div><label className="text-xs text-gray-500 block mb-1">Default model</label>
          <select value={modelType} onChange={e=>setModelType(e.target.value as any)} className="bg-slate-800 text-gray-800 text-sm px-4 py-2 rounded-lg border border-slate-600 w-full">
            <option value="hybrid">ViT + VQC (Quantum)</option><option value="vit">ViT Only (Classical)</option></select></div>
        <div><label className="text-xs text-gray-500 block mb-1">Heatmap opacity</label>
          <input type="range" min={0} max={100} value={heatmapOpacity} onChange={e=>setHeatmapOpacity(Number(e.target.value))} className="w-full accent-blue-500"/><span className="text-xs text-gray-400">{heatmapOpacity}%</span></div>
      </div></Card>
    <Card title="System information" icon={<Info size={14} className="text-purple-400"/>}>
      <div className="space-y-2 text-xs">{[["Backend","FastAPI + PyTorch + CUDA"],["Backbone","ViT-Base/16"],["Quantum","4-qubit VQC (PennyLane)"],["XAI","Grad-CAM + TCAV"],["Detection","DR (5-grade) + DME"],["Training","APTOS 2019"],["Validation","Messidor, IDRiD, EyePACS"]].map(([k,v])=>(
        <div key={k} className="flex justify-between text-gray-500"><span>{k}</span><span className="text-gray-800">{v}</span></div>))}</div></Card></div>);

  const renderAbout=()=>(<div className="p-5 space-y-4"><h2 className="text-lg font-semibold mb-4">About QuantumRetina-X</h2>
    <Card title="Project overview" icon={<Zap size={14} className="text-purple-400"/>}>
      <p className="text-sm text-gray-700 leading-relaxed mb-3">QuantumRetina-X is an explainable hybrid quantum-classical Vision Transformer for <strong className="text-purple-300">early detection</strong> of Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME).</p>
      <p className="text-sm text-gray-500 leading-relaxed">It combines ViT with a 4-qubit VQC in a 16D Hilbert space, plus dual XAI (Grad-CAM + TCAV) for clinically interpretable diagnostics.</p></Card>
    <Card title="6 research contributions" icon={<CheckCircle size={14} className="text-green-400"/>}>
      <div className="space-y-2 text-xs text-gray-500">{["First ViT + VQC for DR and DME early detection","Quantum latent space for micro-lesion detection","Dual XAI (Grad-CAM + TCAV) in quantum medical model","Four-dataset cross-validation (3 countries)","Low-data quantum learning analysis","IoMT-ready lightweight architecture"].map((c,i)=>(
        <div key={i} className="flex items-start gap-2"><span className="text-purple-400 font-bold">{i+1}.</span><span>{c}</span></div>))}</div></Card>
    <div className="text-center text-xs text-gray-300 mt-6">QuantumRetina-X v1.0 ? 2026</div></div>);

  return (
    <div className="flex h-screen overflow-hidden">
      <div className="w-[200px] flex flex-col justify-between py-4 px-3" style={{background:"#1e40af",borderRight:"1px solid #e2e8f0"}}>
        <div>
          <div className="flex items-center gap-2 px-2 mb-8"><Zap size={22} className="text-purple-400"/><span className="text-lg font-semibold text-white">Quantum<span style={{color:\"#93c5fd\"}}>Retina-X</span></span></div>
          {SIDEBAR.map(item=>(<div key={item.label} onClick={()=>{setPage(item.page);setSelectedScan(null)}}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 cursor-pointer text-sm transition ${page===item.page?"bg-blue-600 text-white":"text-gray-500 hover:bg-blue-50"}`}>
            <item.icon size={18}/>{item.label}</div>))}
        </div>
        <div>
          <div className="flex items-center gap-3 px-3 py-2 mb-3 rounded-lg" style={{background:"#1e3a8a"}}>
            <div className="w-8 h-8 rounded-full bg-purple-600/30 flex items-center justify-center"><User size={16} className="text-purple-300"/></div>
            <div><div className="text-xs font-medium text-gray-800">Dr. Admin</div><div className="text-[10px] text-gray-400">Ophthalmologist</div></div>
            <ChevronDown size={14} className="text-gray-400 ml-auto"/></div>
          <div className="p-3 rounded-xl" style={{background:"linear-gradient(135deg,#1e3a8a,#1e40af)"}}>
            <div className="flex items-center gap-2 mb-1"><Zap size={14} className="text-purple-400"/><span className="text-sm font-medium">Quantum + AI</span></div>
            <p className="text-xs text-gray-500 leading-relaxed">Smarter. Earlier. Explainable.<br/>Better Vision for All.</p></div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="flex items-center justify-between px-6 py-3" style={{background:"#1e40af",borderBottom:"1px solid #e2e8f0"}}>
          <div className="flex items-center gap-3"><CheckCircle size={20} className="text-purple-400"/><div><div className="text-base font-semibold">Analysis Results</div><div className="text-xs text-gray-500">AI-powered retinal image analysis with explainability</div></div></div>
          <div className="flex items-center gap-6 text-xs text-gray-500">
            <div><div className="text-gray-400">Scan Date</div><div className="text-gray-800">{new Date().toLocaleDateString()}</div></div>
            <div><div className="text-gray-400">Model</div><div className="text-gray-800">QuantumRetina-X v1.0</div></div>
            <select value={modelType} onChange={e=>setModelType(e.target.value as any)} className="bg-slate-800 text-gray-800 text-xs px-3 py-1.5 rounded-lg border border-slate-600">
              <option value="hybrid">ViT + VQC (Quantum)</option><option value="vit">ViT Only</option></select></div>
        </div>
        {page==="dashboard"&&renderDashboard()}
        {page==="upload"&&<div className="p-5"><h2 className="text-lg font-semibold mb-4">Upload retinal image</h2>{renderUploadZone()}</div>}
        {page==="pipeline"&&renderPipeline()}
        {page==="reports"&&renderReports()}
        {page==="explainability"&&renderExplainability()}
        {page==="insights"&&renderInsights()}
        {page==="settings"&&renderSettings()}
        {page==="about"&&renderAbout()}
        {r&&page==="dashboard"&&(
          <div className="mx-5 mb-4 rounded-xl p-4 flex items-center justify-between" style={{background:"#ffffff",border:"1px solid #e2e8f0"}}>
            <div className="flex items-center gap-8">
              <div><div className="text-xs text-gray-400">Prediction</div><div className="text-sm font-semibold" style={{color:GRADE_COLORS[r.grade]}}>{r.grade_label}{r.dme_detected?" + DME":""}</div></div>
              <div><div className="text-xs text-gray-400">Confidence</div><div className="text-sm font-semibold text-white">{(r.confidence*100).toFixed(1)}%</div></div>
              <div><div className="text-xs text-gray-400">Status</div><div className="text-sm font-medium text-amber-400 flex items-center gap-1"><AlertTriangle size={12}/> Requires clinical review</div></div>
            </div>
            <div className="flex gap-2">
              <button onClick={()=>{setResult(null);setSelectedScan(null);setPreview(null)}} className="px-4 py-2 text-xs rounded-lg bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 transition">New scan</button>
              <button onClick={downloadHeatmap} className="px-4 py-2 text-xs rounded-lg bg-white hover:bg-gray-50 border border-gray-200 text-gray-700 transition flex items-center gap-1"><Download size={14}/> Heatmap</button>
              <button onClick={downloadReport} className="px-4 py-2 text-xs rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition flex items-center gap-1"><FileText size={14}/> Report</button>
            </div>
          </div>)}
        {error&&<div className="mx-5 mb-4 rounded-xl p-4 text-red-600 text-sm" style={{background:"#fef2f2",border:"1px solid #fca5a5"}}>Error: {error}</div>}
      </div>
    </div>
  );
}

