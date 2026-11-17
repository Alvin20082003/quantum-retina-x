// Landing.tsx - QuantumRetina-X product landing page
import { useNavigate } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties, MouseEvent, ReactNode } from "react";

/* ------------------------------------------------------------------ */
/*  Inline CSS (dark theme)                                            */
/* ------------------------------------------------------------------ */
const CSS = `
  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap");
  @keyframes spin  { from { transform: rotate(0deg)   } to { transform: rotate(360deg) } }
  @keyframes spinR { from { transform: rotate(360deg) } to { transform: rotate(0deg)   } }
  @keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-16px) } }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(20px) } to { opacity: 1; transform: none } }
  @keyframes pulse  { 0%,100% { transform: scale(1) } 50% { transform: scale(1.03) } }
  @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
  @keyframes particleDrift {
    0%   { transform: translate(0,0); opacity: 0.3 }
    50%  { transform: translate(15px,-25px); opacity: 0.8 }
    100% { transform: translate(0,0); opacity: 0.3 }
  }
  *::-webkit-scrollbar { width: 8px; }
  *::-webkit-scrollbar-track { background: #0a0a1a; }
  *::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 4px; }
  *::-webkit-scrollbar-thumb:hover { background: #a855f7; }
  .ql-nav-link { font-size: 14px; font-weight: 500; color: #c4b5fd; text-decoration: none; transition: color 0.2s; }
  .ql-nav-link:hover { color: #a855f7; }
  .ql-btn-primary { background: #7c3aed; color: #fff; border: none; padding: 14px 30px; border-radius: 999px; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  .ql-btn-primary:hover { background: #6d28d9; transform: translateY(-1px); box-shadow: 0 8px 24px #7c3aed40; }
  .ql-btn-ghost { background: transparent; color: #c4b5fd; border: 1.5px solid #ffffff20; padding: 13px 26px; border-radius: 999px; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.2s; }
  .ql-btn-ghost:hover { border-color: #a855f7; color: #a855f7; }
  .ql-card { background: #111128; border: 1px solid #ffffff10; border-radius: 20px; padding: 32px; transition: all 0.3s; }
  .ql-card:hover { border-color: #a855f760; box-shadow: 0 20px 60px #7c3aed20; transform: translateY(-4px); }
  .ql-tag { display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; }
  .ql-faq-item { border: 1px solid #ffffff10; border-radius: 16px; overflow: hidden; background: #0f0f1f; transition: border-color 0.2s; }
  .ql-faq-item:hover { border-color: #7c3aed40; }
`;

/* ------------------------------------------------------------------ */
/*  Scroll-reveal hook                                                 */
/* ------------------------------------------------------------------ */
function useReveal<T extends HTMLElement = HTMLDivElement>(threshold = 0.15) {
  const ref = useRef<T | null>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver((entries: IntersectionObserverEntry[]) => {
      if (entries[0]?.isIntersecting) {
        setVisible(true);
        obs.disconnect();
      }
    }, { threshold });
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, visible };
}

const revealStyle = (visible: boolean): CSSProperties => ({
  opacity: visible ? 1 : 0,
  transform: visible ? "none" : "translateY(30px)",
  transition: "opacity 0.6s ease, transform 0.6s ease",
});

/* ------------------------------------------------------------------ */
/*  Animated counter                                                    */
/* ------------------------------------------------------------------ */
function Counter({ target, suffix = "", decimals = 0 }: {
  target: number; suffix?: string; decimals?: number;
}) {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const obs = new IntersectionObserver((entries: IntersectionObserverEntry[]) => {
      if (!entries[0]?.isIntersecting) return;
      obs.disconnect();
      let start = 0; const dur = 1800;
      const step = (ts: number) => {
        if (!start) start = ts;
        const p = Math.min((ts - start) / dur, 1);
        setVal(parseFloat((p * target).toFixed(decimals)));
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }, { threshold: 0.3 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [target, decimals]);
  return <span ref={ref}>{val.toFixed(decimals)}{suffix}</span>;
}

/* ------------------------------------------------------------------ */
/*  3D Eye component (5 orbit rings + iris + pupil + data labels)     */
/* ------------------------------------------------------------------ */
type DotSide = "top" | "bottom" | "left" | "right";
interface OrbitDot { side: DotSide; size: number; color: string; glow: string; }
interface Orbit { inset: number; anim: string; dots: OrbitDot[]; }

const ORBITS: Orbit[] = [
  { inset: 0,   anim: "spin 20s linear infinite",  dots: [{ side: "top", size: 12, color: "#a855f7", glow: "#a855f7" }] },
  { inset: 15,  anim: "spinR 14s linear infinite", dots: [{ side: "bottom", size: 8, color: "#7c3aed", glow: "#7c3aed" }] },
  { inset: 40,  anim: "spin 10s linear infinite",  dots: [{ side: "top", size: 8, color: "#c4b5fd", glow: "#c4b5fd" }, { side: "bottom", size: 6, color: "#a855f7", glow: "#a855f7" }] },
  { inset: 75,  anim: "spinR 7s linear infinite",  dots: [{ side: "right", size: 8, color: "#c4b5fd", glow: "#c4b5fd" }] },
  { inset: 115, anim: "spin 5s linear infinite",   dots: [{ side: "top", size: 6, color: "#a855f7", glow: "#a855f7" }] },
];

function dotStyle(dot: OrbitDot): CSSProperties {
  const s: CSSProperties = {
    position: "absolute", width: dot.size, height: dot.size, borderRadius: "50%",
    background: dot.color, boxShadow: `0 0 16px ${dot.glow}`,
  };
  const half = -dot.size / 2;
  if (dot.side === "top") { s.top = half; s.left = "50%"; s.transform = "translateX(-50%)"; }
  else if (dot.side === "bottom") { s.bottom = half; s.left = "50%"; s.transform = "translateX(-50%)"; }
  else if (dot.side === "right") { s.right = half; s.top = "50%"; s.transform = "translateY(-50%)"; }
  else { s.left = half; s.top = "50%"; s.transform = "translateY(-50%)"; }
  return s;
}

const EYE_LABELS: { text: string; top?: number | string; bottom?: number | string; left?: number | string; right?: number | string; color: string }[] = [
  { text: "81.6% ACC",  top: 8,  right: 28, color: "#a855f7" },
  { text: "κ=0.871",    bottom: 28, left: 12, color: "#7c3aed" },
  { text: "4-Qubit VQC", top: "45%", right: 0, color: "#c4b5fd" },
  { text: "86.4M params", top: 6, left: 16, color: "#c4b5fd" },
  { text: "APTOS 2019", bottom: 10, right: 18, color: "#a855f7" },
];

function Eye3D() {
  return (
    <div style={{ position: "relative", width: 400, height: 400, margin: "0 auto" }}>
      {/* floating data labels - fixed, not affected by pulse */}
      {EYE_LABELS.map((l) => (
        <div key={l.text} style={{ position: "absolute", top: l.top, bottom: l.bottom, left: l.left, right: l.right, fontSize: 10, color: l.color, fontFamily: "monospace", whiteSpace: "nowrap" }}>{l.text}</div>
      ))}

      {/* pulsing wrapper holds the orbits + eyeball */}
      <div style={{ position: "absolute", inset: 0, animation: "pulse 4s ease-in-out infinite" }}>
        {/* ambient glow */}
        <div style={{ position: "absolute", inset: 0, borderRadius: "50%",
          background: "radial-gradient(circle at 40% 35%, #a855f722, #7c3aed11 50%, transparent 70%)" }} />

        {ORBITS.map((o, i) => (
          <div key={i} style={{ position: "absolute", inset: o.inset, borderRadius: "50%", border: "1px solid #a855f720", animation: o.anim }}>
            {o.dots.map((d, j) => <div key={j} style={dotStyle(d)} />)}
          </div>
        ))}

        {/* eyeball */}
        <div style={{ position: "absolute", inset: 130, borderRadius: "50%",
          background: "radial-gradient(circle at 38% 32%, #ffffff 0%, #ede9fe 20%, #c4b5fd 45%, #7c3aed 72%, #3b0764 100%)",
          boxShadow: "0 0 70px #a855f750, 0 0 140px #7c3aed25, inset 0 0 50px #00000045",
          display: "flex", alignItems: "center", justifyContent: "center" }}>
          {/* iris radiating lines */}
          {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => (
            <div key={deg} style={{ position: "absolute", top: "50%", left: "50%", width: 1, height: 70,
              background: "linear-gradient(to bottom, transparent, #a855f730, transparent)",
              transform: `rotate(${deg}deg) translateX(-50%)`, transformOrigin: "top center" }} />
          ))}
          {/* pupil */}
          <div style={{ width: 68, height: 68, borderRadius: "50%",
            background: "radial-gradient(circle at 33% 28%, #1a1a2e 0%, #000 65%)",
            boxShadow: "0 0 24px #00000090, inset 0 0 18px #a855f720", position: "relative" }}>
            <div style={{ position: "absolute", top: 12, left: 14, width: 18, height: 18, borderRadius: "50%",
              background: "radial-gradient(circle, #fff 0%, #ffffff80 60%, transparent 100%)" }} />
            <div style={{ position: "absolute", top: 22, left: 30, width: 6, height: 6, borderRadius: "50%", background: "#ffffffcc" }} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Static content data                                                */
/* ------------------------------------------------------------------ */
const NAV_LINKS = [
  ["Solutions", "solutions"],
  ["Results", "results"],
  ["How it works", "how-it-works"],
  ["Research", "research"],
  ["FAQ", "faq"],
] as const;

const TRUST_LOGOS = ["APTOS 2019", "IEEE", "PennyLane", "HuggingFace", "PyTorch"] as const;

const MINI_STATS = [
  ["81.6%", "Accuracy"],
  ["0.871", "Kappa"],
  ["3,125", "Params"],
  ["4", "Datasets"],
] as const;

const PROBLEM_STATS = [
  { value: "103M", label: "People affected", desc: "Worldwide living with diabetic retinopathy today" },
  { value: "90%", label: "Preventable", desc: "Of vision loss is avoidable with early detection" },
  { value: "50%", label: "Misdiagnosed", desc: "Of manual grading suffers inter-grader variability" },
] as const;

const SOLUTION_STEPS = [
  { title: "Upload Image", detail: "Retinal fundus photograph, PNG or JPEG input" },
  { title: "ViT-Base/16 Backbone", detail: "86.4M params, frozen - extracts rich patch features" },
  { title: "Reducer Linear(768→4)", detail: "Compress the 768-dim CLS token to 4 quantum inputs" },
  { title: "4-Qubit VQC", detail: "PennyLane circuit, parameter-shift gradient training" },
  { title: "DR Grade + DME + XAI", detail: "5-class grade, DME screen, Grad-CAM + TCAV explainability" },
] as const;

const COUNTERS: { label: string; sub: string; color: string; n: number; suffix: string; decimals: number }[] = [
  { label: "Accuracy", sub: "on APTOS 2019 (n=550)", color: "#a855f7", n: 81.6, suffix: "%", decimals: 1 },
  { label: "Mean Accuracy", sub: "across seeds 42/123/456", color: "#c4b5fd", n: 87.5, suffix: "%", decimals: 1 },
  { label: "Trainable Params", sub: "vs 86.4M frozen backbone", color: "#7c3aed", n: 3125, suffix: "", decimals: 0 },
];

const CROSS_DATASETS = [
  { name: "IDRiD", flag: "🇮🇳", acc: "43.5%", n: "India · n=455" },
  { name: "Messidor", flag: "🇫🇷", acc: "59.4%", n: "France · n=1744" },
  { name: "EyePACS", flag: "🇺🇸", acc: "64.2%", n: "USA · n=5000" },
  { name: "APTOS", flag: "🇮🇳", acc: "81.6%", n: "India · trained+tested" },
] as const;

const FEATURES = [
  { icon: "👁️", title: "5-Class DR Grading", desc: "Grade 0 to Grade 4. 81.6% accuracy, 0.871 quadratic kappa on the APTOS test set.", accent: "#a855f7" },
  { icon: "💧", title: "DME Detection", desc: "Separate ViT+VQC model. 89.1% accuracy, AUC 0.928, sensitivity 78.5%.", accent: "#7c3aed" },
  { icon: "🔬", title: "Grad-CAM Heatmaps", desc: "Percentile-stretched, fundus-masked overlays showing exactly where the model looked.", accent: "#c4b5fd" },
  { icon: "🧪", title: "TCAV Lesion Analysis", desc: "4 probes: microaneurysms, hemorrhages, exudates, macular swelling - spatially localised.", accent: "#a855f7" },
  { icon: "📊", title: "Statistical Reproducibility", desc: "3 seeds tested. ViT+VQC: 87.5±4.2% accuracy, kappa=0.916±0.032.", accent: "#7c3aed" },
  { icon: "🌍", title: "Cross-Dataset Transfer", desc: "Zero-shot: IDRiD India, Messidor France, EyePACS USA. No retraining needed.", accent: "#c4b5fd" },
] as const;

const PIPELINE_STEPS = [
  { n: "01", title: "Input Image", detail: "Fundus photograph, PNG or JPEG" },
  { n: "02", title: "Preprocessing", detail: "Resize 224×224, ImageNet normalize (mean/std)" },
  { n: "03", title: "ViT Backbone", detail: "196 patches, 12 layers, 768-dim CLS token" },
  { n: "04", title: "Reducer", detail: "Linear 768→4 dimensionality reduction" },
  { n: "05", title: "VQC", detail: "4 qubits, 2 layers, RY+RX+RY+RZ+CNOT ring" },
  { n: "06", title: "Classification Head", detail: "Linear 4→5, 5-class DR grading output" },
  { n: "07", title: "XAI", detail: "Grad-CAM + TCAV + DME screen" },
] as const;

const PAPERS = [
  {
    title: "Hybrid ViT+VQC for Diabetic Retinopathy Grading",
    author: "Jenna Therese",
    track: "Track 02 · Advanced Computing",
    metrics: [["Accuracy", "81.6%"], ["Kappa", "0.871"], ["Params", "3,125"]] as const,
  },
  {
    title: "DR+DME Detection with Explainable AI",
    author: "Alvin Sudhan",
    track: "Track 01 · Intelligent Systems",
    metrics: [["DME Acc", "89.1%"], ["AUC", "0.928"], ["Sensitivity", "78.5%"]] as const,
  },
] as const;

const FAQS = [
  { q: "What is a Variational Quantum Circuit?", a: "A VQC is a parameterised quantum circuit whose gate angles are learned like neural-network weights. QuantumRetina-X encodes the 4 reduced ViT features into 4 qubits and reads out a 5-class DR grade." },
  { q: "How accurate is the model?", a: "81.6% accuracy with 0.871 quadratic kappa on the APTOS 2019 test set, and a mean of 87.5±4.2% across three random seeds - using just 3,125 trainable parameters." },
  { q: "What datasets were used?", a: "APTOS 2019 for training and primary testing. The model is then evaluated zero-shot on IDRiD (India), Messidor (France), and EyePACS (USA) to measure cross-dataset transfer." },
  { q: "Can it detect DME?", a: "Yes. A separate ViT+VQC model screens for Diabetic Macular Edema, reaching 89.1% accuracy, AUC 0.928, and 78.5% sensitivity." },
  { q: "Is this clinically validated?", a: "No. QuantumRetina-X is a research project and IEEE student paper. It is not a medical device and must not be used for clinical diagnosis." },
] as const;

const GRADES = [
  { g: "G0", label: "No DR", color: "#22c55e" },
  { g: "G1", label: "Mild", color: "#84cc16" },
  { g: "G2", label: "Moderate", color: "#f59e0b" },
  { g: "G3", label: "Severe", color: "#f97316" },
  { g: "G4", label: "Proliferative", color: "#ef4444" },
] as const;

const PARTICLES = [
  { left: "8%",  top: "18%", size: 4, delay: "0s",   dur: "8s"  },
  { left: "20%", top: "72%", size: 5, delay: "1.5s", dur: "10s" },
  { left: "35%", top: "30%", size: 3, delay: "0.8s", dur: "7s"  },
  { left: "48%", top: "82%", size: 6, delay: "2.2s", dur: "11s" },
  { left: "62%", top: "22%", size: 4, delay: "0.4s", dur: "9s"  },
  { left: "72%", top: "65%", size: 5, delay: "3s",   dur: "12s" },
  { left: "85%", top: "35%", size: 3, delay: "1.1s", dur: "8s"  },
  { left: "15%", top: "48%", size: 4, delay: "2.6s", dur: "10s" },
  { left: "55%", top: "55%", size: 5, delay: "0.6s", dur: "9s"  },
  { left: "90%", top: "75%", size: 4, delay: "1.8s", dur: "11s" },
] as const;

/* ------------------------------------------------------------------ */
/*  Landing page                                                       */
/* ------------------------------------------------------------------ */
export default function Landing() {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const [openFaq, setOpenFaq] = useState<number>(-1);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  return (
    <div style={{ fontFamily: "Inter, sans-serif", color: "#fff", background: "#070711", overflowX: "hidden" }}>
      <style>{CSS}</style>

      {/* A. NAVBAR */}
      <nav style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 200, padding: "0 48px",
        height: 64, display: "flex", alignItems: "center", justifyContent: "space-between",
        background: scrolled ? "rgba(10,10,26,0.92)" : "transparent",
        backdropFilter: scrolled ? "blur(16px)" : "none",
        borderBottom: scrolled ? "1px solid #ffffff10" : "none", transition: "all 0.3s" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9, cursor: "pointer" }} onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
          <div style={{ width: 30, height: 30, borderRadius: "50%",
            background: "radial-gradient(circle at 35% 30%, #c4b5fd, #7c3aed, #3b0764)",
            boxShadow: "0 0 12px #a855f760", animation: "pulse 4s ease-in-out infinite" }} />
          <span style={{ fontWeight: 800, fontSize: 17, letterSpacing: -0.5 }}>QuantumRetina<span style={{ color: "#a855f7" }}>-X</span></span>
        </div>
        <div style={{ display: "flex", gap: 36 }}>
          {NAV_LINKS.map(([label, href]) => (
            <a key={label} href={`#${href}`} className="ql-nav-link">{label}</a>
          ))}
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <button className="ql-btn-ghost" onClick={() => navigate("/app")}>Log In</button>
          <button className="ql-btn-primary" onClick={() => navigate("/app")}>Launch App →</button>
        </div>
      </nav>

      {/* B. HERO */}
      <section style={{ minHeight: "100vh", background: "#070711", display: "flex", alignItems: "center",
        paddingTop: 64, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: "5%", right: "2%", width: 700, height: 700, borderRadius: "50%",
          background: "radial-gradient(circle, #a855f712 0%, transparent 70%)", pointerEvents: "none" }} />
        {/* floating particles */}
        {PARTICLES.map((p, i) => (
          <div key={i} style={{ position: "absolute", left: p.left, top: p.top, width: p.size, height: p.size,
            borderRadius: "50%", background: "#a855f7", animation: `particleDrift ${p.dur} ease-in-out ${p.delay} infinite` }} />
        ))}
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "80px 48px",
          display: "grid", gridTemplateColumns: "1fr 1fr", gap: 60, alignItems: "center", width: "100%", className: "ql-hero-grid" }}>
          <div style={{ animation: "fadeUp 0.8s ease both" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>Quantum-Enhanced Retinal AI</div>
            <h1 style={{ fontSize: "clamp(40px,5.5vw,76px)", fontWeight: 900, lineHeight: 1.02, letterSpacing: -2, margin: 0 }}>
              Detect Diabetic Retinopathy Before Vision Loss
            </h1>
            <p style={{ fontSize: 18, color: "#bdb7d4", lineHeight: 1.7, margin: "28px 0 40px", maxWidth: 480 }}>
              A hybrid Vision Transformer + Variational Quantum Circuit system for DR grading - 81.6% accuracy with just 3,125 trainable parameters.
            </p>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <button className="ql-btn-primary" style={{ fontSize: 16, padding: "16px 36px" }} onClick={() => navigate("/app")}>Launch Diagnosis App →</button>
              <button className="ql-btn-ghost" style={{ fontSize: 16, padding: "16px 32px" }}
                onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}>
                How it works
              </button>
            </div>
            <div style={{ display: "flex", gap: 40, marginTop: 52, flexWrap: "wrap" }}>
              {MINI_STATS.map(([v, l]) => (
                <div key={l}>
                  <div style={{ fontSize: 24, fontWeight: 800, color: "#a855f7" }}>{v}</div>
                  <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{l}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ animation: "float 6s ease-in-out infinite" }}>
            <Eye3D />
          </div>
        </div>
      </section>

      {/* C. TRUST BAR */}
      <section style={{ background: "#0a0a1a", padding: "56px 48px", textAlign: "center", borderBottom: "1px solid #ffffff10" }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 2, color: "#666", marginBottom: 28 }}>BACKED BY RESEARCH FROM</div>
        <div className="ql-trust-row" style={{ display: "flex", justifyContent: "center", alignItems: "center", flexWrap: "wrap", gap: 56 }}>
          {TRUST_LOGOS.map((logo) => (
            <span key={logo} style={{ fontSize: 20, fontWeight: 700, color: "#ffffff66", letterSpacing: -0.5 }}>{logo}</span>
          ))}
        </div>
      </section>

      {/* D. PROBLEM */}
      <section style={{ background: "#0f0f1f", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1100, margin: "0 auto", textAlign: "center" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>The Problem</div>
            <h2 style={{ fontSize: "clamp(30px,4.5vw,56px)", fontWeight: 800, lineHeight: 1.1, letterSpacing: -1.5, marginBottom: 72, maxWidth: 820, margin: "0 auto 72px" }}>
              103 million people have diabetic retinopathy. Most don&rsquo;t know it.
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 24 }}>
              {PROBLEM_STATS.map((s) => (
                <div key={s.label} className="ql-card">
                  <div style={{ fontSize: "clamp(44px,6vw,72px)", fontWeight: 900, letterSpacing: -2, color: "#a855f7" }}>{s.value}</div>
                  <div style={{ fontSize: 16, fontWeight: 600, marginTop: 12 }}>{s.label}</div>
                  <div style={{ fontSize: 13, color: "#888", marginTop: 8, lineHeight: 1.6 }}>{s.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      {/* E. SOLUTION */}
      <section id="solutions-anchor" style={{ background: "#111128", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>Our Solution</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, maxWidth: 760, marginBottom: 64 }}>
              A quantum-classical AI pipeline that sees what humans miss.
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 56, alignItems: "center" }}>
              {/* vertical connected pipeline */}
              <div style={{ position: "relative" }}>
                {SOLUTION_STEPS.map((step, i) => (
                  <div key={step.title} style={{ display: "flex", gap: 18, position: "relative", paddingBottom: i < SOLUTION_STEPS.length - 1 ? 28 : 0 }}>
                    <div style={{ position: "relative", display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <div style={{ width: 14, height: 14, borderRadius: "50%", background: "#a855f7", boxShadow: "0 0 12px #a855f7", flexShrink: 0 }} />
                      {i < SOLUTION_STEPS.length - 1 && <div style={{ width: 2, flex: 1, minHeight: 28, background: "linear-gradient(to bottom, #a855f760, #7c3aed20)" }} />}
                    </div>
                    <div style={{ paddingBottom: 0 }}>
                      <div style={{ fontSize: 16, fontWeight: 700 }}>{step.title}</div>
                      <div style={{ fontSize: 13, color: "#999", marginTop: 4, lineHeight: 1.5 }}>{step.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
              {/* mock result card */}
              <div className="ql-card" style={{ maxWidth: 360, margin: "0 auto" }}>
                <div style={{ fontSize: 11, color: "#a855f7", fontWeight: 700, letterSpacing: 1, marginBottom: 10 }}>PREDICTION</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
                  <span style={{ fontSize: 40, fontWeight: 900, color: "#f59e0b" }}>G2</span>
                  <span style={{ fontSize: 18, fontWeight: 700 }}>Moderate</span>
                </div>
                <div style={{ fontSize: 13, color: "#888", marginBottom: 20 }}>Confidence: 81.6%</div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
                  {GRADES.map((gr) => (
                    <span key={gr.g} style={{ padding: "6px 12px", borderRadius: 999, fontSize: 12, fontWeight: 700,
                      border: `1.5px solid ${gr.g === "G2" ? gr.color : "#ffffff20"}`,
                      background: gr.g === "G2" ? `${gr.color}30` : "transparent",
                      color: gr.g === "G2" ? gr.color : "#666" }}>{gr.g} {gr.label}</span>
                  ))}
                </div>
                <div style={{ height: 6, borderRadius: 3, background: "#ffffff10", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: "81.6%", borderRadius: 3, background: "linear-gradient(90deg,#22c55e,#f59e0b,#ef4444)" }} />
                </div>
              </div>
            </div>
          </div>
        </Reveal>
      </section>

      {/* F. RESULTS */}
      <section id="results" style={{ background: "#0a0a1a", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>Real Numbers. Real Model.</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, marginBottom: 72, maxWidth: 600 }}>Research validates our quantum advantage</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 48 }}>
              {COUNTERS.map((c) => (
                <div key={c.label}>
                  <div style={{ fontSize: "clamp(44px,6vw,72px)", fontWeight: 900, letterSpacing: -2, color: c.color }}>
                    <Counter target={c.n} suffix={c.suffix} decimals={c.decimals} />
                  </div>
                  <div style={{ fontSize: 16, fontWeight: 600, marginTop: 12 }}>{c.label}</div>
                  <div style={{ fontSize: 13, color: "#888", marginTop: 6 }}>{c.sub}</div>
                </div>
              ))}
            </div>
            <div style={{ height: 1, background: "#ffffff10", margin: "80px 0" }} />
            <div style={{ fontSize: 13, color: "#666", marginBottom: 20 }}>Zero-shot cross-dataset transfer</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 24 }}>
              {CROSS_DATASETS.map((d) => (
                <div key={d.name} className="ql-card" style={{ padding: 28 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ fontSize: 22 }}>{d.flag}</span>
                    <span style={{ fontSize: 14, fontWeight: 700 }}>{d.name}</span>
                  </div>
                  <div style={{ fontSize: 40, fontWeight: 800, color: "#a855f7" }}>{d.acc}</div>
                  <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>{d.n}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      {/* G. FEATURES */}
      <section id="solutions" style={{ background: "#0f0f1f", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>What it does</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, marginBottom: 64 }}>Everything a clinician needs</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 24 }}>
              {FEATURES.map((f) => (
                <div key={f.title} className="ql-card" style={{ display: "flex", alignItems: "center", gap: 20, padding: "28px 32px" }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 32, marginBottom: 12 }}>{f.icon}</div>
                    <div style={{ fontSize: 17, fontWeight: 700, marginBottom: 8 }}>{f.title}</div>
                    <div style={{ fontSize: 14, color: "#999", lineHeight: 1.65 }}>{f.desc}</div>
                  </div>
                  <div style={{ width: 6, alignSelf: "stretch", borderRadius: 3, background: f.accent, boxShadow: `0 0 16px ${f.accent}80`, flexShrink: 0 }} />
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      {/* H. PIPELINE DETAIL */}
      <section id="how-it-works" style={{ background: "#111128", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>How it works</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, marginBottom: 16 }}>From fundus photo to diagnosis in seconds</h2>
            <p style={{ fontSize: 16, color: "#999", marginBottom: 64, maxWidth: 560 }}>7-stage pipeline: classical vision transformers + quantum circuits.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 24 }}>
              {PIPELINE_STEPS.map((step) => (
                <div key={step.n} className="ql-card">
                  <div style={{ fontSize: 40, fontWeight: 900, color: "#7c3aed", letterSpacing: -1, marginBottom: 12 }}>{step.n}</div>
                  <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 10 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: "#999", lineHeight: 1.6 }}>{step.detail}</div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      {/* I. RESEARCH PAPERS */}
      <section id="research" style={{ background: "#0a0a1a", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>Published Research</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, marginBottom: 56 }}>Two IEEE tracks, one quantum vision</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 24 }}>
              {PAPERS.map((p) => (
                <div key={p.title} className="ql-card" style={{ display: "flex", flexDirection: "column" }}>
                  <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", alignSelf: "flex-start", marginBottom: 20 }}>{p.track}</div>
                  <div style={{ fontSize: 20, fontWeight: 800, lineHeight: 1.3, marginBottom: 12 }}>{p.title}</div>
                  <div style={{ fontSize: 14, color: "#c4b5fd", marginBottom: 24 }}>By {p.author}</div>
                  <div style={{ display: "flex", gap: 24, marginTop: "auto", paddingTop: 20, borderTop: "1px solid #ffffff10" }}>
                    {p.metrics.map(([k, v]) => (
                      <div key={k}>
                        <div style={{ fontSize: 22, fontWeight: 800, color: "#a855f7" }}>{v}</div>
                        <div style={{ fontSize: 11, color: "#777", marginTop: 4 }}>{k}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </section>

      {/* J. FAQ */}
      <section id="faq" style={{ background: "#0f0f1f", padding: "120px 48px" }}>
        <Reveal>
          <div style={{ maxWidth: 820, margin: "0 auto" }}>
            <div className="ql-tag" style={{ background: "#7c3aed20", color: "#c4b5fd", marginBottom: 24 }}>FAQ</div>
            <h2 style={{ fontSize: "clamp(28px,4vw,52px)", fontWeight: 800, letterSpacing: -1, marginBottom: 48 }}>Questions, answered</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {FAQS.map((item, i) => {
                const open = openFaq === i;
                return (
                  <div key={i} className="ql-faq-item">
                    <button onClick={() => setOpenFaq(open ? -1 : i)} style={{
                      width: "100%", textAlign: "left", background: "transparent", border: "none", color: "#fff",
                      padding: "22px 24px", fontSize: 16, fontWeight: 600, cursor: "pointer",
                      display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16,
                    }}>
                      <span>{item.q}</span>
                      <span style={{ color: "#a855f7", fontSize: 22, lineHeight: 1, transition: "transform 0.2s", transform: open ? "rotate(45deg)" : "none" }}>+</span>
                    </button>
                    <div style={{ maxHeight: open ? 240 : 0, overflow: "hidden", transition: "max-height 0.3s ease" }}>
                      <div style={{ padding: "0 24px 22px", color: "#aaa", fontSize: 14, lineHeight: 1.7 }}>{item.a}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </Reveal>
      </section>

      {/* K. CTA */}
      <section className="ql-cta-section" style={{ background: "linear-gradient(135deg, #3b0764 0%, #7c3aed 50%, #a855f7 100%)", padding: "120px 48px", textAlign: "center" }}>
        <div style={{ maxWidth: 700, margin: "0 auto" }}>
          <div className="ql-tag" style={{ background: "#ffffff20", color: "#fff", marginBottom: 28 }}>Launch the app</div>
          <h2 style={{ fontSize: "clamp(36px,6vw,72px)", fontWeight: 900, letterSpacing: -2, lineHeight: 1, marginBottom: 24 }}>Ready to see clearly?</h2>
          <p style={{ fontSize: 18, color: "#e9d5ff", lineHeight: 1.7, marginBottom: 48, maxWidth: 560, margin: "0 auto 48px" }}>
            Upload a fundus image and get DR grade, DME screen, Grad-CAM heatmap, and TCAV lesion analysis - instantly.
          </p>
          <button style={{ background: "#fff", color: "#7c3aed", border: "none", padding: "18px 48px", borderRadius: 999,
            fontSize: 18, fontWeight: 700, cursor: "pointer", transition: "all 0.2s", boxShadow: "0 8px 32px #00000030" }}
            onClick={() => navigate("/app")}
            onMouseEnter={(e: MouseEvent<HTMLButtonElement>) => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 12px 40px #00000040"; }}
            onMouseLeave={(e: MouseEvent<HTMLButtonElement>) => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "0 8px 32px #00000030"; }}>
            Launch QuantumRetina-X →
          </button>
        </div>
      </section>

      {/* L. FOOTER */}
      <footer style={{ background: "#070711", padding: "64px 48px", color: "#888" }}>
        <div className="ql-footer-row" style={{ maxWidth: 1200, margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 24 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 14 }}>
              <div style={{ width: 26, height: 26, borderRadius: "50%", background: "radial-gradient(circle at 35% 30%, #c4b5fd, #7c3aed, #3b0764)" }} />
              <span style={{ color: "#fff", fontWeight: 700, fontSize: 16 }}>QuantumRetina-X</span>
            </div>
            <div style={{ fontSize: 12, lineHeight: 1.6 }}>Hybrid ViT + VQC for Diabetic Retinopathy Grading</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>IEEE Research Project · APTOS 2019 · PennyLane · PyTorch</div>
          </div>
          <div style={{ display: "flex", gap: 32, fontSize: 13 }}>
            {["GitHub", "Paper 1 (Track 02)", "Paper 2 (Track 01)", "Launch App"].map((l) => (
              <a key={l} href={l === "Launch App" ? "/app" : "#"} style={{ color: "#888", textDecoration: "none", transition: "color 0.2s" }}
                onMouseEnter={(e: MouseEvent<HTMLAnchorElement>) => { e.currentTarget.style.color = "#a855f7"; }}
                onMouseLeave={(e: MouseEvent<HTMLAnchorElement>) => { e.currentTarget.style.color = "#888"; }}>{l}</a>
            ))}
          </div>
          <div style={{ fontSize: 12 }}>2026 QuantumRetina-X. Research use only.</div>
        </div>
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Reveal wrapper - fades content up when it enters the viewport      */
/* ------------------------------------------------------------------ */
function Reveal({ children }: { children: ReactNode }) {
  const { ref, visible } = useReveal<HTMLDivElement>();
  return (
    <div ref={ref} style={revealStyle(visible)}>
      {children}
    </div>
  );
}
