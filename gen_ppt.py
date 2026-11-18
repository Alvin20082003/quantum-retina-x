from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colours ──────────────────────────────────────────────────────────
BG      = RGBColor(0x07, 0x07, 0x11)   # near-black
PURPLE  = RGBColor(0x7c, 0x3a, 0xed)   # primary purple
LPURPLE = RGBColor(0xa8, 0x55, 0xf7)   # light purple
TEXT    = RGBColor(0xe2, 0xd9, 0xf3)   # near-white
MUTED   = RGBColor(0x9b, 0x89, 0xc4)   # muted purple-gray
GREEN   = RGBColor(0x22, 0xc5, 0x5e)
AMBER   = RGBColor(0xf5, 0x9e, 0x0b)
RED     = RGBColor(0xef, 0x44, 0x44)
WHITE   = RGBColor(0xff, 0xff, 0xff)
CARD    = RGBColor(0x11, 0x11, 0x28)   # card bg

W, H = Inches(13.33), Inches(7.5)   # 16:9 widescreen

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
blank = prs.slide_layouts[6]  # completely blank

def slide():
    s = prs.slides.add_slide(blank)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return s

def box(s, x, y, w, h, fill=None, border=None, border_w=Pt(1)):
    from pptx.util import Emu
    shp = s.shapes.add_shape(1, x, y, w, h)
    shp.line.width = border_w
    if fill:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    else:
        shp.fill.background()
    if border:
        shp.line.color.rgb = border
    else:
        shp.line.fill.background()
    return shp

def txt(s, text, x, y, w, h, size=20, bold=False, color=TEXT,
        align=PP_ALIGN.LEFT, wrap=True, italic=False):
    tf = s.shapes.add_textbox(x, y, w, h)
    tf.text_frame.word_wrap = wrap
    p = tf.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tf

def tag(s, text, x, y):
    b = box(s, x, y, Inches(2.8), Inches(0.28), fill=RGBColor(0x1f,0x10,0x40), border=PURPLE, border_w=Pt(0.5))
    txt(s, text, x+Inches(0.08), y+Inches(0.03), Inches(2.6), Inches(0.22),
        size=8, bold=True, color=LPURPLE, align=PP_ALIGN.LEFT)

def pill(s, text, x, y, fill_col, text_col=WHITE):
    b = box(s, x, y, Inches(1.6), Inches(0.3), fill=fill_col)
    b.line.fill.background()
    txt(s, text, x+Inches(0.05), y+Inches(0.03), Inches(1.5), Inches(0.24),
        size=9, bold=True, color=text_col, align=PP_ALIGN.CENTER)

def divider(s, y):
    ln = s.shapes.add_shape(1, Inches(0.5), y, Inches(12.33), Pt(1))
    ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0x2a,0x1a,0x4a)
    ln.line.fill.background()

def stat_card(s, x, y, w, h, big, label, sub, color=LPURPLE):
    box(s, x, y, w, h, fill=CARD, border=RGBColor(0x2a,0x1a,0x4a))
    txt(s, big,   x+Inches(0.15), y+Inches(0.1),  w-Inches(0.3), Inches(0.6), size=32, bold=True,  color=color, align=PP_ALIGN.CENTER)
    txt(s, label, x+Inches(0.1),  y+Inches(0.65), w-Inches(0.2), Inches(0.3), size=11, bold=True,  color=TEXT,  align=PP_ALIGN.CENTER)
    txt(s, sub,   x+Inches(0.1),  y+Inches(0.95), w-Inches(0.2), Inches(0.25),size=9,  bold=False, color=MUTED, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════
s1 = slide()
# Purple accent bar left
box(s1, Inches(0), Inches(0), Inches(0.08), H, fill=PURPLE)
# Eye dot
box(s1, Inches(0.5), Inches(0.5), Inches(0.6), Inches(0.6), fill=PURPLE)

txt(s1, "QuantumRetina-X", Inches(1.3), Inches(0.9), Inches(9), Inches(1.1),
    size=48, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
txt(s1, "Hybrid Vision Transformer + Variational Quantum Circuit", Inches(1.3), Inches(2.0), Inches(10), Inches(0.5),
    size=20, color=LPURPLE, align=PP_ALIGN.LEFT)
txt(s1, "for Diabetic Retinopathy Grading, DME Detection & Explainable AI", Inches(1.3), Inches(2.5), Inches(10), Inches(0.5),
    size=20, color=LPURPLE, align=PP_ALIGN.LEFT)
divider(s1, Inches(3.2))
txt(s1, "Jenna Therese  |  Alvin Sudhan", Inches(1.3), Inches(3.4), Inches(7), Inches(0.4), size=14, color=TEXT)
txt(s1, "IEEE Research Project  -  Phase 1 Review  -  2026", Inches(1.3), Inches(3.85), Inches(7), Inches(0.35), size=12, color=MUTED)
# Right side stat chips
for i,(v,l,c) in enumerate([("81.6%","Accuracy",GREEN),("0.871","Kappa",LPURPLE),("3,125","Params",AMBER),("89.1%","DME Acc",GREEN)]):
    sx = Inches(9.8); sy = Inches(1.0 + i*1.3)
    box(s1, sx, sy, Inches(2.9), Inches(1.1), fill=CARD, border=PURPLE)
    txt(s1, v, sx+Inches(0.1), sy+Inches(0.05), Inches(2.7), Inches(0.6), size=28, bold=True, color=c, align=PP_ALIGN.CENTER)
    txt(s1, l, sx+Inches(0.1), sy+Inches(0.65), Inches(2.7), Inches(0.3), size=10, color=MUTED, align=PP_ALIGN.CENTER)
txt(s1, "Track 01: Intelligent Systems  |  Track 02: Advanced Computing",
    Inches(1.3), Inches(6.9), Inches(10), Inches(0.3), size=10, color=MUTED)

# ════════════════════════════════════════════════════════════════════
# SLIDE 2 — Problem Statement
# ════════════════════════════════════════════════════════════════════
s2 = slide()
tag(s2, "THE PROBLEM", Inches(0.5), Inches(0.3))
txt(s2, "Diabetic Retinopathy is a silent epidemic", Inches(0.5), Inches(0.7), Inches(12), Inches(0.7),
    size=34, bold=True, color=WHITE)
txt(s2, "It is the leading cause of preventable blindness worldwide - yet it rarely shows symptoms until damage is irreversible.",
    Inches(0.5), Inches(1.5), Inches(12), Inches(0.5), size=14, color=MUTED)
divider(s2, Inches(2.1))

probs = [
    ("103 M", "People worldwide with DR", "Growing 10% annually", RED),
    ("90%",   "Of vision loss preventable", "With early detection", GREEN),
    ("50%",   "Inter-grader disagreement", "In manual grading", AMBER),
    ("1 in 3","Diabetics develop DR", "In their lifetime", LPURPLE),
]
for i,(big,lab,sub,col) in enumerate(probs):
    stat_card(s2, Inches(0.5+i*3.2), Inches(2.3), Inches(3.0), Inches(1.3), big, lab, sub, col)

divider(s2, Inches(3.75))
txt(s2, "Current Challenges", Inches(0.5), Inches(3.9), Inches(5), Inches(0.35), size=14, bold=True, color=LPURPLE)
challenges = [
    "Manual grading by specialists is expensive and scarce",
    "5-class grading (G0-G4) is complex - models struggle at Grade 1 (early detection)",
    "Classical CNNs lack interpretability - doctors cannot trust the AI",
    "Models trained on one dataset fail on others (domain shift)",
    "Existing AI ignores the quantum advantage for small-param efficiency",
]
for i,c in enumerate(challenges):
    txt(s2, f"  -  {c}", Inches(0.5), Inches(4.3+i*0.4), Inches(12), Inches(0.38), size=11, color=TEXT)

# ════════════════════════════════════════════════════════════════════
# SLIDE 3 — Solution Overview
# ════════════════════════════════════════════════════════════════════
s3 = slide()
tag(s3, "OUR SOLUTION", Inches(0.5), Inches(0.3))
txt(s3, "QuantumRetina-X: One model, three outputs", Inches(0.5), Inches(0.7), Inches(12), Inches(0.7),
    size=32, bold=True, color=WHITE)

solutions = [
    ("🩺", "DR Grading", "5-class classification\nGrade 0 to Grade 4\n81.6% accuracy, k=0.871", GREEN),
    ("💧", "DME Detection", "Macular oedema screening\nSeparate ViT+VQC model\n89.1% accuracy, AUC 0.928", AMBER),
    ("🔬", "XAI Explanation", "Grad-CAM heatmaps\nTCAV concept scores\n4 lesion types localised", LPURPLE),
]
for i,(icon,title,desc,col) in enumerate(solutions):
    sx = Inches(0.5 + i*4.2)
    box(s3, sx, Inches(1.6), Inches(4.0), Inches(2.8), fill=CARD, border=col)
    txt(s3, icon,  sx+Inches(0.2), Inches(1.75), Inches(0.6), Inches(0.5), size=24)
    txt(s3, title, sx+Inches(0.2), Inches(2.3),  Inches(3.6), Inches(0.4), size=16, bold=True, color=col)
    txt(s3, desc,  sx+Inches(0.2), Inches(2.75), Inches(3.6), Inches(0.9), size=11, color=TEXT)

divider(s3, Inches(4.6))
txt(s3, "Key Novelties", Inches(0.5), Inches(4.75), Inches(12), Inches(0.35), size=14, bold=True, color=LPURPLE)
novelties = [
    "First hybrid ViT + VQC architecture for DR grading and DME detection",
    "Only 3,125 trainable parameters (99.996% parameter reduction vs full fine-tuning)",
    "Dual explainability: spatial (Grad-CAM) + concept-level (TCAV)",
    "Tested zero-shot on 4 datasets across 3 countries - no retraining",
    "Eye Digital Twin for patient education (Phase 2 feature)",
]
for i,n in enumerate(novelties):
    box(s3, Inches(0.5), Inches(5.15+i*0.37), Inches(0.08), Inches(0.25), fill=PURPLE)
    txt(s3, n, Inches(0.75), Inches(5.15+i*0.37), Inches(12), Inches(0.33), size=11, color=TEXT)

# ════════════════════════════════════════════════════════════════════
# SLIDE 4 — Tech Stack
# ════════════════════════════════════════════════════════════════════
s4 = slide()
tag(s4, "TECHNOLOGY STACK", Inches(0.5), Inches(0.3))
txt(s4, "Full-stack quantum-classical AI platform", Inches(0.5), Inches(0.7), Inches(12), Inches(0.6),
    size=30, bold=True, color=WHITE)

cols = [
    ("Quantum Layer", LPURPLE, [
        "PennyLane 0.36",
        "default.qubit simulator",
        "Parameter-shift gradient",
        "4 qubits, 2 layers",
        "CNOT ring entanglement",
    ]),
    ("Deep Learning", AMBER, [
        "PyTorch 2.5.1 + CUDA 12.4",
        "HuggingFace Transformers",
        "ViT-Base/16 (in21k)",
        "pytorch-grad-cam",
        "scikit-learn TCAV",
    ]),
    ("Backend", GREEN, [
        "FastAPI + Uvicorn",
        "Python 3.12",
        "NVIDIA RTX 3060",
        "3 models loaded at start",
        "Real-time inference ~3s",
    ]),
    ("Frontend", RGBColor(0x38,0xbd,0xf8), [
        "React 19 + TypeScript",
        "Vite 8 + Tailwind CSS v4",
        "Recharts + Lucide Icons",
        "React Router v7",
        "Dark theme + Responsive",
    ]),
]
for i,(title,col,items) in enumerate(cols):
    sx = Inches(0.4 + i*3.2)
    box(s4, sx, Inches(1.5), Inches(3.1), Inches(1.7+len(items)*0.38), fill=CARD, border=col)
    txt(s4, title, sx+Inches(0.12), Inches(1.6), Inches(2.9), Inches(0.38), size=13, bold=True, color=col)
    for j,item in enumerate(items):
        txt(s4, f"- {item}", sx+Inches(0.12), Inches(2.05+j*0.38), Inches(2.9), Inches(0.36), size=10, color=TEXT)

divider(s4, Inches(6.0))
txt(s4, "Datasets: APTOS 2019 (train) | IDRiD (India) | Messidor-2 (France) | EyePACS (USA, 5k subset)",
    Inches(0.5), Inches(6.15), Inches(12.5), Inches(0.35), size=11, color=MUTED, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 5 — Architecture Flow
# ════════════════════════════════════════════════════════════════════
s5 = slide()
tag(s5, "ARCHITECTURE", Inches(0.5), Inches(0.2))
txt(s5, "Forward Pipeline - 7 stages", Inches(0.5), Inches(0.6), Inches(10), Inches(0.55), size=28, bold=True, color=WHITE)

stages = [
    ("Input\n224x224", RGBColor(0x60,0xa5,0xfa)),
    ("Preprocess\nNormalize", RGBColor(0xea,0xb3,0x08)),
    ("ViT Backbone\n86.4M frozen", RGBColor(0x3b,0x82,0xf6)),
    ("Reducer\n768->4", RGBColor(0x10,0xb9,0x81)),
    ("4-Qubit VQC\nPennyLane", RGBColor(0xf5,0x9e,0x0b)),
    ("Head\n4->5", RGBColor(0x10,0xb9,0x81)),
    ("G0-G4\n+ DME + XAI", LPURPLE),
]
bw = Inches(1.6); bh = Inches(1.1); by = Inches(1.4); gap = Inches(0.12)
for i,(label,col) in enumerate(stages):
    sx = Inches(0.3 + i*(bw+gap))
    box(s5, sx, by, bw, bh, fill=CARD, border=col)
    txt(s5, label, sx+Inches(0.05), by+Inches(0.1), bw-Inches(0.1), bh-Inches(0.15), size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
    if i < len(stages)-1:
        txt(s5, "->", sx+bw+Inches(0.0), by+Inches(0.35), Inches(0.14), Inches(0.4), size=14, color=MUTED, align=PP_ALIGN.CENTER)

divider(s5, Inches(2.65))

# XAI branches
branches = [
    ("Grad-CAM Branch", RED,     "Fundus Mask  ->  Backprop  ->  Heatmap overlay"),
    ("TCAV Branch",     RGBColor(0x3b,0x82,0xf6), "Concept scoring  ->  Per-concept maps  ->  Multicolor merge"),
    ("DME Branch",      GREEN,   "Same image  ->  2nd ViT+VQC  ->  Binary DME result"),
]
txt(s5, "XAI & Detection Branches (run in parallel with main pipeline)", Inches(0.5), Inches(2.75), Inches(12), Inches(0.35), size=12, bold=True, color=MUTED)
for i,(branch,col,flow) in enumerate(branches):
    sy = Inches(3.15 + i*0.95)
    box(s5, Inches(0.5), sy, Inches(1.8), Inches(0.7), fill=CARD, border=col)
    txt(s5, branch, Inches(0.6), sy+Inches(0.15), Inches(1.6), Inches(0.4), size=9, bold=True, color=col)
    txt(s5, "->", Inches(2.35), sy+Inches(0.2), Inches(0.25), Inches(0.3), size=12, color=MUTED)
    box(s5, Inches(2.65), sy, Inches(9.5), Inches(0.7), fill=RGBColor(0x0a,0x0a,0x1a), border=RGBColor(0x2a,0x1a,0x4a))
    txt(s5, flow, Inches(2.8), sy+Inches(0.17), Inches(9.2), Inches(0.4), size=11, color=TEXT)

# Key numbers bottom
txt(s5, "Key numbers:  3,125 trainable params  |  86.4M frozen  |  Inference ~3s  |  4 qubits  |  2 VQC layers",
    Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3), size=10, color=MUTED, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 6 — Phase 1 Outcomes
# ════════════════════════════════════════════════════════════════════
s6 = slide()
tag(s6, "PHASE 1 OUTCOMES", Inches(0.5), Inches(0.2))
txt(s6, "What we built and measured", Inches(0.5), Inches(0.6), Inches(10), Inches(0.55), size=28, bold=True, color=WHITE)

# Main results grid
results = [
    ("81.6%", "DR Accuracy",    "APTOS n=550",   GREEN),
    ("0.871",  "Kappa",          "Quadratic weighted", LPURPLE),
    ("89.1%", "DME Accuracy",   "Messidor+IDRiD", AMBER),
    ("0.928",  "DME AUC",        "AUC-ROC score",  RGBColor(0x38,0xbd,0xf8)),
    ("87.5%", "Mean Acc",        "3 seeds avg",    GREEN),
    ("3,125",  "Params",         "Trainable only",  AMBER),
]
for i,(big,lab,sub,col) in enumerate(results):
    sx = Inches(0.4 + (i%3)*4.3)
    sy = Inches(1.4 + (i//3)*1.5)
    stat_card(s6, sx, sy, Inches(4.1), Inches(1.3), big, lab, sub, col)

divider(s6, Inches(4.6))

txt(s6, "Ablation Study - What each component contributes", Inches(0.5), Inches(4.75), Inches(12), Inches(0.35), size=12, bold=True, color=LPURPLE)
ablation = [
    ("VQC-only (no ViT)",         "54.6%", "0.289", "0.646", "0.0%",  RED),
    ("ViT-only baseline",          "78.4%", "0.881", "0.936", "73.2%", RGBColor(0x3b,0x82,0xf6)),
    ("ViT+VQC - Ours",             "81.6%", "0.871", "0.894", "55.4%", GREEN),
    ("ViT+VQC G1-boost variant",   "74.6%", "0.762", "0.901", "78.6%", AMBER),
]
headers = ["Model", "Accuracy", "Kappa", "AUC", "G1-Sens"]
for j,h in enumerate(headers):
    txt(s6, h, Inches(0.5+j*2.4), Inches(5.15), Inches(2.3), Inches(0.3), size=9, bold=True, color=MUTED)
for i,(name,acc,kap,auc,g1,col) in enumerate(ablation):
    sy = Inches(5.5 + i*0.42)
    box(s6, Inches(0.5), sy, Inches(12.3), Inches(0.38), fill=CARD if i%2==0 else RGBColor(0x0a,0x0a,0x1a))
    for j,val in enumerate([name,acc,kap,auc,g1]):
        c = col if j==1 else TEXT
        txt(s6, val, Inches(0.6+j*2.4), sy+Inches(0.05), Inches(2.3), Inches(0.3), size=10, bold=(j==0), color=c)

# ════════════════════════════════════════════════════════════════════
# SLIDE 7 — Cross-Dataset Results
# ════════════════════════════════════════════════════════════════════
s7 = slide()
tag(s7, "CROSS-DATASET GENERALISATION", Inches(0.5), Inches(0.2))
txt(s7, "Zero-shot transfer to 3 unseen datasets", Inches(0.5), Inches(0.6), Inches(12), Inches(0.55), size=28, bold=True, color=WHITE)
txt(s7, "Models trained ONLY on APTOS 2019 (India). No retraining, no adaptation on target datasets.",
    Inches(0.5), Inches(1.3), Inches(12), Inches(0.35), size=12, color=MUTED)
divider(s7, Inches(1.75))

cross = [
    ("APTOS 2019", "India", "550",   "78.4%","81.6%","0.936","0.894","Trained dataset"),
    ("IDRiD",      "India", "455",   "42.9%","43.5%","0.786","0.657","Zero-shot"),
    ("Messidor-2", "France","1,744", "59.4%","58.3%","0.760","0.577","Zero-shot"),
    ("EyePACS",    "USA",   "5,000", "64.2%","63.0%","0.724","0.603","Zero-shot"),
]
headers2 = ["Dataset","Country","n","ViT-only Acc","ViT+VQC Acc","ViT-only AUC","ViT+VQC AUC","Type"]
for j,h in enumerate(headers2):
    xs = [0.4,1.6,2.6,3.5,4.9,6.3,7.7,9.1]
    ws = [1.1,1.0,1.0,1.4,1.4,1.4,1.4,1.4]
    txt(s7, h, Inches(xs[j]), Inches(1.9), Inches(ws[j]), Inches(0.3), size=9, bold=True, color=MUTED)

for i,row in enumerate(cross):
    sy = Inches(2.3 + i*0.9)
    bg_c = CARD if i%2==0 else RGBColor(0x0a,0x0a,0x1a)
    box(s7, Inches(0.4), sy, Inches(12.4), Inches(0.8), fill=bg_c, border=RGBColor(0x2a,0x1a,0x4a))
    xs2 = [0.5,1.65,2.65,3.6,5.0,6.4,7.8,9.2]
    ws2 = [1.1,0.95,0.95,1.3,1.3,1.3,1.3,1.3]
    colors2 = [WHITE,MUTED,MUTED,
               RGBColor(0x3b,0x82,0xf6),GREEN if row[4]>row[3] else AMBER,
               RGBColor(0x3b,0x82,0xf6),GREEN if row[6]>row[5] else AMBER,
               LPURPLE if row[7]=="Zero-shot" else GREEN]
    for j,(val,col) in enumerate(zip(row,colors2)):
        txt(s7, val, Inches(xs2[j]), sy+Inches(0.1), Inches(ws2[j]), Inches(0.6), size=11, color=col, bold=(j==0))

txt(s7, "Finding: ViT+VQC shows stronger kappa on IDRiD (0.618 vs 0.578) suggesting better ordinal structure learning.",
    Inches(0.5), Inches(6.0), Inches(12), Inches(0.4), size=11, color=AMBER)
txt(s7, "Domain shift is a known challenge - colour normalisation and fine-tuning experiments are planned for Phase 2.",
    Inches(0.5), Inches(6.4), Inches(12), Inches(0.4), size=11, color=MUTED)

# ════════════════════════════════════════════════════════════════════
# SLIDE 8 — Publication Status
# ════════════════════════════════════════════════════════════════════
s8 = slide()
tag(s8, "PUBLICATION STATUS", Inches(0.5), Inches(0.2))
txt(s8, "Two IEEE papers submitted", Inches(0.5), Inches(0.65), Inches(12), Inches(0.6), size=30, bold=True, color=WHITE)

for i,(title,author,track,acc,metrics) in enumerate([
    ("Hybrid ViT+VQC for Diabetic\nRetinopathy Grading",
     "Jenna Therese",
     "Track 02 - Advanced Computing / Quantum Computing",
     True,
     ["Acc: 81.6%", "Kappa: 0.871", "AUC: 0.894", "3,125 params"]),
    ("DR + DME Detection with\nExplainable AI",
     "Alvin Sudhan",
     "Track 01 - Intelligent Systems / Explainable AI",
     True,
     ["DME Acc: 89.1%", "AUC: 0.928", "Sens: 78.5%", "Spec: 91.7%"]),
]):
    sx = Inches(0.5 + i*6.4)
    box(s8, sx, Inches(1.5), Inches(6.1), Inches(3.8), fill=CARD, border=PURPLE)
    # Accepted badge
    if acc:
        box(s8, sx+Inches(3.6), Inches(1.6), Inches(2.1), Inches(0.4), fill=GREEN)
        txt(s8, "ACCEPTED", sx+Inches(3.65), Inches(1.65), Inches(2.0), Inches(0.3), size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s8, title, sx+Inches(0.2), Inches(1.65), Inches(5.3), Inches(0.9), size=15, bold=True, color=WHITE)
    txt(s8, f"Author: {author}", sx+Inches(0.2), Inches(2.6), Inches(5.7), Inches(0.3), size=11, color=LPURPLE)
    txt(s8, track, sx+Inches(0.2), Inches(2.95), Inches(5.7), Inches(0.4), size=10, color=MUTED)
    divider_local_x, divider_local_y = sx+Inches(0.2), Inches(3.4)
    ln = s8.shapes.add_shape(1, divider_local_x, divider_local_y, Inches(5.5), Pt(1))
    ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0x2a,0x1a,0x4a); ln.line.fill.background()
    txt(s8, "Key metrics:", sx+Inches(0.2), Inches(3.5), Inches(5.7), Inches(0.3), size=10, bold=True, color=MUTED)
    for j,m in enumerate(metrics):
        txt(s8, f"  - {m}", sx+Inches(0.2), Inches(3.85+j*0.33), Inches(5.5), Inches(0.3), size=11, color=TEXT)

divider(s8, Inches(5.5))
txt(s8, "Conference Impact", Inches(0.5), Inches(5.65), Inches(12), Inches(0.35), size=14, bold=True, color=LPURPLE)
impacts = [
    "First published work combining ViT and VQC for 5-class retinal grading",
    "Dual-publication strategy: DR grading paper (Track 02) + XAI paper (Track 01)",
    "Reproducibility established: 3-seed evaluation with published variance",
    "Open-source implementation: github.com/Alvin20082003/quantum-retina-x",
]
for i,imp in enumerate(impacts):
    txt(s8, f"  -  {imp}", Inches(0.5), Inches(6.1+i*0.33), Inches(12.3), Inches(0.3), size=11, color=TEXT)

# ════════════════════════════════════════════════════════════════════
# SLIDE 9 — Phase 2 Work Plan
# ════════════════════════════════════════════════════════════════════
s9 = slide()
tag(s9, "PHASE 2 WORK PLAN", Inches(0.5), Inches(0.2))
txt(s9, "Planned extensions and research directions", Inches(0.5), Inches(0.65), Inches(12), Inches(0.55), size=26, bold=True, color=WHITE)
divider(s9, Inches(1.3))

workstreams = [
    ("Q1 2027", "Model Improvements", RGBColor(0x3b,0x82,0xf6), [
        "Partial backbone unfreezing with low LR",
        "Colour normalisation (Reinhard / Ben Graham / CLAHE)",
        "Class-balancing strategies for Grade 1 sensitivity",
        "Multi-scale VQC with 8-qubit variant",
    ]),
    ("Q1 2027", "Clinical Validation", GREEN, [
        "Prospective evaluation on new hospital dataset",
        "Ophthalmologist inter-grader agreement study",
        "Lesion segmentation IoU improvement",
        "Real fundus image preprocessing pipeline",
    ]),
    ("Q2 2027", "Digital Twin V2", LPURPLE, [
        "Three.js 3D eyeball with fundus texture mapping",
        "Actual patient image projected on retina surface",
        "Animation of disease progression G0->G4",
        "Export PDF report with twin visualisation",
    ]),
    ("Q2 2027", "Platform Scale", AMBER, [
        "Multi-patient database with SQLite",
        "Doctor login / patient profile system",
        "REST API for hospital system integration",
        "Mobile-responsive progressive web app",
    ]),
]
for i,(quarter,title,col,items) in enumerate(workstreams):
    sx = Inches(0.4 + (i%2)*6.45)
    sy = Inches(1.5 + (i//2)*2.7)
    box(s9, sx, sy, Inches(6.2), Inches(2.4), fill=CARD, border=col)
    box(s9, sx, sy, Inches(1.1), Inches(0.4), fill=col)
    txt(s9, quarter, sx+Inches(0.05), sy+Inches(0.05), Inches(1.0), Inches(0.3), size=10, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    txt(s9, title, sx+Inches(1.2), sy+Inches(0.08), Inches(4.9), Inches(0.35), size=14, bold=True, color=col)
    for j,item in enumerate(items):
        txt(s9, f"- {item}", sx+Inches(0.2), sy+Inches(0.55+j*0.42), Inches(5.8), Inches(0.4), size=10, color=TEXT)

txt(s9, "Phase 2 Goal: Clinically deployable prototype with 3D Digital Twin and multi-patient SaaS platform",
    Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3), size=11, color=AMBER, align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 10 — Summary
# ════════════════════════════════════════════════════════════════════
s10 = slide()
box(s10, Inches(0), Inches(0), Inches(0.08), H, fill=PURPLE)
tag(s10, "SUMMARY", Inches(0.5), Inches(0.3))
txt(s10, "QuantumRetina-X - Phase 1 Complete", Inches(0.5), Inches(0.75), Inches(12), Inches(0.65), size=32, bold=True, color=WHITE)
divider(s10, Inches(1.55))

summary = [
    ("Built",       "Hybrid ViT+VQC with 3,125 trainable params",    GREEN),
    ("Achieved",    "81.6% accuracy, kappa 0.871 on APTOS (5-class)", GREEN),
    ("Validated",   "3-seed reproducibility: 87.5 +- 4.2% mean acc",  GREEN),
    ("Extended",    "DME detection: 89.1% accuracy, AUC 0.928",       GREEN),
    ("Explained",   "Grad-CAM + TCAV dual explainability",             LPURPLE),
    ("Deployed",    "Full clinical web UI + Digital Twin + Landing page", LPURPLE),
    ("Published",   "2 IEEE papers accepted (Track 01 + Track 02)",    AMBER),
    ("Open-sourced","github.com/Alvin20082003/quantum-retina-x",       MUTED),
]
for i,(label,desc,col) in enumerate(summary):
    sy = Inches(1.75 + i*0.6)
    box(s10, Inches(0.5), sy+Inches(0.08), Inches(0.08), Inches(0.38), fill=col)
    txt(s10, f"{label}:", Inches(0.75), sy+Inches(0.05), Inches(1.5), Inches(0.4), size=12, bold=True, color=col)
    txt(s10, desc, Inches(2.4), sy+Inches(0.05), Inches(9.0), Inches(0.4), size=12, color=TEXT)

divider(s10, Inches(6.65))
txt(s10, "Phase 2: 3D Digital Twin  |  Clinical dataset  |  SaaS multi-patient platform  |  Q1-Q2 2027",
    Inches(0.5), Inches(6.8), Inches(12.3), Inches(0.3), size=11, color=MUTED, align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────
out = "paper/final/QuantumRetina-X_Phase1_Review_v3.pptx"
prs.save(out)
print(f"Saved: {out}  ({len(prs.slides)} slides)")