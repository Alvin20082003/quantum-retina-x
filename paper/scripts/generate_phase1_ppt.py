"""Generates paper/QuantumRetina-X_Phase1_Review_v2.pptx - a Phase-I project
review deck matching the exact slide structure, section titles, and table
headers of the review template the user shared (title, agenda, abstract,
objectives, base paper details, literature survey table x5, inferences,
proposed system x4, references, thank you). Plain black-on-white styling
only - no added color scheme, matching the template's own look rather than
introducing a new design. Content is drawn from the real QuantumRetina-X
Paper 1 (architecture, presented as Phase 1) and Paper 2 (dual
explainability + DME, presented as the Phase 2 add-on). Literature-survey
entries are real, verified publications (checked via web search during
drafting) - author lists were cross-checked against search results but
should be re-verified against the original source before submission.
University/guide/student fields are left as placeholders since that
information isn't part of the project content.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BLACK = RGBColor(0x00, 0x00, 0x00)
DARK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0x99, 0x99, 0x99)
TABLE_HEAD = RGBColor(0xC6, 0xE0, 0xB4)
ROW_A = RGBColor(0xFF, 0xFF, 0xFF)
ROW_B = RGBColor(0xE2, 0xEF, 0xDA)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

PAGE = [0]

def new_slide():
    return prs.slides.add_slide(BLANK)

def frame(slide):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.15), Inches(0.15), Inches(13.03), Inches(7.2))
    rect.fill.background()
    rect.line.color.rgb = LINE
    rect.line.width = Pt(1)
    rect.shadow.inherit = False
    return rect

def add_footer(slide):
    PAGE[0] += 1
    box = slide.shapes.add_textbox(Inches(0.3), Inches(7.05), Inches(7), Inches(0.3))
    tf = box.text_frame
    tf.paragraphs[0].text = "23 July 2026    School of Computing - CSE"
    tf.paragraphs[0].font.size = Pt(10)
    tf.paragraphs[0].font.color.rgb = GREY
    box2 = slide.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.5), Inches(0.3))
    tf2 = box2.text_frame
    tf2.paragraphs[0].text = str(PAGE[0])
    tf2.paragraphs[0].font.size = Pt(10)
    tf2.paragraphs[0].font.color.rgb = GREY
    tf2.paragraphs[0].alignment = PP_ALIGN.RIGHT

def title(slide, text, size=32):
    frame(slide)
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.3), Inches(12.5), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = BLACK
    p.alignment = PP_ALIGN.CENTER
    ln = slide.shapes.add_connector(1, Inches(0.5), Inches(1.15), Inches(12.83), Inches(1.15))
    ln.line.color.rgb = LINE
    ln.line.width = Pt(1)
    return box

def body_box(slide, left=0.6, top=1.4, width=12.1, height=5.5):
    return slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))

def bullets(tf, items, size=18, first=True, space_after=10):
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if (first and i == 0) else tf.add_paragraph()
        p.text = "•  " + item
        p.font.size = Pt(size)
        p.font.color.rgb = DARK
        p.space_after = Pt(space_after)

def make_table(slide, rows, cols, left, top, width, height, col_widths=None):
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)
    return table

def style_header_row(table, headers):
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = TABLE_HEAD
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = BLACK
            p.alignment = PP_ALIGN.CENTER

def set_cell(table, r, c, text, size=10, align=PP_ALIGN.LEFT, bold=False, row_shade=None):
    cell = table.cell(r, c)
    cell.text = text
    cell.fill.solid(); cell.fill.fore_color.rgb = row_shade if row_shade else WHITE
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = DARK
        p.alignment = align
    cell.text_frame.word_wrap = True
    cell.margin_left = Inches(0.05); cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)

# =============================================================================
# 1. TITLE SLIDE
# =============================================================================
s = new_slide()
frame(s)
logo_ph = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.3), Inches(0.4), Inches(2.7), Inches(1.0))
logo_ph.fill.background()
logo_ph.line.color.rgb = LINE
tf = logo_ph.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]; p.text = "[Insert Institution Logo]"
p.font.size = Pt(11); p.font.color.rgb = GREY; p.alignment = PP_ALIGN.CENTER

hdr = s.shapes.add_textbox(Inches(0.5), Inches(1.55), Inches(12.3), Inches(0.9))
tf = hdr.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "DEPARTMENT OF [YOUR DEPARTMENT]"
p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = BLACK; p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph()
p2.text = "PROJECT PHASE-I"
p2.font.size = Pt(20); p2.font.bold = True; p2.font.color.rgb = BLACK; p2.alignment = PP_ALIGN.CENTER

title_box = s.shapes.add_textbox(Inches(0.8), Inches(2.9), Inches(11.7), Inches(1.6))
tf = title_box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "QuantumRetina-X: A Hybrid Quantum-Classical Vision Transformer for Diabetic Retinopathy Grading and Explainable Diabetic Macular Edema Diagnosis"
p.font.size = Pt(24); p.font.bold = False; p.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
p.alignment = PP_ALIGN.CENTER

stud_box = s.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(5.8), Inches(1.4))
tf = stud_box.text_frame
p = tf.paragraphs[0]; p.text = "PROJECT STUDENTS"; p.font.bold = True; p.font.size = Pt(16); p.font.color.rgb = BLACK
p2 = tf.add_paragraph(); p2.text = "[Your Name 1], [Reg. No.]"; p2.font.size = Pt(14); p2.font.color.rgb = DARK
p3 = tf.add_paragraph(); p3.text = "[Your Name 2], [Reg. No.]"; p3.font.size = Pt(14); p3.font.color.rgb = DARK

guide_box = s.shapes.add_textbox(Inches(6.9), Inches(5.3), Inches(5.6), Inches(1.4))
tf = guide_box.text_frame
p = tf.paragraphs[0]; p.text = "GUIDE"; p.font.bold = True; p.font.size = Pt(16); p.font.color.rgb = BLACK
p2 = tf.add_paragraph(); p2.text = "[Guide Name], [Designation]"; p2.font.size = Pt(14); p2.font.color.rgb = DARK
add_footer(s)

# =============================================================================
# 2. AGENDA
# =============================================================================
s = new_slide(); title(s, "AGENDA")
box = body_box(s, top=1.6)
tf = box.text_frame
bullets(tf, [
    "Abstract", "Objective(s)", "Literature Survey",
    "Inferences from Literature Survey", "Proposed System", "References",
], size=24, space_after=18)
add_footer(s)

# =============================================================================
# 3. ABSTRACT
# =============================================================================
s = new_slide(); title(s, "ABSTRACT")
box = body_box(s, top=1.5, height=5.4)
tf = box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = (
    "Diabetic retinopathy (DR) and diabetic macular edema (DME) are leading causes of "
    "preventable blindness, and automated screening matters where eye specialists are scarce. "
    "This project, QuantumRetina-X, presents a hybrid Vision Transformer and variational "
    "quantum circuit architecture for DR grading, extended with a dual-disease, "
    "dual-explainability layer."
)
p.font.size = Pt(16); p.alignment = PP_ALIGN.JUSTIFY; p.space_after = Pt(12)

p2 = tf.add_paragraph()
p2.text = (
    "Phase 1 (Paper 1) establishes the core architecture: a frozen ViT-Base/16 backbone feeds "
    "a linear reducer and a 4-qubit variational quantum circuit for 5-class DR grading, "
    "validated with a 3-way ablation study, a Grade-1-sensitivity training variant, and "
    "zero-shot testing on three datasets never seen in training."
)
p2.font.size = Pt(16); p2.alignment = PP_ALIGN.JUSTIFY; p2.space_after = Pt(12)

p3 = tf.add_paragraph()
p3.text = (
    "Phase 2 (Paper 2), an add-on to Phase 1, extends the same shared backbone with a "
    "diabetic macular edema head and two explanation methods validated quantitatively against "
    "real expert evidence: Gradient-weighted Class Activation Mapping (Grad-CAM) checked "
    "against IDRiD's pixel-level lesion masks, and Testing with Concept Activation Vectors "
    "(TCAV) using probes trained on real cropped lesion patches. Every result, including the "
    "honest accuracy drop under real dataset shift, is reported transparently."
)
p3.font.size = Pt(16); p3.alignment = PP_ALIGN.JUSTIFY
add_footer(s)

# =============================================================================
# 4. OBJECTIVE(S)
# =============================================================================
s = new_slide(); title(s, "OBJECTIVE(S)")
box = body_box(s, top=1.6)
tf = box.text_frame; tf.word_wrap = True
bullets(tf, [
    "Phase 1: Design and evaluate a hybrid ViT + 4-qubit VQC architecture for 5-class DR grading, with a 3-way ablation study and zero-shot cross-dataset generalization test.",
    "Phase 1: Introduce a Grade-1-sensitivity-focused training variant to improve detection of the earliest, hardest-to-catch disease stage.",
    "Phase 2 (add-on): Extend the shared backbone with a diabetic macular edema detection head trained on a combined Messidor + IDRiD cohort.",
    "Phase 2 (add-on): Quantitatively validate Grad-CAM and TCAV explanations against real expert-annotated lesion masks and clinically-grounded concept probes.",
], size=18, space_after=16)
add_footer(s)

# =============================================================================
# 5. BASE PAPER DETAILS
# =============================================================================
s = new_slide(); title(s, "Base Paper Details")
box = body_box(s, top=1.8)
tf = box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Base Paper Title: "; r.font.bold = True; r.font.size = Pt(18); r.font.color.rgb = BLACK
r2 = p.add_run(); r2.text = "QuantumNet: An Enhanced Diabetic Retinopathy Detection Model Using Classical Deep Learning-Quantum Transfer Learning"
r2.font.size = Pt(18); r2.font.color.rgb = DARK
p.space_after = Pt(20)

p2 = tf.add_paragraph()
r = p2.add_run(); r.text = "Year: "; r.font.bold = True; r.font.size = Pt(18); r.font.color.rgb = BLACK
r2 = p2.add_run(); r2.text = "2025"; r2.font.size = Pt(18); r2.font.color.rgb = DARK
p2.space_after = Pt(20)

p3 = tf.add_paragraph()
r = p3.add_run(); r.text = "Journal Name: "; r.font.bold = True; r.font.size = Pt(18); r.font.color.rgb = BLACK
r2 = p3.add_run(); r2.text = "MethodsX (ScienceDirect)"; r2.font.size = Pt(18); r2.font.color.rgb = DARK
add_footer(s)

# =============================================================================
# 6-10. LITERATURE SURVEY (5 slides)
# =============================================================================
lit = [
    ("Bali, Mishra, Yenkikar & Chikmurge", "MethodsX (ScienceDirect)\n2025",
     "QuantumNet: An Enhanced Diabetic Retinopathy Detection Model Using Classical Deep Learning-Quantum Transfer Learning",
     "CNN (DRNet) + PCA + 4-qubit Variational Quantum Circuit + Grad-CAM; tested on APTOS, IDRiD, EyePACS.",
     "Quantum-hybrid model validated across three real datasets; includes Grad-CAM.",
     "CNN backbone, not ViT; DR only, no DME; heatmap not checked against expert masks; no TCAV."),
    ("Jain, Gupta & Singhal", "arXiv\n2024",
     "Diabetic Retinopathy Detection Using Quantum Transfer Learning",
     "ResNet-18 / Inception-V3 features + Variational Quantum Classifier on APTOS 2019.",
     "Early, clear demonstration that quantum transfer learning is viable for DR.",
     "CNN backbone; no explainability; no DME; single dataset only."),
    ("De, Nandy Pal & Hazra", "Evolutionary Intelligence\n2024",
     "FQPDR: Federated Quantum Neural Network for Privacy-Preserving Early Detection of Diabetic Retinopathy",
     "Federated-learning quantum neural network for DR detection without centralizing patient data.",
     "Addresses data-privacy, a real deployment concern for medical imaging.",
     "No explainability; DR only; no ViT backbone."),

    ("Menaka, Muthusamy & Sidhu", "Scientific Reports\n2026",
     "A Novel Diabetic Retinopathy Detection From Fundus Images Using Hybrid Quantum Convolutional Neural Network Models",
     "Hybrid QCNN focused on optic-disc boundary detection challenges in DR screening.",
     "Recent; tackles a specific, real preprocessing weak point (optic-disc boundaries).",
     "DR only; no explainability validation; no DME."),
    ("Bhoopalan, Sekar, Nagaprasad, et al.", "Scientific Reports\n2025",
     "Task Optimized Vision Transformer for Diabetic Retinopathy Detection and Classification in Resource Constrained Early Diagnosis Settings",
     "A ViT tuned (layer-wise LR scheduling, attention-head tuning) for low-resource deployment.",
     "Efficient ViT suited to low-cost hardware / rural screening settings.",
     "Purely classical, no quantum component; no explainability; DR only."),
    ("Mohan, Murugan, Goel & Roy", "IEEE R10-HTC\n2022",
     "ViT-DR: Vision Transformers in Diabetic Retinopathy Grading Using Fundus Images",
     "Patch-based Vision Transformer for DR severity classification.",
     "Solid classical ViT baseline for DR grading (91.4% accuracy reported).",
     "No quantum component; no explainability; no DME."),

    ("Nazih, Aseeri, Atallah & El-Sappagh", "IEEE Access\n2023",
     "Vision Transformer Model for Predicting the Severity of Diabetic Retinopathy in Fundus Photography-Based Retina Images",
     "ViT-based severity prediction model for DR from fundus photographs.",
     "Strong ViT reference for severity-level grading.",
     "Classical only, no explainability output, no DME."),
    ("Li, Hu, Yu, Zhu, Fu & Heng", "IEEE Trans. Medical Imaging\n2019",
     "CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading",
     "Cross-disease attention network jointly grading DR and DME with only image-level supervision.",
     "Real precedent for grading both diseases from one model.",
     "Fully classical CNN-attention architecture; no quantum component; no quantitative XAI."),
    ("Bressler, Aviv, Margalit, et al.", "Ophthalmology Science\n2025",
     "Autonomous Screening for Diabetic Macular Edema Using Deep Learning Processing of Retinal Images",
     "Large-scale (32,049-image) deep learning DME screening model validated on EyePACS + IDRiD.",
     "Large validation cohort; strong sensitivity/specificity and AUC (~0.95).",
     "DME only, no DR grading; classical CNN; no explainability; no quantum component."),

    ("Takinami, Morikawa & Oshika", "medRxiv\n2025",
     "Pre-Trained Vision Transformer With Masked Autoencoder for Automated Diabetic Macular Edema Detection From Optical Coherence Tomography Images",
     "Self-supervised (Masked Autoencoder) ViT pretraining for DME detection on OCT images.",
     "Strong self-supervised ViT pretraining strategy for DME.",
     "OCT-based, not fundus photographs; classical only; no quantum; no DR grading."),
    ("Van Craenendonck, Elen, Gerrits & De Boever", "Translational Vision Science & Technology\n2020",
     "Systematic Comparison of Heatmapping Techniques in Deep Learning in the Context of Diabetic Retinopathy Lesion Detection",
     "Compares Grad-CAM and other heatmap methods quantitatively against IDRiD expert lesion masks.",
     "Rigorous quantitative heatmap-vs-ground-truth methodology, directly informs this project's validation approach.",
     "No quantum component; no TCAV; DR only, no DME."),

    ("Kim, Wattenberg, Gilmer, et al.", "Proc. ICML, PMLR vol. 80\n2018",
     "Interpretability Beyond Feature Attribution: Quantitative Testing With Concept Activation Vectors (TCAV)",
     "Foundational concept-based explainability method used directly as this project's TCAV module.",
     "Introduces the human-concept-level explanation approach this project builds on.",
     "General-purpose method, not originally applied to retinal images; no quantum component."),
    ("Djoumessi, Huang, Kuhlewein, et al.", "PLOS Digital Health\n2025",
     "An Inherently Interpretable AI Model Improves Screening Speed and Accuracy for Early Diabetic Retinopathy",
     "Interpretable-by-design DR screening model, clinically validated for speed and early-stage accuracy.",
     "Strong clinical validation of early-stage detection speed and accuracy gains.",
     "Classical, no quantum; no DME; interpretability is architectural, not concept-based (no TCAV)."),
]

col_widths = [1.7, 1.5, 2.5, 2.7, 1.9, 1.9]
slide_groups = [lit[0:3], lit[3:6], lit[6:9], lit[9:11], lit[11:13]]
for idx, group in enumerate(slide_groups, start=1):
    s = new_slide()
    title(s, f"LITERATURE SURVEY({idx}/5)", size=28)
    rows = 1 + len(group)
    table = make_table(s, rows, 6, 0.25, 1.35, sum(col_widths), 0.9 + 1.5 * len(group), col_widths)
    style_header_row(table, ["AUTHOR", "JOURNAL NAME &\nYEAR OF PUBLICATION", "TITLE", "DESCRIPTIONS", "PROS", "CONS"])
    for r, (author, jy, title_, desc, pros, cons) in enumerate(group, start=1):
        shade = ROW_A if r % 2 else ROW_B
        set_cell(table, r, 0, author, size=9.5, row_shade=shade)
        set_cell(table, r, 1, jy, size=9.5, row_shade=shade)
        set_cell(table, r, 2, title_, size=9, row_shade=shade)
        set_cell(table, r, 3, desc, size=9, row_shade=shade)
        set_cell(table, r, 4, pros, size=9, row_shade=shade)
        set_cell(table, r, 5, cons, size=9, row_shade=shade)
    add_footer(s)

# =============================================================================
# 11. INFERENCES FROM LITERATURE SURVEY
# =============================================================================
s = new_slide(); title(s, "INFERENCES FROM LITERATURE SURVEY", size=28)
box = body_box(s, top=1.5)
tf = box.text_frame; tf.word_wrap = True
bullets(tf, [
    "Quantum-assisted DR models in the literature are all built on CNN backbones (QuantumNet, FQPDR, etc.); none combine a Vision Transformer with a variational quantum circuit.",
    "Joint DR and DME grading exists (e.g., CANet) but only with fully classical architectures - no quantum-hybrid system grades both diseases from one shared backbone.",
    "Grad-CAM is used in quantum-hybrid DR papers, but none quantitatively validate the heatmap against expert pixel-level lesion masks the way classical DR literature (Van Craenendonck et al.) does.",
    "TCAV has been applied to DR with clinically meaningful concepts, but never combined with a quantum-hybrid backbone or extended to a second disease (DME).",
    "No reviewed quantum or classical joint DR/DME system reports an honest zero-shot cross-dataset generalization test or a single, fully-traceable worked example.",
    "Grade-1 (earliest-stage) sensitivity is rarely reported as its own tunable metric, despite early detection being the clinically hardest and most important case.",
], size=16, space_after=14)
add_footer(s)

# =============================================================================
# 12. PROPOSED SYSTEM (1/4)
# =============================================================================
s = new_slide(); title(s, "PROPOSED SYSTEM (1/4)", size=28)
box = body_box(s, top=1.35, height=2.1)
tf = box.text_frame; tf.word_wrap = True
p0 = tf.paragraphs[0]; p0.text = "Phase 1:"; p0.font.bold = True; p0.font.size = Pt(15); p0.font.color.rgb = BLACK
bullets(tf, [
    "Frozen ViT-Base/16 backbone extracts a single feature vector from the preprocessed fundus image.",
    "A linear reducer compresses the 768-dim [CLS] embedding to 4 values, angle-encoded into a 4-qubit variational quantum circuit.",
    "A final linear layer maps the circuit's Pauli-Z measurements to a 5-class DR grade.",
    "Validated with a 3-way ablation (ViT-only / VQC-only / hybrid), a Grade-1-sensitivity training variant, and zero-shot testing on 3 unseen datasets.",
], size=14, first=False, space_after=5)

box2 = body_box(s, top=3.55, height=2.0)
tf2 = box2.text_frame; tf2.word_wrap = True
p0 = tf2.paragraphs[0]; p0.text = "Phase 2 (add-on to Phase 1):"; p0.font.bold = True; p0.font.size = Pt(15); p0.font.color.rgb = BLACK
bullets(tf2, [
    "Adds a second head from the same shared backbone for binary DME detection.",
    "Adds Grad-CAM spatial explanation, quantitatively validated (IoU / precision / recall) against real IDRiD pixel-level expert lesion masks.",
    "Adds TCAV concept explanation using probes trained on real cropped lesion patches (microaneurysm, hemorrhage, exudate, edema).",
    "Adds one complete worked example tracing a real image through every stage with real, checkable intermediate numbers.",
], size=14, first=False, space_after=5)

adv_title = s.shapes.add_textbox(Inches(0.6), Inches(5.75), Inches(6), Inches(0.35))
adv_title.text_frame.paragraphs[0].text = "Advantages of Proposed System"
adv_title.text_frame.paragraphs[0].font.bold = True
adv_title.text_frame.paragraphs[0].font.size = Pt(15)
adv_title.text_frame.paragraphs[0].font.color.rgb = BLACK
adv_box = body_box(s, top=6.1, height=0.85)
tfa = adv_box.text_frame; tfa.word_wrap = True
bullets(tfa, [
    "Integrates a quantum-hybrid ViT with a joint dual-disease output.",
    "Generates dual, quantitatively-validated explainability (Grad-CAM + TCAV), not just example pictures.",
    "Reports honest cross-dataset and Grade-1 sensitivity results.",
    "Improves practical verifiability beyond a single accuracy number.",
], size=11, space_after=1)
add_footer(s)

# =============================================================================
# 13. PROPOSED SYSTEM(2/4) - Architecture diagram
# =============================================================================
s = new_slide(); title(s, "PROPOSED SYSTEM(2/4)", size=28)
try:
    pic_h = Inches(5.3)
    pic_w = Emu(int(pic_h * 1704 / 1218))
    pic_left = Emu(int((SLIDE_W - pic_w) / 2))
    s.shapes.add_picture("paper/fig_architecture.png", pic_left, Inches(1.5), height=pic_h)
except Exception:
    ph = s.shapes.add_textbox(Inches(2), Inches(3), Inches(9), Inches(1))
    ph.text_frame.paragraphs[0].text = "[Insert architecture diagram: paper/fig_architecture.png]"
add_footer(s)

# =============================================================================
# 14. PROPOSED SYSTEM(3/4) - Module table
# =============================================================================
s = new_slide(); title(s, "PROPOSED SYSTEM(3/4)", size=28)
modules = [
    ("Requirement Analysis", "Define DR grading + DME detection + dual-explainability requirements; select APTOS, Messidor, IDRiD, and EyePACS datasets."),
    ("Data Preparation", "Resize/normalize fundus images (224x224); build a combined Messidor+IDRiD DME cohort; crop real lesion patches from IDRiD segmentation masks for TCAV probe training."),
    ("Model Development - Phase 1", "Train the frozen ViT-Base/16 + linear reducer + 4-qubit VQC pipeline; run the 3-way ablation study; train the Grade-1-sensitivity variant."),
    ("Model Development - Phase 2", "Add the DME head; implement fundus-masked Grad-CAM; train TCAV concept probes on real lesion crops."),
    ("System Integration", "Integrate the inference pipeline into a results dashboard showing grade, confidence, DME result, risk level, and explanation summary."),
    ("Testing & Deployment", "Run zero-shot cross-dataset evaluation; validate Grad-CAM faithfulness (IoU/recall/precision) against expert masks; deploy the end-to-end worked-example demo."),
]
table = make_table(s, len(modules) + 1, 2, 0.4, 1.4, 12.5, 5.5, [3.5, 9.0])
style_header_row(table, ["Modules", "Activities"])
for r, (m, a) in enumerate(modules, start=1):
    shade = ROW_A if r % 2 else ROW_B
    set_cell(table, r, 0, m, size=11, bold=True, row_shade=shade)
    set_cell(table, r, 1, a, size=10.5, row_shade=shade)
add_footer(s)

# =============================================================================
# 15. PROPOSED SYSTEM-Algorithm(4/4)
# =============================================================================
s = new_slide(); title(s, "PROPOSED SYSTEM-Algorithm(4/4)", size=26)
box = body_box(s, top=1.5)
tf = box.text_frame; tf.word_wrap = True
bullets(tf, [
    "Image Preprocessing (Resizing to 224x224, Normalization)",
    "ViT-Base/16 Patch Embedding + Self-Attention Feature Extraction (frozen backbone)",
    "Linear Dimensionality Reduction (768 to 4)",
    "4-Qubit Variational Quantum Circuit (angle encoding, entangling layers, Pauli-Z measurement)",
    "Linear Classification Head (5-class DR grade / binary DME)",
    "Gradient-weighted Class Activation Mapping (Grad-CAM) with fundus-mask background exclusion - Phase 2",
    "Testing with Concept Activation Vectors (TCAV) using real lesion-patch-trained probes - Phase 2",
    "Zero-Shot Cross-Dataset Evaluation Protocol",
], size=17, space_after=12)
add_footer(s)

# =============================================================================
# 16. References
# =============================================================================
s = new_slide(); title(s, "References", size=32)
box = body_box(s, top=1.3, height=5.6)
tf = box.text_frame; tf.word_wrap = True
refs = [
 "Bali, M., Mishra, V. P., Yenkikar, A., & Chikmurge, D. (2025). QuantumNet: An enhanced diabetic retinopathy detection model using classical deep learning-quantum transfer learning. MethodsX (ScienceDirect).",
 "Bhoopalan, R., Sekar, P., Nagaprasad, N., et al. (2025). Task optimized vision transformer for diabetic retinopathy detection and classification in resource constrained early diagnosis settings. Scientific Reports, 15, 39047.",
 "Bressler, I., Aviv, R., Margalit, D., Cohen, G. Y., Ianchulev, T., Savant, S. V., Ramsey, D. J., & Dvey-Aharon, Z. (2025). Autonomous screening for diabetic macular edema using deep learning processing of retinal images. Ophthalmology Science, 5(4), 100722.",
 "De, D., Nandy Pal, M., & Hazra, D. (2024). FQPDR: Federated quantum neural network for privacy-preserving early detection of diabetic retinopathy. Evolutionary Intelligence, 17, 4047-4068.",
 "Djoumessi, K., Huang, Z., Kuhlewein, L., Rickmann, A., Simon, N., Koch, L. M., & Berens, P. (2025). An inherently interpretable AI model improves screening speed and accuracy for early diabetic retinopathy. PLOS Digital Health, 4(5), e0000831.",
 "Jain, A., Gupta, R., & Singhal, J. (2024). Diabetic retinopathy detection using quantum transfer learning. arXiv:2405.01734.",
 "Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). Proc. 35th ICML, PMLR 80.",
 "Li, X., Hu, X., Yu, L., Zhu, L., Fu, C.-W., & Heng, P.-A. (2019). CANet: Cross-disease attention network for joint diabetic retinopathy and diabetic macular edema grading. IEEE Trans. Medical Imaging, 39(5), 1483-1493.",
 "Menaka, S. R., Muthusamy, S., & Sidhu, P. K. (2026). A novel diabetic retinopathy detection from fundus images using hybrid quantum convolutional neural network models. Scientific Reports, 16.",
 "Mohan, N. J., Murugan, R., Goel, T., & Roy, P. (2022). ViT-DR: Vision transformers in diabetic retinopathy grading using fundus images. Proc. IEEE 10th Region 10 Humanitarian Technology Conf. (R10-HTC), 167-172.",
 "Nazih, W., Aseeri, A. O., Atallah, O. Y., & El-Sappagh, S. (2023). Vision transformer model for predicting the severity of diabetic retinopathy in fundus photography-based retina images. IEEE Access, 11, 117546-117561.",
 "Takinami, S., Morikawa, S., & Oshika, T. (2025). Pre-trained vision transformer with masked autoencoder for automated diabetic macular edema detection from optical coherence tomography images. medRxiv.",
 "Van Craenendonck, T., Elen, B., Gerrits, N., & De Boever, P. (2020). Systematic comparison of heatmapping techniques in deep learning in the context of diabetic retinopathy lesion detection. Translational Vision Science & Technology, 9(2), 64.",
]
for i, ref in enumerate(refs):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = "•  " + ref
    p.font.size = Pt(10.5)
    p.font.color.rgb = DARK
    p.space_after = Pt(5)
add_footer(s)

# =============================================================================
# 17. THANK YOU
# =============================================================================
s = new_slide(); frame(s)
box = s.shapes.add_textbox(Inches(1), Inches(2.8), Inches(11.3), Inches(1.2))
tf = box.text_frame
p = tf.paragraphs[0]; p.text = "THANK YOU"; p.font.size = Pt(44); p.font.bold = True
p.font.color.rgb = BLACK; p.alignment = PP_ALIGN.CENTER
box2 = s.shapes.add_textbox(Inches(1.5), Inches(4.2), Inches(10.3), Inches(1.0))
tf2 = box2.text_frame; tf2.word_wrap = True
p2 = tf2.paragraphs[0]
p2.text = "We thank our Department, Guide, and Panel Members for their guidance and support throughout Phase I of this project."
p2.font.size = Pt(16); p2.font.color.rgb = DARK; p2.alignment = PP_ALIGN.CENTER
add_footer(s)

prs.save("paper/QuantumRetina-X_Phase1_Review_v2.pptx")
print("FINAL SAVED: paper/QuantumRetina-X_Phase1_Review_v2.pptx")
