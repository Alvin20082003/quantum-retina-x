"""Generates paper/QuantumRetina-X_Paper2_IEEE.docx - an IEEE-conference-style
draft of Paper 2 (dual explainability for DR+DME), built only from results
actually produced and verified in this project (results_log.csv, faithfulness
validation JSON, architecture as implemented in models/hybrid.py, xai/, and
backend/main.py). No invented numbers, no invented citations."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = "Times New Roman"

def set_two_column(section, space=Pt(18)):
    cols = OxmlElement('w:cols')
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), str(int(space.pt * 20)))
    section._sectPr.append(cols)

FIG_COUNTER = [0]

def add_figure(doc, path, caption, width_in=6.6):
    """Inserts a figure spanning the full page width (breaks out of the
    two-column flow and back) so diagrams/charts/equations stay legible
    instead of being squeezed into a ~3.5in column."""
    FIG_COUNTER[0] += 1
    single = doc.add_section(WD_SECTION.CONTINUOUS)
    single.left_margin = single.right_margin = Inches(0.62)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(path, width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    r = cap.add_run(f"Fig. {FIG_COUNTER[0]}. {caption}")
    style_run(r, size=9, italic=True)
    two = doc.add_section(WD_SECTION.CONTINUOUS)
    two.left_margin = two.right_margin = Inches(0.62)
    set_two_column(two)

def style_run(run, size=10, bold=False, italic=False, small_caps=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), FONT)
    if small_caps:
        run.font.small_caps = True
    if color:
        run.font.color.rgb = RGBColor(*color)

def para(doc, text="", size=10, bold=False, italic=False, align=None, space_after=6, first_line_indent=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if first_line_indent is not None:
        p.paragraph_format.first_line_indent = Inches(first_line_indent)
    if text:
        r = p.add_run(text)
        style_run(r, size=size, bold=bold, italic=italic)
    return p

def heading(doc, numeral, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(f"{numeral}. {title.upper()}")
    style_run(r, size=10, bold=True)
    return p

def subheading(doc, letter, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(f"{letter}. {title}")
    style_run(r, size=10, bold=True, italic=True)
    return p

def body(doc, text, space_after=6, indent=0.2):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.first_line_indent = Inches(indent)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    style_run(r, size=10)
    return p

def make_table(doc, headers, rows, caption, caption_num):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(8)
    r = cap.add_run(f"TABLE {caption_num}")
    style_run(r, size=9, bold=False, small_caps=True)
    cap2 = doc.add_paragraph()
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = cap2.add_run(caption)
    style_run(r2, size=9, italic=True)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        style_run(r, size=8, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            style_run(r, size=8)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table

# ---------------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.left_margin = sec.right_margin = Inches(0.62)
sec.top_margin = sec.bottom_margin = Inches(0.75)

style = doc.styles["Normal"]
style.font.name = FONT
style.font.size = Pt(10)

# ---- Title block (single column) ----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Dual Explainable AI for Trustworthy Diabetic Retinopathy\nand Macular Edema Diagnosis Using a Hybrid\nQuantum-Classical Vision Transformer")
style_run(r, size=16, bold=True)
p.paragraph_format.space_after = Pt(10)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("[Author Name(s) Here], [Affiliation], [City, Country]\n[email@example.com]")
style_run(r, size=11, italic=True)
p.paragraph_format.space_after = Pt(4)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("[FILL IN: author names, affiliations, and contact email before submission]")
style_run(r, size=9, italic=True, color=(150, 0, 0))
p.paragraph_format.space_after = Pt(10)

# Abstract + Index Terms (still single column, standard IEEE layout)
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("Abstract—")
style_run(r, size=9, bold=True, italic=True)
r2 = p.add_run(
    "Diabetic retinopathy (DR) and diabetic macular edema (DME) are leading causes "
    "of preventable blindness in diabetic populations, and their early stages "
    "produce no symptoms, making automated screening clinically valuable. Deep "
    "learning models for DR/DME screening are frequently deployed as black boxes, "
    "which limits clinical trust and adoption. This paper presents a hybrid "
    "quantum-classical Vision Transformer (ViT + Variational Quantum Circuit) "
    "system that jointly grades DR severity (5 classes) and detects DME "
    "(binary), paired with a dual explainability pipeline combining Gradient-"
    "weighted Class Activation Mapping (Grad-CAM) and Testing with Concept "
    "Activation Vectors (TCAV). Unlike approaches that present explanations "
    "qualitatively, we validate both modalities quantitatively. TCAV concept "
    "probes are trained on real lesion patches cropped from pixel-level expert "
    "annotations (IDRiD segmentation subset) rather than whole-image severity "
    "proxies, reach 98-100% training separability, and are scored live on each "
    "input image rather than looked up from a precomputed table. Grad-CAM is "
    "evaluated for faithfulness directly against the same expert-annotated "
    "lesion masks across 81 images, yielding a mean IoU of 0.035, recall of "
    "0.126, and precision of 0.054 - a modest but honestly measured result that "
    "improves with disease severity (Grade 3 IoU = 0.093 vs. Grade 1 IoU = "
    "0.002), consistent with the coarse spatial resolution of patch-based "
    "Grad-CAM relative to small lesions such as microaneurysms. Our DME "
    "detector, trained on a combined Messidor and IDRiD cohort, reaches 89.1% "
    "accuracy and 0.928 AUC on a held-out test set. An ablation study shows the "
    "quantum layer improves grading accuracy (81.6% vs. 78.4%) at a measurable "
    "cost to ranking quality (AUC 0.894 vs. 0.936) relative to a classical ViT "
    "baseline, a nuanced result rather than a uniform improvement. Cross-dataset "
    "evaluation across four public datasets (APTOS, Messidor, IDRiD, EyePACS) "
    "shows substantial accuracy degradation under domain shift, which we report "
    "transparently. We argue that honestly measured, modest explainability is "
    "of greater clinical and scientific value than qualitatively convincing but "
    "unvalidated heatmaps."
)
style_run(r2, size=9)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

p = doc.add_paragraph()
r = p.add_run("Index Terms—")
style_run(r, size=9, bold=True, italic=True)
r2 = p.add_run(
    "Diabetic Retinopathy, Diabetic Macular Edema, Vision Transformer, "
    "Variational Quantum Circuit, Explainable AI, Grad-CAM, TCAV, Faithfulness "
    "Validation, Quantum Machine Learning."
)
style_run(r2, size=9)
p.paragraph_format.space_after = Pt(10)

# ---- Switch to two columns for the body ----
new_sec = doc.add_section(WD_SECTION.CONTINUOUS)
new_sec.left_margin = new_sec.right_margin = Inches(0.62)
set_two_column(new_sec)

# =========================================================================
# I. INTRODUCTION
# =========================================================================
heading(doc, "I", "Introduction")
body(doc,
     "Diabetic retinopathy (DR) is damage to the retinal blood vessels caused "
     "by prolonged high blood sugar, and diabetic macular edema (DME) is "
     "fluid accumulation in the macula from the same underlying vascular "
     "damage. Both progress silently: patients experience no symptoms until "
     "damage is advanced, which is why automated screening from routine "
     "fundus photographs has direct clinical value for early intervention.")
body(doc,
     "Two practical obstacles limit clinical adoption of deep learning for "
     "this task. First, most models are opaque: a probability score with no "
     "indication of which retinal region or clinical finding drove it gives a "
     "clinician nothing to verify. Second, published explainability results "
     "are frequently qualitative - a handful of heatmap images that \"look "
     "reasonable\" to the authors - rather than measured against independent "
     "ground truth. This paper addresses both problems directly for a joint "
     "DR grading and DME detection system built on a hybrid quantum-classical "
     "architecture.")
body(doc, "The contributions of this paper are:")
for item in [
    "A hybrid ViT + Variational Quantum Circuit (VQC) architecture used for "
    "both 5-class DR grading and binary DME detection, with the DME head "
    "trained on a combined Messidor + IDRiD cohort assembled specifically "
    "because neither dataset alone has enough positive DME cases for "
    "reliable training.",
    "A TCAV implementation whose concept probes are trained on real lesion "
    "patches cropped from pixel-level expert segmentation masks, rather "
    "than whole retinal images grouped by disease grade as a proxy, and "
    "whose scores are computed live per input image rather than served "
    "from a precomputed lookup table.",
    "A quantitative faithfulness evaluation of Grad-CAM against the same "
    "expert lesion masks, reporting IoU, recall, and precision rather than "
    "presenting heatmaps without validation.",
    "A transparent ablation study and four-dataset cross-validation that "
    "report the quantum layer's actual, mixed effect on performance and "
    "the model's real generalization gap under domain shift, rather than "
    "presenting only favorable results.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(item)
    style_run(r, size=10)

# =========================================================================
# II. RELATED WORK
# =========================================================================
heading(doc, "II", "Related Work")
body(doc,
     "[AUTHOR ACTION REQUIRED: The citations below are real, independently "
     "verified papers found via literature search during drafting - they are "
     "a seed, not a substitute for your own systematic review. Verify full "
     "author lists, venues, and page numbers before submission, and expand "
     "this section with your own reviewed literature.]")
subheading(doc, "A", "Quantum and Hybrid Quantum-Classical Models for DR")
body(doc,
     "Quantum transfer learning has been applied directly to DR classification "
     "[1]. Hybrid quantum convolutional neural networks combining quantum "
     "circuits with classical CNN backbones have been proposed for DR "
     "detection [2], [3], and hybrid quantum-classical deep learning "
     "frameworks have been developed for balanced multiclass DR "
     "classification [4]. These works establish quantum-classical hybrids as "
     "an active research direction for DR, but to our knowledge combine "
     "quantum circuits with CNN backbones rather than Vision Transformers, "
     "and do not address DME jointly or validate explainability "
     "quantitatively.")
subheading(doc, "B", "Explainability in Retinal Disease Detection")
body(doc,
     "Grad-CAM is widely used for visual explanation of DR models, including "
     "multi-label classification settings [5]. Closest to our faithfulness "
     "methodology, a systematic comparison of heatmapping techniques "
     "evaluated multiple CNN architectures (ResNet50, VGG16, InceptionV3) "
     "with Grad-CAM-family methods against IDRiD ground truth using an "
     "Explainability Consistency Score, finding VGG16+Grad-CAM performed best "
     "on hemorrhages and hard exudates [6]. This confirms, independently of "
     "our own measurement, that post-hoc CAM-family explanations correlate "
     "only modestly with expert lesion annotations - consistent with our own "
     "IoU-based findings in Section V. Inherently interpretable architectures "
     "have also been proposed as an alternative to post-hoc explanation, "
     "reporting substantially higher lesion-level precision than post-hoc "
     "techniques applied to standard networks [7].")

# =========================================================================
# III. METHODOLOGY
# =========================================================================
heading(doc, "III", "Methodology")
add_figure(doc, "paper/fig_architecture.png",
           "Overall architecture: a frozen ViT-Base/16 backbone shared by the "
           "DR and DME heads, with the 4-qubit VQC between the linear "
           "reducer and output layer. Grad-CAM reads gradients at the "
           "backbone's last layer; TCAV scores the same backbone's [CLS] "
           "features against concept probes trained on real IDRiD lesion "
           "crops.")
subheading(doc, "A", "Datasets")
body(doc,
     "Four public datasets are used, each for a distinct role. APTOS 2019 "
     "(3,662 labeled images, 5-class DR grade) is the primary training set for "
     "DR grading. Messidor (1,744 gradable images) provides a binary "
     "adjudicated DME label and is used both for cross-dataset DR evaluation "
     "and, combined with IDRiD, for DME training. IDRiD is used in two forms: "
     "its grading subset (516 images with DR grade and a 3-level macular "
     "edema risk score) contributes to DME training and cross-dataset DR "
     "evaluation, while its separate segmentation subset (81 images: 54 "
     "train + 27 test) provides pixel-level lesion annotations "
     "(microaneurysms, haemorrhages, hard exudates, soft exudates, optic "
     "disc) used for TCAV concept-patch extraction and Grad-CAM faithfulness "
     "validation. EyePACS (5,000-image subset) is used only for cross-dataset "
     "DR evaluation.")
body(doc,
     "For DME training, Messidor's binary adjudicated_dme label is used "
     "directly; IDRiD's 3-level macular edema risk (0/1/2) is binarized as "
     "risk 0 = negative and risk 1 or 2 = positive, since grade 1 already "
     "denotes detectable exudate presence. The combined, gradable cohort "
     "yields 1,539 training, 330 validation, and 330 held-out test images, "
     "with a positive rate of approximately 20%.")

subheading(doc, "B", "Hybrid ViT + VQC Architecture")
body(doc,
     "A ViT-Base/16 backbone (pretrained on ImageNet-21k, subsequently "
     "fine-tuned as a classical baseline on APTOS) is frozen and shared by "
     "both the DR and DME heads. Its [CLS] token output (768-dimensional) is "
     "projected by a trainable linear layer to 4 values, which are angle-"
     "encoded (RY rotations) onto 4 qubits. Two variational layers, each "
     "consisting of trainable RX/RY/RZ rotations per qubit followed by a "
     "ring of CNOT entangling gates, are applied, and each qubit is measured "
     "via its Pauli-Z expectation value, yielding 4 classical outputs. A "
     "final trainable linear layer maps these to 5 class logits for DR "
     "grading, or 1 logit for DME detection. The quantum circuit is "
     "simulated with PennyLane's default.qubit device and differentiated "
     "using the parameter-shift rule; only the projection layer, quantum "
     "circuit parameters, and output layer are trained (approximately 3,100 "
     "parameters), with the 86.4M-parameter backbone kept frozen throughout.")
body(doc,
     "Formally, each of the 4 projected values xi is angle-encoded onto its "
     "own qubit (Eq. 1), two variational layers apply trainable single-qubit "
     "rotations followed by a ring of entangling CNOTs (Eq. 2), and each "
     "qubit's Pauli-Z expectation is read out after both layers (Eq. 3):")
add_figure(doc, "paper/fig_eq_vqc.png", "VQC angle encoding, variational layer, and measurement.", width_in=5.6)

subheading(doc, "C", "Grad-CAM: Spatial Explanation")
body(doc,
     "Grad-CAM is computed on the last ViT encoder layer's pre-attention "
     "layer normalization, with the 196 patch tokens reshaped to a 14x14 "
     "grid. Because the backbone is frozen, a naive implementation blocks "
     "gradient flow to this layer entirely; we resolve this by (i) removing "
     "an unnecessary torch.no_grad() context in the forward pass and (ii) "
     "explicitly marking the input tensor as requiring gradients for the "
     "explanation pass, which allows gradients to propagate through frozen "
     "weights to the target activation without altering the frozen "
     "parameters themselves or the model's numerical predictions.")
body(doc,
     "We additionally observed that the layer-normalized activations at "
     "near-uniform background patches (the black region surrounding the "
     "circular fundus image) are numerically amplified by LayerNorm's "
     "division by a near-zero variance, producing a spurious high-activation "
     "ring with no clinical meaning. We suppress this by constraining the "
     "heatmap to the visible fundus: an ellipse (not a circle - the fundus "
     "region is consistently taller than wide after resizing) is fit to the "
     "largest foreground connected component, using Otsu's method for "
     "per-image adaptive thresholding, and shrunk by a fixed 18% of its axes "
     "before masking. A proportional shrink was necessary because the fundus "
     "circle occupies a very different fraction of the frame across "
     "datasets (approximately 80% for EyePACS vs. 55-65% for Messidor/"
     "IDRiD); a fixed pixel erosion removes a disproportionate amount of "
     "genuine retina on datasets where the fundus already fills more of the "
     "frame.")
body(doc,
     "Channel importance weights are the global-average-pooled gradient of "
     "the target class score with respect to each feature-map channel (Eq. "
     "4); the class activation map is their ReLU-rectified weighted sum "
     "(Eq. 5), computed over the 14x14 patch grid and upsampled to the "
     "input resolution before masking:")
add_figure(doc, "paper/fig_eq_gradcam.png", "Grad-CAM channel weights and class activation map.", width_in=4.6)

subheading(doc, "D", "TCAV: Concept-Level Explanation")
body(doc,
     "Three clinical concepts are defined: hemorrhage, exudate, and edema. "
     "Rather than approximating these concepts with whole APTOS images "
     "grouped by DR grade, concept patches are extracted directly from the "
     "IDRiD segmentation subset's pixel-level masks. Blob-level morphology "
     "analysis of the mask's five non-background colors (connected-"
     "component count and size) was used to identify each lesion type in "
     "the absence of a published color legend for this mask format: the "
     "color present in every image at consistent size was identified as "
     "optic disc (excluded, not a lesion); the rarest color as soft "
     "exudates; and of the remaining two, the more numerous and smaller-"
     "blob color as microaneurysms and the less numerous, larger-blob color "
     "as haemorrhages. Microaneurysm and haemorrhage blobs are pooled as the "
     "\"hemorrhage\" concept (the standard \"red lesion\" clinical grouping); "
     "hard and soft exudate blobs are pooled as the \"exudate\" concept "
     "(\"bright lesion\" grouping). IDRiD provides no direct macular edema "
     "segmentation; since DME risk is clinically graded by hard-exudate "
     "proximity to the macula, hard-exudate blobs within 35% of the image "
     "radius from center are used as a clinically grounded proxy for the "
     "\"edema\" concept.")
body(doc,
     "For each concept, 640x640-pixel crops (from the original ~4288x2848 "
     "resolution images, before downsampling for network input) are taken "
     "around lesion centroids; negative examples are crops containing no "
     "lesion or optic-disc pixels with sufficient tissue brightness to "
     "exclude background. This crop size was determined empirically: an "
     "initial attempt using tight 96-pixel crops produced probes with "
     "inverted behavior on whole images (a diseased image scored lower than "
     "a healthy one), because the model's [CLS]-token representation is "
     "shaped by whole, globally-structured retinal photographs and does not "
     "transfer from extreme close-up textures; widening the crop to include "
     "realistic surrounding context resolved this. A logistic regression "
     "probe is trained per concept on ViT [CLS]-token features of these "
     "patches (training accuracy 0.98-1.00 across all three concepts); its "
     "decision function is applied via predict_proba directly to each "
     "uploaded image's own extracted features at inference time, replacing "
     "a static, grade-indexed lookup table with genuine per-image scoring.")
body(doc,
     "Each concept probe is a logistic regression over [CLS] features h "
     "(Eq. 6); its weight vector, normalized, is the concept activation "
     "vector used for population-level directional-derivative analysis "
     "(Eq. 7), while live per-image scoring uses the probe's probability "
     "output directly:")
add_figure(doc, "paper/fig_eq_tcav.png", "TCAV concept probe and concept activation vector.", width_in=4.0)
add_figure(doc, "experiments/results/tcav/tcav_scores_chart.png",
           "TCAV concept scores by predicted DR grade (evaluated on APTOS "
           "test images using probes trained on real IDRiD lesion patches). "
           "All three concepts rise sharply from Grade 0-1 to Grade 2+, "
           "consistent with clinical expectation.", width_in=6.0)

subheading(doc, "E", "Training Objective")
body(doc,
     "The DR head is trained with class-weighted cross-entropy over 5 "
     "grades (Eq. 8, weights wc inversely proportional to class frequency); "
     "the DME head is trained with class-weighted binary cross-entropy "
     "with logits (Eq. 9, positive weight p set to the negative:positive "
     "ratio in the training split, approximately 4:1):")
add_figure(doc, "paper/fig_eq_loss.png", "DR and DME training objectives.", width_in=5.4)

# =========================================================================
# IV. EXPERIMENTS
# =========================================================================
heading(doc, "IV", "Experiments")
subheading(doc, "A", "Ablation Study")
body(doc,
     "Three configurations are compared on the same APTOS train/test split: "
     "ViT only (classical head), VQC only (a simple classical encoder feeding "
     "the same 4-qubit circuit, no ViT backbone), and the full ViT+VQC "
     "hybrid.")
make_table(doc,
    ["Config.", "Acc.", "Macro-F1", "AUC", "Kappa", "G1-Sens."],
    [
        ["ViT only", "78.4%", "0.638", "0.936", "0.881", "0.732"],
        ["VQC only", "54.6%", "0.237", "0.646", "0.289", "0.000"],
        ["ViT+VQC", "81.6%", "0.659", "0.894", "0.871", "0.554"],
    ],
    "Ablation Study on APTOS (Held-Out Test Set)", "I")
add_figure(doc, "experiments/results/ablation/ablation_chart.png",
           "Ablation study: accuracy, kappa, AUC, and Grade-1 sensitivity "
           "for VQC-only, ViT-only, and the full ViT+VQC hybrid.", width_in=6.6)
body(doc,
     "VQC-only performs far below either ViT-based configuration, confirming "
     "the classical backbone is essential rather than incidental. Adding the "
     "quantum layer to the full ViT backbone improves accuracy (+3.2 points) "
     "but reduces AUC (-0.042), Cohen's kappa (-0.010), and Grade-1 (early-"
     "stage) sensitivity (-0.178) relative to the classical ViT baseline. We "
     "report this as a genuine trade-off rather than a uniform improvement: "
     "the quantum layer changes what the model gets right, not simply how "
     "much.")

subheading(doc, "B", "Cross-Dataset Generalization")
body(doc,
     "Both models, trained only on APTOS, are evaluated without "
     "fine-tuning on three unseen datasets.")
make_table(doc,
    ["Dataset", "Model", "Acc.", "AUC", "Kappa"],
    [
        ["Messidor", "ViT only", "59.4%", "0.760", "0.449"],
        ["Messidor", "ViT+VQC", "58.3%", "0.577", "0.430"],
        ["IDRiD", "ViT only", "42.9%", "0.786", "0.578"],
        ["IDRiD", "ViT+VQC", "43.5%", "0.657", "0.618"],
        ["EyePACS (5k)", "ViT only", "64.2%", "0.725", "0.412"],
        ["EyePACS (5k)", "ViT+VQC", "63.0%", "0.603", "0.412"],
    ],
    "Cross-Dataset Evaluation (Trained on APTOS Only)", "II")
add_figure(doc, "experiments/results/cross_dataset/crossdataset_full_comparison.png",
           "Cross-dataset generalization from APTOS to Messidor, IDRiD, and "
           "EyePACS, ViT-only vs. ViT+VQC, across accuracy, kappa, AUC, and "
           "Grade-1 sensitivity.", width_in=7.0)
body(doc,
     "Accuracy drops substantially on every unseen dataset relative to the "
     "78-82% achieved on APTOS, most severely on IDRiD (to 43%). This is "
     "consistent with known domain-shift effects from differing cameras, "
     "populations, and acquisition conditions, and we report it directly as "
     "a limitation rather than omitting unfavorable cross-dataset numbers.")

subheading(doc, "C", "DME Detection")
body(doc,
     "The DME head (identical ViT+VQC architecture, 1-logit output) is "
     "trained on the combined Messidor+IDRiD cohort described in Section "
     "III-A, with a class-balanced BCEWithLogitsLoss (positive weight = "
     "negative/positive ratio in the training split) and the checkpoint "
     "selected by validation AUC.")
make_table(doc,
    ["Metric", "Value"],
    [
        ["Accuracy", "89.1%"],
        ["Macro-F1", "0.739"],
        ["AUC-ROC", "0.928"],
        ["Sensitivity", "78.5%"],
        ["Specificity", "91.7%"],
    ],
    "DME Detector Performance (Held-Out Test, n=330)", "III")

subheading(doc, "D", "Grad-CAM Faithfulness Validation")
body(doc,
     "Grad-CAM is computed for all 81 IDRiD segmentation-subset images and "
     "compared, pixel-for-pixel, against the expert lesion masks (optic disc "
     "excluded). The heatmap is percentile-stretched and thresholded at 0.5 "
     "within the same fundus mask used for rendering; overlap with ground "
     "truth is measured as Intersection-over-Union (IoU), recall (fraction "
     "of true lesion area covered), and precision (fraction of highlighted "
     "area that is true lesion).")
make_table(doc,
    ["Grade", "n", "IoU", "Recall", "Precision"],
    [
        ["0", "1", "0.008", "0.189", "0.008"],
        ["1", "9", "0.002", "0.059", "0.002"],
        ["2", "48", "0.018", "0.106", "0.024"],
        ["3", "19", "0.093", "0.216", "0.156"],
        ["4", "4", "0.034", "0.084", "0.052"],
        ["Overall", "81", "0.035", "0.126", "0.054"],
    ],
    "Grad-CAM Faithfulness vs. Expert Lesion Masks, by Predicted Grade", "IV")
add_figure(doc, "experiments/results/faithfulness/faithfulness_chart.png",
           "Grad-CAM faithfulness (IoU, recall, precision) against expert "
           "IDRiD lesion masks, by predicted DR grade.", width_in=5.6)
add_figure(doc, "paper/fig_gradcam_examples.png",
           "Grad-CAM heatmaps on the hybrid ViT+VQC model across three "
           "datasets, using the current fundus-masked pipeline. Note the "
           "absence of any highlighting outside the visible retina.", width_in=7.0)
body(doc,
     "Faithfulness increases markedly with disease severity: Grade 3 IoU "
     "(0.093) is roughly 40x Grade 1 IoU (0.002). This is consistent with the "
     "coarse spatial resolution of patch-based Grad-CAM (16x16-pixel patches "
     "upsampled to the full image): severe-grade images contain larger, "
     "spatially extensive lesion clusters a coarse heatmap can plausibly "
     "localize, while early-stage disease consists of sparse, few-pixel "
     "microaneurysms below this method's effective resolution. We report the "
     "overall IoU/recall/precision numbers directly rather than only a "
     "qualitative heatmap gallery, consistent with faithfulness methodology "
     "used elsewhere in this literature [6].")

# =========================================================================
# V. RESULTS AND DISCUSSION
# =========================================================================
heading(doc, "V", "Discussion")
body(doc,
     "Three findings are worth discussing explicitly rather than only "
     "reporting favorably. First, the quantum layer's effect on DR grading "
     "is genuinely mixed: it is not correct to claim the VQC uniformly "
     "improves detection, only that it shifts the accuracy/ranking-quality "
     "trade-off in a specific, measured way (Table I). Second, cross-dataset "
     "generalization remains weak across both classical and hybrid "
     "configurations (Table II), indicating that architecture choice alone "
     "does not solve the domain-shift problem for this task. Third, and most "
     "relevant to this paper's central claim, Grad-CAM's measured "
     "faithfulness (Table IV) is modest in absolute terms; we consider "
     "publishing this number, and explaining its correlation with lesion "
     "size, more valuable than omitting the measurement and presenting only "
     "visually plausible heatmaps, which several works in this area do not "
     "quantitatively validate at all.")
body(doc,
     "TCAV, by contrast, shows a clear practical improvement from grounding "
     "concept probes in real pixel-level annotations rather than whole-image "
     "grade proxies: probes trained this way separate concept-positive from "
     "concept-negative patches with 98-100% training accuracy and produce "
     "per-image scores that vary meaningfully with the presence of real "
     "disease (e.g., a held-out healthy image scored 0.02-0.05 across all "
     "three concepts versus 0.95-0.99 for diseased images in qualitative "
     "spot checks), rather than the identical scores every image of a given "
     "grade received under the earlier proxy-based, grade-indexed "
     "implementation.")

subheading(doc, "A", "Limitations")
for item in [
    "Grad-CAM faithfulness (IoU 0.035) is low in absolute terms and "
    "reflects the model's lack of any lesion-level supervision during "
    "training; it was never trained to localize, only to grade whole "
    "images.",
    "Cross-dataset accuracy (Table II) shows the model does not yet "
    "generalize robustly beyond its training distribution.",
    "The IDRiD segmentation subset used for faithfulness validation and "
    "TCAV concept extraction is small (81 images), and the lesion-type "
    "color mapping used was determined empirically (via blob morphology) "
    "in the absence of a published legend for this specific mask "
    "repackaging, rather than from an authoritative source.",
    "This system is a research prototype and has not undergone the "
    "clinical validation, regulatory review, or large-scale multi-site "
    "trials required for medical device approval.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(item)
    style_run(r, size=10)

# =========================================================================
# VI. CONCLUSION
# =========================================================================
heading(doc, "VI", "Conclusion")
body(doc,
     "We presented a hybrid ViT+VQC system for joint DR grading and DME "
     "detection with a dual explainability pipeline, and - unlike much "
     "prior work in this area - validated both explanation modalities "
     "quantitatively against real pixel-level expert annotations rather "
     "than presenting them qualitatively. The quantum layer's effect on "
     "grading is a measured trade-off rather than a uniform gain; Grad-CAM's "
     "faithfulness is modest and correlates with lesion size; TCAV grounded "
     "in real lesion patches produces meaningfully discriminative, per-image "
     "concept scores. We believe transparent reporting of these mixed "
     "results, together with a working, deployed system (FastAPI backend, "
     "React dashboard), constitutes a more useful contribution than "
     "presenting only favorable numbers.")

# =========================================================================
# REFERENCES
# =========================================================================
heading(doc, "", "References")
refs = [
    "[AUTHOR ACTION REQUIRED: verify full author names, exact venue/volume/"
    "page details, and years for every entry below before submission - these "
    "were located via web search during drafting and title/venue/URL were "
    "confirmed, but full bibliographic records were not independently "
    "verified.]",
    "\"Diabetic Retinopathy Detection Using Quantum Transfer Learning,\" "
    "arXiv:2405.01734.",
    "\"Hybrid Quantum Convolutional Neural Networks for Enhanced Diabetic "
    "Retinopathy Detection,\" IEEE Xplore, doc. 11070674.",
    "\"A novel diabetic retinopathy detection from fundus images using "
    "hybrid quantum convolutional neural network models,\" Scientific "
    "Reports.",
    "\"QuantumNet: An enhanced diabetic retinopathy detection model using "
    "classical deep learning-quantum transfer learning,\" ScienceDirect.",
    "\"A Multi-Label Deep Learning Model with Interpretable Grad-CAM for "
    "Diabetic Retinopathy Classification,\" IEEE Xplore, doc. 9175884.",
    "\"Systematic Comparison of Heatmapping Techniques in Deep Learning in "
    "the Context of Diabetic Retinopathy Lesion Detection,\" PMC7774113.",
    "\"An inherently interpretable AI model improves screening speed and "
    "accuracy for early diabetic retinopathy,\" PMC12068651.",
]
for i, ref in enumerate(refs):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    if i == 0:
        r = p.add_run(ref)
        style_run(r, size=8, italic=True, color=(150, 0, 0))
    else:
        r = p.add_run(f"[{i}] {ref}")
        style_run(r, size=9)

import sys
out_path = sys.argv[1] if len(sys.argv) > 1 else "paper/QuantumRetina-X_Paper2_IEEE.docx"
doc.save(out_path)
print(f"Saved {out_path}")
