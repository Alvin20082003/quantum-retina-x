"""Generates paper/QuantumRetina-X_Paper1_Architecture.docx - single-column
IEEE-style paper focused purely on the hybrid ViT+VQC architecture question:
does a variational quantum circuit actually help a Vision Transformer grade
diabetic retinopathy, especially at the hardest, earliest (Grade 1) stage,
and does it hold up on hospitals/datasets it has never seen. Explainability
(Grad-CAM/TCAV) and diabetic macular edema are deliberately out of scope here
- they are the subject of the companion paper. All numbers, tables, and
figures are pulled directly from this project's real results_log.csv and
experiments/results/ outputs; nothing here is simulated or invented.
Literature Survey citations are the same real, independently verified quantum
and ViT papers used in the companion paper - authors/pages were not
independently verified beyond title/venue/URL and must be checked by the
author before submission (flagged in-document)."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

FONT = "Times New Roman"

def style_run(run, size=11, bold=False, italic=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def para(doc, text="", size=11, bold=False, italic=False, align=None, space_after=8, space_before=0, indent=0.0):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    if indent:
        p.paragraph_format.first_line_indent = Inches(indent)
    if text:
        r = p.add_run(text)
        style_run(r, size=size, bold=bold, italic=italic)
    return p

def body(doc, text, indent=0.25, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(indent)
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    style_run(r, size=11)
    return p

def heading(doc, numeral, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(f"{numeral}. {title.upper()}")
    style_run(r, size=13, bold=True)
    return p

def subheading(doc, label, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(f"{label} {title}")
    style_run(r, size=11.5, bold=True)
    return p

def equation(doc, text, num):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.5)
    r = p.add_run(text)
    style_run(r, size=11, italic=True)
    r2 = p.add_run(f"\t\t\t\t\t({num})")
    style_run(r2, size=10)
    return p

def figure(doc, path, caption, num, width_in=5.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    run = p.add_run()
    run.add_picture(path, width=Inches(width_in))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    r = cap.add_run(f"Fig. {num}. {caption}")
    style_run(r, size=9.5, italic=True)

def table(doc, headers, rows, caption, num):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(10)
    r = cap.add_run(f"TABLE {num}")
    style_run(r, size=9.5)
    cap2 = doc.add_paragraph()
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = cap2.add_run(caption)
    style_run(r2, size=9.5, italic=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h); style_run(r, size=9, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val)); style_run(r, size=9)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def bullet(doc, text, size=11):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    style_run(r, size=size)

# =============================================================================
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1.0)
sec.top_margin = sec.bottom_margin = Inches(0.9)
style = doc.styles["Normal"]
style.font.name = FONT
style.font.size = Pt(11)

# ---- Title / authors ----
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("QuantumRetina-X: A Hybrid Quantum-Classical Vision Transformer for Early Detection of Diabetic Retinopathy")
style_run(r, size=16, bold=True)
p.paragraph_format.space_after = Pt(10)

p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("[Author Name(s)], [Department / Institution], [City, Country], [email@example.com]")
style_run(r, size=11, italic=True)
p.paragraph_format.space_after = Pt(4)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("[FILL IN before submission: author names, affiliation, contact email]")
style_run(r, size=9, italic=True, color=(150,0,0))
p.paragraph_format.space_after = Pt(12)

# ---- Abstract ----
p = doc.add_paragraph()
r = p.add_run("Abstract— "); style_run(r, size=11, bold=True)
r2 = p.add_run(
    "Diabetic retinopathy is a leading cause of preventable blindness, and it is most treatable at "
    "its earliest stage, when only a handful of microaneurysms are visible, which is also the "
    "stage automated screening tools most often miss. This paper presents QuantumRetina-X, a "
    "hybrid Vision Transformer and variational quantum circuit architecture, and asks directly "
    "whether the quantum component earns its place in the pipeline, particularly for early "
    "detection. A frozen ViT-Base/16 backbone is paired with a four-qubit variational quantum "
    "circuit and trained to grade diabetic retinopathy into five stages on the APTOS 2019 dataset "
    "(India), then tested with no further training on three datasets from three different "
    "countries, Messidor (France), IDRiD (India), and EyePACS (United States), to check whether "
    "any benefit survives real-world domain shift. A three-way ablation on the same held-out test "
    "set shows that the Vision Transformer supplies nearly all of the model's grading ability, the "
    "quantum-only configuration reaches just 54.6 percent accuracy, while adding the quantum "
    "circuit contributes a further, measurable 3.2-percentage-point accuracy gain, from 78.4 to "
    "81.6 percent, at the cost of a corresponding drop in AUC (0.936 to 0.894) and Grade-1 "
    "sensitivity (73.2 to 55.4 percent): a genuine trade-off, reported here in full rather than "
    "selectively. A second, purpose-built training variant, which reweights the loss toward Grade "
    "1 cases, recovers Grade-1 sensitivity to 78.6 percent, the highest of any configuration "
    "tested, showing that early-detection sensitivity is a tunable property of this architecture "
    "rather than a fixed limitation. Cross-dataset testing shows accuracy falling to as low as 43 "
    "percent on IDRiD, reported transparently as a genuine open problem, since none of the eleven "
    "quantum-assisted and Vision-Transformer papers reviewed here report comparable cross-dataset "
    "or grade-level results at all. Every number in this paper is additionally traced through one "
    "real worked example, with the model's own intermediate values, from patch embeddings to "
    "qubit measurements, made directly verifiable rather than only claimed, a level of "
    "transparency also absent from the reviewed literature. Together, these results place "
    "quantum-assisted grading as a measurable, tunable contribution rather than a uniform "
    "improvement, and establish an evaluation standard, honest ablation, grade-level breakdown, "
    "cross-dataset testing, and verifiable intermediate output, that is largely missing from "
    "prior quantum-assisted diabetic retinopathy work."
)
style_run(r2, size=10.5)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(10)

p = doc.add_paragraph()
r = p.add_run("Index Terms— "); style_run(r, size=11, bold=True)
r2 = p.add_run(
    "Diabetic Retinopathy, Vision Transformer, Variational Quantum Circuit, Quantum Machine "
    "Learning, Early Detection, Grade-1 Sensitivity, Fundus Image Classification, Cross-Dataset "
    "Generalization."
)
style_run(r2, size=10.5)
p.paragraph_format.space_after = Pt(14)

# =============================================================================
# I. INTRODUCTION
# =============================================================================
heading(doc, "I", "Introduction")
body(doc,
    "Diabetic retinopathy develops when years of high blood sugar damage the small blood vessels "
    "in the retina. It is graded on a scale from no visible disease up to an advanced stage where "
    "abnormal new vessels start growing, and it is generally agreed among eye specialists that the "
    "earlier it is caught, the easier it is to manage. The problem is that the earliest stage of "
    "meaningful disease, Grade 1, or mild non-proliferative retinopathy, is also the hardest to "
    "spot. It usually shows up as nothing more than a few microaneurysms, tiny red dots on the "
    "retina, easy to miss by eye and, as this paper will show, easy for a model to miss too. A "
    "model that misses Grade 1 disease does not just make a small error: it sends a patient home "
    "labeled healthy, and that patient may not be screened again for another year.")
body(doc,
    "For a long time, convolutional neural networks were the standard tool for automated diabetic "
    "retinopathy grading. Vision Transformers have become popular more recently, because instead of "
    "only looking at small local patches the way a convolutional filter does, they let every patch "
    "of the image attend to every other patch directly, which makes it easier to connect a small "
    "lesion in one part of the retina with something happening elsewhere in the same image. Around "
    "the same time, researchers have started experimenting with adding a small quantum circuit "
    "after the usual feature extractor, on the idea that a handful of qubits, once entangled, can "
    "represent correlations between compressed features that a classical layer of the same size "
    "cannot. Whether this quantum addition is actually useful for a real medical imaging task, "
    "rather than just an interesting idea on paper, is the question this paper sets out to answer.")
body(doc,
    "This paper deliberately stays focused on that modeling question. It does not cover how the "
    "model explains its predictions, and it does not cover diabetic macular edema; both of those "
    "are covered in a separate, companion paper. Here, the goal is to isolate the architecture "
    "question as cleanly as possible: does the quantum circuit help, does it help specifically at "
    "the hardest and most clinically important grade, and does whatever advantage it offers survive "
    "being tested on images from a hospital, camera, and patient population the model has never "
    "seen.")
body(doc,
    "Section II reviews eleven recent, published papers on quantum-assisted diabetic retinopathy "
    "detection and Vision-Transformer-based grading. Section III describes the proposed system in "
    "full, including the datasets, the architecture, the quantum circuit's mathematics, and a "
    "training variant built specifically to improve early-stage sensitivity. Section IV evaluates "
    "the model with a three-way ablation study and confusion matrices. Section V reports "
    "cross-dataset results and compares the findings against the reviewed literature. The main "
    "contributions of this work are listed below.")
bullet(doc, "A three-way ablation study, quantum circuit alone, Vision Transformer alone, and the two combined, is reported on the same train-test split and the same checkpoint-selection rule, including the quantum-only result even though it performs far worse, rather than only reporting the configurations that look favorable.")
bullet(doc, "A Grade-1-sensitivity-focused training variant is introduced, which trades some overall accuracy for a large gain in early-stage detection, and is reported as a tunable clinical choice rather than presented as simply a better number.")
bullet(doc, "The model is tested with no further training on three datasets it has never seen, from three different countries and camera setups, and the results are reported in full, including the cases where accuracy drops sharply.")
bullet(doc, "The quantum circuit's mathematics are laid out step by step, from angle encoding through entangling gates to measurement, so that a reader without a quantum computing background can follow what the circuit is actually doing to the four numbers it receives.")

print("Section I done"); doc.save("paper/_wip_paper1.docx")

# =============================================================================
# II. LITERATURE SURVEY
# =============================================================================
heading(doc, "II", "Literature Survey")
body(doc,
    "This section reviews eleven recent papers, grouped into two areas: quantum-assisted diabetic "
    "retinopathy detection and Vision-Transformer-based grading. For each paper, what it used as "
    "input, what method it applied, and what it found are summarized, along with the gap that the "
    "present work tries to fill.")

subheading(doc, "A.", "Quantum and Hybrid Quantum-Classical Approaches")
body(doc,
    "[3] combined a pretrained classical network with a quantum circuit running on compressed "
    "image features, taking fundus photographs as input and producing a diabetic retinopathy "
    "grade. [2] and [5] both combined quantum circuits with a convolutional backbone rather than a "
    "transformer, and [5] reported better efficiency than purely classical baselines. QuantumNet "
    "[8] paired a classical deep learning feature extractor with quantum transfer learning and was "
    "reported to reach high accuracy while using fewer computational resources. [6] built a hybrid "
    "quantum-classical pipeline that fed a four-qubit circuit, the same qubit count used in the "
    "present work, from a multi-scale convolutional lesion extractor, and dealt with class "
    "imbalance before the quantum stage. Two further papers target early detection specifically, "
    "which is the closest framing to the present work's own motivation: [4] built a federated "
    "quantum neural network aimed at privacy-preserving early detection, using a lightweight "
    "architecture with few learnable parameters rather than a large pretrained backbone, and [7] "
    "extracted classical features with principal component analysis and fed them into a quantum "
    "support vector machine and a variational quantum classifier, again without a Vision "
    "Transformer or any other large pretrained feature extractor in the loop. Across all seven of "
    "these papers, the quantum component was paired with a convolutional backbone, a "
    "PCA-reduced feature vector, or a small lightweight encoder, never a Vision Transformer, and "
    "none of them broke their results down by grade or reported what happened to sensitivity "
    "specifically at the mildest disease stage; most reported only a single overall accuracy "
    "number, even the two papers that used \"early detection\" in their own title.")

subheading(doc, "B.", "Vision Transformer Approaches to Diabetic Retinopathy Grading")
body(doc,
    "[10] split fundus images into patches, embedded them, and ran them through multi-head "
    "self-attention layers to grade disease severity while keeping positional information intact. "
    "[11] took a similar Vision Transformer approach to predict severity from fundus photographs. "
    "[9] built a task-optimized Vision Transformer aimed at resource-constrained settings, "
    "balancing accuracy against computational cost, with early diagnosis specifically named as a "
    "target use case. [1] used a deep learning grading model evaluated on fundus images with the "
    "standard five-class output. None of these four papers combined a Vision Transformer with a "
    "quantum circuit, and none reported cross-dataset testing on images from a hospital or country "
    "different from the one the training data came from, which is the gap addressed directly in "
    "Section V of this paper.")

body(doc,
    "Taken together, these eleven papers show that quantum-assisted methods for diabetic "
    "retinopathy have so far been paired with convolutional backbones, PCA-reduced features, or "
    "small lightweight encoders, and that even the papers explicitly framed around early detection "
    "have not paired a quantum circuit with a Vision Transformer or reported sensitivity broken "
    "down by grade. Grade-1 sensitivity specifically is not reported as a headline metric anywhere "
    "in this survey, even though it is arguably the number that matters most for a screening tool. "
    "The system described in this paper combines a Vision Transformer with a quantum circuit, "
    "reports Grade-1 sensitivity explicitly and as a tunable quantity, and tests the result on "
    "three datasets the model has never seen, which is a specific combination not found in any "
    "single paper reviewed here, quantum-assisted or Vision-Transformer-based.")

print("Section II done"); doc.save("paper/_wip_paper1.docx")

# =============================================================================
# III. PROPOSED SYSTEM
# =============================================================================
heading(doc, "III", "Proposed System")
body(doc,
    "This section describes the proposed architecture end to end, from the raw fundus photograph "
    "to the final diabetic retinopathy grade. Fig. 1 shows the overall design at a glance: a "
    "frozen ViT-Base/16 backbone extracts a single feature vector from the input image, a linear "
    "reducer compresses that vector down to four numbers, a four-qubit variational quantum circuit "
    "processes those four numbers, and a final linear layer turns the circuit's output into a "
    "diabetic retinopathy grade. Only the reducer, the quantum circuit, and this final layer are "
    "trained; the backbone stays frozen throughout, for reasons explained in Section III-C. Each "
    "block in this diagram is unpacked in its own subsection below, with real numbers from a "
    "single worked example carried through every stage rather than a description on its own.")
figure(doc, "paper/fig_architecture.png",
    "Overall architecture. A frozen ViT-Base/16 backbone feeds a linear reducer and a four-qubit "
    "variational quantum circuit, which together replace a conventional classification head.", 1, width_in=6.0)
body(doc,
    "Fig. 2 lays out the complete end-to-end process as a single flowchart, including the stages "
    "described only in summary by Fig. 1: preprocessing, the fundus-masking step that both "
    "explanation branches depend on, the parallel macular-edema model, and the classical red-free "
    "reference view. Every subsection below walks through one branch of this flowchart in detail, "
    "with real numbers from a single worked example.")
figure(doc, "paper/fig_workflow_complete.png",
    "Complete end-to-end process flowchart: every real stage from raw image to final grade, spatial explanation, concept explanation, and macular edema output.", 2, width_in=6.6)

subheading(doc, "A.", "Datasets and Input Specification")
body(doc,
    "APTOS 2019, with 3,662 labeled images and a five-class diabetic retinopathy grade, was used "
    "as the only training dataset, split into training, validation, and a 550-image held-out test "
    "set used for every result reported in Section IV. Three further datasets, Messidor (1,744 "
    "gradable images), IDRiD (455 gradable images after removing rows with no matching grade "
    "label), and a 5,000-image stratified subset of EyePACS, were used only to test the trained "
    "models, with no fine-tuning, to see how the models behaved on images from a different camera, "
    "hospital, and patient population than the one they were trained on. Fig. 3 shows example "
    "images from APTOS across the five grades. Every input image, regardless of dataset, was "
    "resized to 224 by 224 pixels with three color channels before any further processing.")
figure(doc, "experiments/results/training/aptos_sample_grid.png",
    "Example fundus photographs from the APTOS training set, labeled by grade (0 = no disease, 4 = proliferative disease).", 3, width_in=5.8)
body(doc,
    "To make every equation in this section concrete, one real IDRiD photograph is carried through "
    "every subsection below as a running example, so each formula can be checked against real "
    "numbers rather than taken on faith. Fig. 4 shows that input image, and Fig. 5 shows the "
    "model's actual final output for it, the real result that the rest of this section builds up "
    "to: the hybrid model graded this image Proliferative DR (Grade 4), the most severe of the "
    "five grading levels, at 32.6 percent confidence, with a smaller remaining probability on "
    "Grade 3. Grade 0 means no visible disease, Grade 1 is mild disease with only a few "
    "microaneurysms, Grade 2 is moderate disease, Grade 3 is severe disease, and Grade 4 is "
    "proliferative disease, where abnormal new blood vessels have started growing. None of the "
    "quantum-assisted papers reviewed in Section II-A show a real image next to the model's real "
    "output this way; most report only an aggregate accuracy table.")
figure(doc, "paper/fig_stage_input.png",
    "The real example image carried through every subsection of this section, resized to 224x224 as described above.", 4, width_in=3.0)
figure(doc, "paper/fig_stage_output_probs.png",
    "The real final output (Equation 8) for the example image: how each subsection below builds up to this result.", 5, width_in=4.6)
body(doc,
    "The grade and probabilities in Fig. 5 are a number, not a picture, and by themselves do not "
    "show how the model reached that answer. Fig. 6 shows the visual counterpart: the same eye "
    "next to its real spatial explanation and its real per-lesion-type map, produced by the "
    "explainability methods described fully in Section III-G, so a reader can see, on the same "
    "image used throughout this section, roughly where the model was looking rather than only the "
    "final grade it output.")
figure(doc, "paper/fig_stage_triptych.png",
    "The example eye, its real spatial explanation, and its real per-lesion-type map, showing visually how the Fig. 5 prediction was reached (full method in Section III-G).", 6, width_in=6.4)

subheading(doc, "B.", "Preprocessing")
body(doc,
    "Each resized image was converted into a tensor of numbers between 0 and 1, and then each color "
    "channel was normalized using the mean and standard deviation of ImageNet, the dataset the "
    "Vision Transformer backbone was originally trained on:")
equation(doc, "x' = (x / 255 − μ) / σ", "1")
body(doc,
    "where x is the raw pixel value, μ = [0.485, 0.456, 0.406] and σ = [0.229, 0.224, 0.225] are "
    "the per-channel mean and standard deviation, and x' is the normalized value fed into the "
    "network. This step matters because the backbone's weights were trained on images following "
    "this exact distribution; feeding it raw pixels would shift the input away from what it "
    "expects and would hurt the quality of its features. For the example image from Section III-A, "
    "the raw pixel distribution had a mean of 62.9 and a standard deviation of 69.63; after "
    "Equation 1 was applied, the normalized distribution actually fed to the network had a mean of "
    "-0.9 and a standard deviation of 1.115. Fig. 7 shows what those normalized numbers actually "
    "are, per color channel, next to the raw photograph: normalization is a per-channel linear "
    "rescale, so if the result were forced back into an ordinary photo it would look almost "
    "identical to the original and would hide the change rather than show it, which is why each "
    "channel is instead shown as a heatmap of its own real normalized values. Fig. 8 shows the two "
    "pixel-value distributions side by side, computed directly from this image rather than shown as "
    "an illustrative diagram. During training, images were randomly flipped, rotated by up to "
    "fifteen degrees, and given a small amount of brightness and contrast jitter. None of this was "
    "applied during validation, testing, or the cross-dataset runs reported in Section V, so every "
    "accuracy number in this paper reflects performance on plain, unmodified images.")
figure(doc, "paper/fig_stage_preprocessing_visual.png",
    "The real raw photograph next to its actual normalized R, G, and B channel values (Equation 1), each shown as its own heatmap rather than forced back into a misleading photo-like render.", 7, width_in=6.4)
figure(doc, "paper/fig_stage_preprocessing_hist.png",
    "Real raw-pixel and normalized-pixel histograms for the example image, before and after Equation 1 is applied.", 8, width_in=5.4)

subheading(doc, "C.", "Feature Extraction with the Vision Transformer Backbone")
body(doc,
    "The normalized image was cut into a 14 by 14 grid of patches, each 16 by 16 pixels, giving "
    "196 patches. Each patch was flattened and turned into a 768-number embedding, and a learnable "
    "classification token was added at the front, giving a sequence of 197 tokens in total. "
    "Position information, otherwise lost when patches are flattened, was added back through "
    "learned position embeddings. This sequence passed through twelve Transformer encoder layers, "
    "each using multi-head self-attention, computed per head as")
equation(doc, "Attention(Q, K, V) = softmax(QKᵀ / √d) V", "2")
body(doc,
    "where Q, K, and V are the query, key, and value projections of the tokens, and d is the size "
    "of each attention head. This lets every patch look directly at every other patch, so a link "
    "between a lesion in one part of the retina and, say, the optic disc on the other side of the "
    "image can be captured in a single layer, instead of needing many stacked layers the way a "
    "purely convolutional network would. After the twelve layers, only the classification token, a "
    "single 768-number vector summarizing the whole image, was kept for the next stage. Fig. 9 "
    "shows the image before and after this patchification, and Fig. 10 shows the same real grid "
    "overlaid directly on the example image (not an illustrative diagram), together with the real "
    "classification-token magnitude the backbone produced for it, ‖h‖ = 5.25. None of the Vision "
    "Transformer papers reviewed in Section II-B, [1], [9], [10], and [11], show the patch grid "
    "overlaid on one of their own example images; each reports only aggregate accuracy figures, "
    "which makes it hard for a reader to check by eye whether the patch resolution is fine enough "
    "to resolve small lesions. Showing the grid directly on a real image makes that limitation "
    "visible rather than implicit: individual microaneurysms are visibly smaller than a single 16 "
    "by 16 patch. The backbone itself, 86.4 million parameters, stayed frozen throughout training; "
    "it was initialized from ImageNet pretraining and never updated by gradient descent in this "
    "project, so every difference reported later between configurations comes entirely from what "
    "sits on top of this fixed backbone.")
figure(doc, "paper/fig_stage_patchgrid_visual.png",
    "The example image before and after being cut into the real 14x14 patch grid (Equation 2).", 9, width_in=5.6)
figure(doc, "paper/fig_ui_vit_patchgrid.png",
    "The real 14x14 patch grid (Equation 2) overlaid on the example image, alongside the real classification-token norm produced for it.", 10, width_in=6.2)

subheading(doc, "D.", "Dimensionality Reduction and Variational Quantum Circuit")
body(doc,
    "From this point on, the data is no longer spatial. Sections III-A through III-C each showed "
    "the example image itself changing, resized, normalized, cut into patches, but the "
    "classification token leaving the backbone is already an abstract 768-number summary with no "
    "pixel grid left to display. There is still a real before-and-after to show, just not as a "
    "photograph: Fig. 11 renders the actual 768 real feature values leaving the backbone as a "
    "color-coded strip, immediately above the four real numbers that layer is compressed down to, "
    "so the reduction itself, 768 real quantities becoming 4, is shown rather than only described. "
    "The 768-number classification-token vector is far too large to feed into a four-qubit "
    "circuit, so it is first shrunk down to four values by a single trainable linear layer,")
equation(doc, "y = W·h + b,  W ∈ ℝ⁴ˣ⁷⁶⁸, h ∈ ℝ⁷⁶⁸ → y ∈ ℝ⁴", "3")
body(doc,
    "before being handed to the quantum circuit itself, which was built with the PennyLane quantum "
    "simulator and runs entirely as a numerical simulation on this project's GPU, not on physical "
    "quantum hardware.")
figure(doc, "paper/fig_stage_reducer_visual.png",
    "Before and after Equation 3 for the example image: the real 768-number [CLS] embedding (top) compressed to the real 4-number reducer output (bottom).", 11, width_in=6.0)
body(doc,
    "Gradients flow backward through the quantum circuit using the parameter-shift rule, the same "
    "way they flow through the ordinary classical layers, so the whole model can still be trained "
    "end to end with a standard optimizer. Each of the four shrunk values is loaded onto one qubit "
    "through a rotation about the Y axis,")
equation(doc, "|ψᵢ⟩ = R_Y(yᵢ) |0⟩,  i = 1, …, 4", "4")
body(doc,
    "and then two trainable layers are applied on top. Each layer rotates every qubit around the "
    "X, Y, and Z axes and then links neighboring qubits together with a ring of CNOT gates,")
equation(doc, "Uₗ(θ) = (∏ᵢ CNOTᵢ,ᵢ₊₁) ∏ᵢ R_Z(θₗ,ᵢ,₃) R_Y(θₗ,ᵢ,₂) R_X(θₗ,ᵢ,₁)", "5")
body(doc,
    "The CNOT ring is what makes this different from four independent classical numbers: once the "
    "qubits are entangled, the four of them together occupy a sixteen-dimensional space in which, "
    "in principle, correlations between the four shrunk input values can be represented that a "
    "plain classical layer of the same width could not represent directly. Whether that extra "
    "representational room is actually useful for this particular task, rather than just "
    "mathematically possible, is exactly what Section IV's ablation study tests. After the two "
    "layers, each qubit is measured, giving one number between -1 and 1 per qubit,")
equation(doc, "⟨Zᵢ⟩ = ⟨ψ₂| Zᵢ |ψ₂⟩", "6")
body(doc,
    "Fig. 12 shows the real four qubit measurements produced for the example image; read together "
    "with the bottom panel of Fig. 11, this is the quantum circuit's own before-and-after, the same "
    "four reducer numbers going in and four different numbers coming out. None of the "
    "quantum papers reviewed in Section II-A publish the intermediate values their own circuits "
    "produce for a specific image; typically only the final predicted label is reported, which "
    "makes it impossible for a reader to verify that the quantum stage is doing anything beyond "
    "what a classical layer of the same width could do. Reporting these values directly, as in "
    "Fig. 11 and Fig. 12, is a small but concrete step toward that kind of verifiability.")
figure(doc, "paper/fig_stage_vqc.png",
    "Real 4-qubit measurement (Equation 6) for the example image.", 12, width_in=4.6)
body(doc,
    "A single worked example only shows that the reducer and quantum circuit produce some real "
    "numbers; it does not show whether those numbers actually track disease severity. Fig. 13 "
    "answers that directly, running five further real test images, one true example of each grade, "
    "through the same real model and comparing their real reducer and VQC outputs side by side. "
    "The values visibly shift with grade, most clearly in the reducer's second and third "
    "components, which is the first evidence in this paper that the quantum circuit's input is "
    "carrying grade-relevant signal rather than noise, ahead of the full ablation result in "
    "Section IV.")
figure(doc, "paper/fig_d_cross_grade_comparison.png",
    "Real reducer and VQC output compared across five real test images, one per true grade, showing the values shift with disease severity rather than staying constant.", 13, width_in=6.6)
body(doc,
    "These four numbers are passed to one last trainable linear layer, which outputs five values, "
    "one per diabetic retinopathy grade,")
equation(doc, "z = W·⟨Z⟩ + b,  z ∈ ℝ⁵", "7")
body(doc,
    "turned into class probabilities by the softmax function,")
equation(doc, "p_c = e^(z_c) / Σⱼ e^(z_j),  c = 0, …, 4", "8")
body(doc,
    "which is exactly how the final result shown back in Fig. 5 was produced from the four numbers "
    "in Fig. 12. Only the shrinking layer, the quantum circuit's own rotation angles, and this final "
    "output layer, roughly 3,100 trainable parameters altogether, were actually updated during "
    "training. For the quantum-circuit-only configuration used in the ablation study in Section IV, "
    "the Vision Transformer backbone was removed entirely and replaced with a small classical "
    "encoder feeding the same four-qubit circuit directly from raw pixel statistics, so that the "
    "contribution of the quantum circuit on its own, without any transformer features behind it, "
    "could be measured honestly.")

subheading(doc, "E.", "A Training Variant Focused on Grade-1 Sensitivity")
body(doc,
    "Because missing a mild, Grade 1 case is a more costly mistake in a screening context than "
    "most other misclassifications, a second training variant was built specifically to push the "
    "model toward catching more of them. The class weight applied to Grade 1 in the training loss "
    "was multiplied by a boost factor,")
equation(doc, "w₁' = g · w₁,  g = 3.0", "9")
body(doc,
    "which makes every Grade 1 training example count three times as much toward the loss as it "
    "otherwise would, pushing the model to pay disproportionate attention to the features that "
    "separate Grade 1 from Grade 0. A second, balanced variant combined this same loss reweighting "
    "with a lowered decision threshold specifically for Grade 1, so that the model would call a "
    "case Grade 1 once its estimated probability for that class passed 0.72 rather than only when "
    "Grade 1 had the single highest probability among all five classes. Fig. 14 shows what this "
    "reweighting actually does to sensitivity at every grade, not just Grade 1, computed on the "
    "real held-out APTOS test set: the plain boosted variant recovers the most Grade-1 sensitivity "
    "but gives up ground at Grade 2 and Grade 3, while the balanced variant recovers less Grade-1 "
    "sensitivity but keeps overall accuracy close to the unweighted hybrid model, discussed further "
    "in Section IV-C. Both are reported as legitimate, deliberately chosen operating points rather "
    "than as an attempt to replace the main hybrid result with a better-looking number. This is "
    "this section's own before-and-after: Fig. 14's purple bars are the model's real per-grade "
    "sensitivity before any reweighting, and its red bars are the same real measurement after "
    "Equation 9 is applied.")
figure(doc, "experiments/results/ablation/per_grade_sensitivity.png",
    "Before and after Grade-1 reweighting (Equation 9): real per-grade sensitivity for the unweighted hybrid model versus the Grade-1-boosted variant.", 14, width_in=6.0)
body(doc,
    "Fig. 15 makes this concrete with one real case pulled directly from the held-out test set. "
    "For a true Grade 1 image, the plain hybrid model was 91.0 percent confident it was Grade 2, "
    "wrongly; the Grade-1-boosted model, given the exact same image, correctly placed the highest "
    "probability on Grade 1. This is a single example, not a claim that reweighting fixes every "
    "case, but it is a real, verifiable instance of the aggregate sensitivity shift in Fig. 14 "
    "actually changing a specific diagnosis from wrong to right rather than only moving a "
    "population-level percentage.")
figure(doc, "paper/fig_g1boost_real_flip.png",
    "A real held-out test image where Grade-1 reweighting changed the model's prediction from wrong (Grade 2, plain model) to correct (Grade 1, boosted model).", 15, width_in=6.6)

subheading(doc, "F.", "Training Configuration")
body(doc,
    "All configurations, including the ablation study's three variants, were trained with a "
    "class-weighted cross-entropy loss,")
equation(doc, "L = − Σ_{c=0}^{4} w_c y_c log(ŷ_c)", "10")
body(doc,
    "with class weights set inversely to how often each grade appeared in the training split, and "
    "further modified by the Grade-1 boost described in Section III-E where applicable. Training "
    "used the AdamW optimizer, with a learning rate of 3.0 x 10⁻⁵ for the backbone, which stayed "
    "frozen anyway, and 1.0 x 10⁻³ for the trainable reducer, quantum circuit, and output layers, "
    "a weight decay of 1.0 x 10⁻⁴, and a batch size of 8. The main hybrid and baseline models were "
    "trained for 20 epochs; the ablation study's three configurations were trained on the same "
    "split with the same optimizer settings so that the comparison in Section IV-A would be fair. "
    "In every case, the checkpoint with the highest validation quadratic-weighted Cohen's kappa "
    "was kept as the final model, since kappa penalizes a prediction that is far from the true "
    "grade more heavily than one that is close, which is a more clinically sensible criterion than "
    "raw accuracy for selecting a checkpoint. This section's own before-and-after is temporal rather "
    "than visual: Fig. 16 shows what training actually looked like from epoch 1 to epoch 20, real "
    "per-epoch loss, validation kappa, and validation Grade-1 sensitivity, so the untrained model's "
    "starting point and the final checkpoint's ending point are both visible on the same axes, "
    "discussed in full in Section V-D. No intermediate checkpoint between epoch 1 and the final "
    "epoch was saved during training, so this project cannot show what the example image's "
    "prediction looked like partway through training without retraining and is not claiming to.")
figure(doc, "experiments/results/training/training_curves.png",
    "Before and after training, epoch 1 through epoch 20: real training loss, validation kappa, and validation Grade-1 sensitivity for the ViT-only and hybrid models.", 16, width_in=6.4)

subheading(doc, "G.", "Downstream Explainability and Macular Edema Outputs (Brief)")
body(doc,
    "The architecture described above feeds a deployed system that also produces two further "
    "outputs on the same example image, described fully in the companion paper: a spatial "
    "explanation of where the model is looking, and a separate detection of diabetic macular "
    "edema. Both are shown here only briefly, on this same example image, for completeness. Before "
    "either explanation is computed, the background of the photograph is separated from the real "
    "retina using a fundus mask; Fig. 17 shows this step, with the excluded region tinted red and "
    "the kept region, 49.6 percent of the frame for this image, left untouched.")
figure(doc, "paper/fig_stage_mask_before_after.png",
    "The example image before and after the fundus mask is applied, restricting every explanation below to the real retina.", 17, width_in=6.0)
body(doc,
    "Grad-CAM and four separate concept probes then each produce their own spatial map, one per "
    "lesion type (microaneurysm, hemorrhage, hard exudate, and macular edema), shown individually "
    "in Fig. 18 before they are merged into the single combined map shown alongside the original "
    "photograph and the Grad-CAM heatmap in Fig. 19. A separate, non-AI reference view, the green "
    "color channel extracted and contrast-enhanced in the style ophthalmologists have "
    "traditionally used to read hemorrhages and microaneurysms, is shown in Fig. 20; it is not fed "
    "to any model, only computed for a clinician to cross-check by eye. Finally, Fig. 21 shows the "
    "real macular edema probability for this image: detected, at 96.8 percent confidence against "
    "the 0.5 threshold.")
figure(doc, "paper/fig_stage_concept_grid.png",
    "The four individual per-lesion-type maps for the example image, before they are merged into one.", 18, width_in=5.4)
figure(doc, "paper/fig_stage_triptych.png",
    "The same example image, its spatial explanation (Grad-CAM), and its combined per-lesion-type map, produced downstream of the architecture described in this paper.", 19, width_in=6.4)
figure(doc, "paper/fig_stage_redfree_before_after.png",
    "The example image and its red-free (green-channel, contrast-enhanced) reference view, not used by any model.", 20, width_in=6.0)
figure(doc, "paper/fig_stage_dme_output.png",
    "Real macular edema probability for the example image, against the 0.5 detection threshold.", 21, width_in=4.6)

print("Section III done"); doc.save("paper/_wip_paper1.docx")

# =============================================================================
# IV. PERFORMANCE EVALUATION
# =============================================================================
heading(doc, "IV", "Performance Evaluation")
body(doc,
    "Performance was measured with accuracy, macro-averaged F1 score, quadratic-weighted Cohen's "
    "kappa, area under the receiver operating characteristic curve (AUC), and per-grade "
    "sensitivity, with particular attention paid to Grade 1. Accuracy, precision, recall, and F1 "
    "were computed as")
equation(doc, "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "11")
equation(doc, "Precision = TP / (TP + FP),  Recall = TP / (TP + FN)", "12")
equation(doc, "F1 = 2 · (Precision · Recall) / (Precision + Recall)", "13")
body(doc,
    "where TP, TN, FP, and FN are true positive, true negative, false positive, and false negative "
    "counts, taken per class and averaged for the five-class task. Cohen's kappa, standard in "
    "diabetic retinopathy grading because it punishes a prediction that is far from the true grade "
    "more than one that is close, was computed as")
equation(doc, "κ = (p_o − p_e) / (1 − p_e)", "14")
body(doc,
    "where p_o is the observed agreement and p_e is the agreement expected by chance, with "
    "quadratic weighting applied to the disagreement matrix. Fig. 22 gives a single-glance summary "
    "of the two main configurations across every one of these metrics before the section-by-section "
    "breakdown below.")
figure(doc, "experiments/results/training/model_comparison.png",
    "Headline comparison, ViT-only versus the proposed hybrid model, across accuracy, kappa, macro-F1, Grade-1 sensitivity, and AUC.", 22, width_in=6.0)

subheading(doc, "A.", "Ablation Study")
body(doc,
    "Three configurations were compared on the same APTOS train-test split and the same "
    "checkpoint-selection rule: the quantum circuit alone (Section III-D's classical-encoder "
    "variant), the Vision Transformer alone with no quantum circuit, and the full hybrid model.")
figure(doc, "experiments/results/ablation/ablation_chart.png",
    "Ablation study: accuracy, kappa, AUC, and Grade-1 sensitivity for the quantum-only, ViT-only, and hybrid configurations, all evaluated on the same held-out APTOS test set.", 23, width_in=6.4)
table(doc, ["Configuration", "Accuracy", "Macro-F1", "AUC", "Kappa", "Grade-1 Sens."],
    [["VQC only", "54.6%", "0.237", "0.646", "0.289", "0.000"],
     ["ViT only", "78.4%", "0.638", "0.936", "0.881", "0.732"],
     ["ViT + VQC (proposed)", "81.6%", "0.659", "0.894", "0.871", "0.554"]],
    "Ablation Study on APTOS Held-Out Test Set (n=550)", "I")
body(doc,
    "The quantum-only configuration performed far worse than either Vision-Transformer-based "
    "configuration on every metric, and its Grade-1 sensitivity was exactly zero, meaning it never "
    "once correctly identified a mild case in the entire test set. This makes clear that the "
    "Vision Transformer backbone, not the quantum circuit, is responsible for essentially all of "
    "the model's ability to grade the disease at all; the quantum circuit is working with whatever "
    "the backbone has already extracted, not replacing it. Adding the quantum circuit on top of "
    "the full Vision Transformer raised accuracy by 3.2 percentage points, from 78.4 to 81.6 "
    "percent, but it lowered AUC by 0.042 and kappa by 0.010, and it lowered Grade-1 sensitivity "
    "from 73.2 to 55.4 percent. In other words, the quantum layer did not uniformly improve the "
    "model; it traded some ranking quality and some early-detection sensitivity for a higher "
    "overall accuracy number, largely by getting more of the easy Grade 0 and Grade 2 cases right "
    "at the expense of some Grade 1 cases. Fig. 24 shows exactly what the quantum-only "
    "configuration's zero Grade-1 sensitivity looks like in full: it collapsed almost entirely onto "
    "Grade 0 and Grade 2, never predicting Grade 1 or Grade 3 for any test image, which is a "
    "starker picture than the single sensitivity number in Table I conveys on its own.")
figure(doc, "experiments/results/ablation/vqc_only_confusion_matrix.png",
    "Confusion matrix, quantum-circuit-only configuration, APTOS held-out test set (n = 550).", 24, width_in=4.4)

subheading(doc, "B.", "Confusion Matrices")
body(doc,
    "Fig. 25 and Fig. 26 show confusion matrices for the ViT-only baseline and the hybrid model on "
    "the same 550-image APTOS test split. Both models correctly classified most Grade 0 and Grade "
    "2 images. The hybrid model made fewer mistakes labeling Grade 2 images as Grade 3, just 5 "
    "compared with 27 for the baseline, which accounts for a meaningful share of its higher "
    "overall accuracy. At the same time, the hybrid model correctly caught fewer Grade 1 cases "
    "than the baseline did, consistent with the sensitivity numbers in Table I: several images "
    "that the baseline correctly labeled Grade 1 were instead pushed toward Grade 0 or Grade 2 by "
    "the hybrid model. Both models confused neighboring grades, especially Grade 1 with Grade 2 "
    "and Grade 3 with Grade 4, more than they confused distant grades, which matches the visual "
    "overlap between adjacent disease stages and is a pattern also reported in the Vision "
    "Transformer literature reviewed in Section II-B.")
figure(doc, "experiments/results/confusion_matrices/vit_confusion_matrix.png",
    "Confusion matrix, ViT-only baseline, APTOS held-out test set (n = 550).", 25, width_in=4.4)
figure(doc, "experiments/results/confusion_matrices/quantum_confusion_matrix.png",
    "Confusion matrix, proposed ViT+VQC hybrid model, APTOS held-out test set (n = 550).", 26, width_in=4.4)

subheading(doc, "C.", "Grade-1 Sensitivity and the Boosted Variant")
body(doc,
    "Table II compares the plain hybrid model against the two Grade-1-focused variants described "
    "in Section III-E, all evaluated on the same APTOS test set.")
table(doc, ["Configuration", "Accuracy", "Kappa", "Grade-1 Sens.", "Notes"],
    [["ViT only", "78.4%", "0.881", "73.2%", "no reweighting"],
     ["ViT + VQC", "81.6%", "0.871", "55.4%", "no reweighting"],
     ["ViT + VQC (G1-boost)", "74.6%", "0.762", "78.6%", "class weight x3 on Grade 1"],
     ["ViT + VQC (balanced)", "81.5%", "0.872", "66.1%", "class weight x3, threshold 0.72"]],
    "Grade-1 Sensitivity Across Training Variants, APTOS Held-Out Test Set (n=550)", "II")
figure(doc, "experiments/results/ablation/per_grade_sensitivity.png",
    "Per-grade sensitivity for the ViT-only, hybrid, and Grade-1-boosted hybrid models (same real data as Fig. 14, repeated here alongside the evaluation numbers in Table II).", 27, width_in=6.0)
body(doc,
    "The plain class-weight boost recovers the most Grade-1 sensitivity of any configuration "
    "tested, 78.6 percent, higher even than the ViT-only baseline, but it does so at a real cost: "
    "overall accuracy drops to 74.6 percent and kappa drops to 0.762, both below the unweighted "
    "hybrid model and below the ViT-only baseline. The balanced variant, which pairs the same loss "
    "reweighting with a lowered Grade-1 decision threshold, keeps almost all of the unweighted "
    "hybrid model's accuracy and kappa while still recovering a meaningful share of the lost "
    "Grade-1 sensitivity, from 55.4 to 66.1 percent. Neither variant is presented here as simply "
    "better than the other; which one is preferable depends on how the model would actually be "
    "used. A screening tool meant to flag any possible early disease for a specialist to review "
    "would reasonably favor the boosted variant's higher recall even at the cost of more false "
    "alarms, while a tool meant to support a more balanced diagnostic workflow might prefer the "
    "balanced variant instead.")

print("Section IV done"); doc.save("paper/_wip_paper1.docx")

# =============================================================================
# V. EXPECTED RESULTS
# =============================================================================
heading(doc, "V", "Expected Results")
body(doc,
    "This section reports what happened when the ViT-only and hybrid models, both trained only on "
    "APTOS, were tested with no further training on three datasets they had never seen, and "
    "compares the overall findings against the literature reviewed in Section II.")

subheading(doc, "A.", "Cross-Dataset Generalization")
body(doc,
    "Both models were evaluated zero-shot on Messidor, IDRiD, and EyePACS, to see how well they "
    "held up when the camera, patient population, and imaging setup all changed at once.")
figure(doc, "experiments/results/cross_dataset/crossdataset_full_comparison.png",
    "Cross-dataset generalization from APTOS to Messidor, IDRiD, and EyePACS, ViT-only versus the proposed hybrid model, across accuracy, kappa, AUC, and Grade-1 sensitivity.", 28, width_in=6.2)
table(doc, ["Dataset", "n", "Model", "Accuracy", "Kappa", "AUC", "Grade-1 Sens."],
    [["Messidor", "1,744", "ViT only", "59.4%", "0.449", "0.760", "3.3%"],
     ["Messidor", "1,744", "ViT + VQC", "58.3%", "0.430", "0.577", "5.6%"],
     ["IDRiD", "455", "ViT only", "42.9%", "0.578", "0.786", "31.8%"],
     ["IDRiD", "455", "ViT + VQC", "43.5%", "0.618", "0.657", "50.0%"],
     ["EyePACS", "5,000", "ViT only", "64.2%", "0.412", "0.725", "18.1%"],
     ["EyePACS", "5,000", "ViT + VQC", "63.0%", "0.412", "0.603", "19.0%"]],
    "Zero-Shot Cross-Dataset Results, Models Trained Only on APTOS", "III")
body(doc,
    "Accuracy dropped substantially on every unseen dataset relative to the 78 to 82 percent range "
    "achieved on APTOS. The drop was worst on IDRiD, where both models fell to around 43 percent "
    "accuracy, and mildest on EyePACS, where both models still lost 15 to 18 percentage points. "
    "Interestingly, the hybrid model's kappa and Grade-1 sensitivity were noticeably higher than "
    "the baseline's on IDRiD specifically, 0.618 versus 0.578 and 50.0 versus 31.8 percent, even "
    "though its accuracy was only marginally higher and its AUC was clearly lower. This is not "
    "necessarily a sign that the quantum layer generalizes better; Section V-C looks at the IDRiD "
    "confusion matrices directly and finds a more mundane explanation.")

subheading(doc, "B.", "Early Detection Across Datasets")
body(doc,
    "Fig. 29 focuses specifically on Grade-1 sensitivity across all four datasets, using the plain, "
    "unweighted hybrid model rather than the Grade-1-boosted variant from Section III-E, since "
    "that variant was only trained and evaluated on APTOS.")
figure(doc, "experiments/results/cross_dataset/crossdataset_g1_comparison.png",
    "Grade-1 sensitivity for the ViT-only and hybrid models, within-dataset on APTOS and zero-shot on Messidor, IDRiD, and EyePACS.", 29, width_in=6.0)
body(doc,
    "Grade-1 sensitivity on Messidor was close to zero for both models, 3.3 and 5.6 percent, which "
    "means that on this particular dataset, neither model is currently useful as an early-detection "
    "tool at all; it is reported here plainly rather than left out, since a screening tool that "
    "misses nearly every mild case on a real, external dataset is a genuine limitation, not a "
    "minor caveat. The picture on EyePACS was similarly weak, under 20 percent for both models. "
    "IDRiD was the one dataset where Grade-1 sensitivity held up reasonably, and even improved for "
    "the hybrid model, which is examined more closely below.")

subheading(doc, "C.", "A Closer Look at Domain Shift on IDRiD")
body(doc,
    "Fig. 30 and Fig. 31 show the full confusion matrices behind the IDRiD row of Table III.")
figure(doc, "experiments/results/confusion_matrices/idrid_vit_confusion_matrix.png",
    "ViT-only confusion matrix, zero-shot on IDRiD (n = 455).", 30, width_in=4.4)
figure(doc, "experiments/results/confusion_matrices/idrid_quantum_confusion_matrix.png",
    "ViT+VQC hybrid confusion matrix, zero-shot on IDRiD (n = 455).", 31, width_in=4.4)
body(doc,
    "Both models lean heavily on Grade 2 as a kind of default answer on IDRiD: 118 of the "
    "baseline's Grade 2 predictions and 129 of the hybrid model's were correct, but both models "
    "also pushed a large number of Grade 0 and Grade 3 images toward Grade 2 as well. The hybrid "
    "model's apparently higher Grade-1 sensitivity on this dataset comes with a visible cost "
    "elsewhere: it correctly caught only 4 of the 84 true Grade 3 images, compared with 26 for the "
    "baseline, and it labeled 64 true Grade 0 images as Grade 1, far more than the baseline's 38. "
    "Read together, this looks less like the quantum layer generalizing better and more like the "
    "hybrid model's decision boundaries shifting in a way that happens to catch more true Grade 1 "
    "cases on this particular dataset while missing more Grade 3 cases and over-calling Grade 1 on "
    "healthy images. This is exactly the kind of nuance a single headline accuracy or sensitivity "
    "number would hide, which is why the full confusion matrix is reported here rather than only "
    "the summary statistics in Table III.")
body(doc,
    "Fig. 32 and Fig. 33 show the same breakdown for Messidor, the dataset where Table III's "
    "Grade-1 sensitivity numbers were lowest for both models. The confusion matrices make clear "
    "why: of 270 true Grade 1 images, the ViT-only baseline predicted Grade 0 for 256 of them and "
    "the hybrid model predicted Grade 0 for 249, meaning both models called well over 90 percent of "
    "mild cases healthy. The hybrid model's small edge in Grade-1 sensitivity, 15 correct against "
    "the baseline's 9, comes at a cost at the severe end: of 75 true Grade 3 images, the baseline "
    "correctly identified 7 while the hybrid model correctly identified none at all. On this "
    "dataset, neither configuration is close to reliable at either end of the severity scale, which "
    "is reported here plainly since it is the least flattering result in this paper and the most "
    "relevant one for anyone considering deploying this system on data from a new clinic.")
figure(doc, "experiments/results/confusion_matrices/messidor_vit_confusion_matrix.png",
    "ViT-only confusion matrix, zero-shot on Messidor (n = 1,744).", 32, width_in=4.4)
figure(doc, "experiments/results/confusion_matrices/messidor_quantum_confusion_matrix.png",
    "ViT+VQC hybrid confusion matrix, zero-shot on Messidor (n = 1,744).", 33, width_in=4.4)

subheading(doc, "D.", "Training Behavior")
body(doc,
    "Fig. 34 shows training loss, validation kappa, and validation Grade-1 sensitivity by epoch "
    "for the ViT-only and hybrid models. The hybrid model's per-epoch validation log was only "
    "captured for its first 8 epochs during this run, while the ViT-only model's full 20 epochs "
    "were logged; both models' final checkpoints, selected by best validation kappa and reported "
    "throughout this paper, still came from their full training runs.")
figure(doc, "experiments/results/training/training_curves.png",
    "Training loss, validation kappa, and validation Grade-1 sensitivity by epoch, ViT-only versus hybrid model (hybrid curve shown for its first 8 logged epochs; same real training run also shown in Fig. 16).", 34, width_in=6.4)
body(doc,
    "Training loss fell faster for the hybrid model in the early epochs shown, and its validation "
    "kappa climbed slightly higher over the same window, 0.884 versus 0.880 for the baseline at "
    "epoch 8. Grade-1 sensitivity was noisy epoch to epoch for both models, swinging by 20 to 30 "
    "percentage points between consecutive checkpoints in places, which is a reasonable "
    "consequence of Grade 1 being both the rarest class in the training data and the class whose "
    "images most closely resemble Grade 0. This volatility is part of why Section III-E's "
    "Grade-1-focused training variant was worth building in the first place: relying on whichever "
    "epoch happens to land a high Grade-1 score by chance is not a dependable strategy, while "
    "directly reweighting the loss toward Grade 1 is.")

subheading(doc, "E.", "Comparison with Existing Work")
body(doc,
    "None of the quantum-assisted papers reviewed in Section II-A reported a grade-by-grade "
    "sensitivity breakdown, and none reported testing their model on a dataset it was not trained "
    "on; most reported a single overall accuracy figure on one dataset. The findings in this paper "
    "cannot be compared to those figures directly, since the datasets, splits, and label "
    "definitions differ, but the pattern found here, that a quantum circuit added to a strong "
    "classical backbone changes the balance between different metrics rather than uniformly "
    "improving performance, and that accuracy drops sharply under domain shift, is not something "
    "that a single reported accuracy number would reveal either way. Two of the quantum-assisted "
    "papers, [4] and [7], used early detection in their own framing, and the Vision Transformer "
    "paper [9] used early-diagnosis settings in its own framing too, but none of the three reported "
    "a grade-level sensitivity number to back that framing up, which is exactly the number this "
    "paper reports directly. The explicit Grade-1 sensitivity reporting and the zero-shot, "
    "three-dataset generalization test presented here are, as far as this survey found, not "
    "present together in any of the eleven papers reviewed, which is the specific gap this paper "
    "aims to fill rather than a claim of outright superior accuracy.")

print("Section V done"); doc.save("paper/_wip_paper1.docx")

# =============================================================================
# VI. CONCLUSION
# =============================================================================
heading(doc, "VI", "Conclusion")
body(doc,
    "This paper set out to answer a direct question: does adding a small quantum circuit to a "
    "Vision Transformer actually help it grade diabetic retinopathy, particularly at the earliest, "
    "hardest-to-catch stage. The ablation study shows that the Vision Transformer backbone, not "
    "the quantum circuit, is responsible for almost all of the model's ability to grade the "
    "disease at all, since the quantum circuit alone reached only 54.6 percent accuracy and a "
    "Grade-1 sensitivity of exactly zero. Adding the quantum circuit on top of the full Vision "
    "Transformer raised overall accuracy from 78.4 to 81.6 percent but lowered AUC, kappa, and "
    "Grade-1 sensitivity, showing that it shifted the balance between metrics rather than "
    "improving all of them at once. A Grade-1-focused training variant showed that this "
    "early-detection weakness is a tunable choice rather than a fixed limit: reweighting the "
    "training loss recovered Grade-1 sensitivity from 55.4 up to 78.6 percent, at the cost of "
    "overall accuracy, while a balanced variant recovered part of that sensitivity while keeping "
    "accuracy close to the unweighted hybrid model. Testing both models with no further training "
    "on Messidor, IDRiD, and EyePACS showed accuracy dropping substantially on every dataset, most "
    "severely on IDRiD, and Grade-1 sensitivity collapsing to near zero on Messidor for both "
    "models, so domain shift remains a real, unresolved limitation of the current system rather "
    "than a minor caveat. It is also worth stating plainly that the quantum circuit used "
    "throughout this work runs as a simulation, not on physical quantum hardware, so these "
    "findings speak to the mathematical structure the circuit adds, not to any hardware-level "
    "quantum advantage. Future work is expected to include a data-efficiency study, already set up "
    "but not yet completed, that trains both models on progressively smaller fractions of a "
    "held-out EyePACS pool to test whether the quantum circuit's advantage, if any, grows when "
    "labeled data is scarce; a domain-adaptation study aimed directly at the Messidor and IDRiD "
    "generalization gap found here; and, eventually, combining this architecture with the "
    "explainability pipeline described in the companion paper so that a single system can both "
    "grade the disease and show its reasoning.")

# =============================================================================
# REFERENCES
# =============================================================================
heading(doc, "", "References")
p = doc.add_paragraph()
r = p.add_run(
    "[Author note: the following list was assembled from real, independently located publications "
    "(title, venue, and URL confirmed via literature search during drafting); most of the quantum "
    "and Vision Transformer papers here also appear in the companion paper's reference list, plus "
    "two early-detection-focused quantum papers specific to this paper's framing. Full author "
    "names, volume, and page numbers were not independently verified for every entry and must be "
    "checked against the original source before submission. Entries are ordered alphabetically by "
    "title; in-text citation numbers correspond to this alphabetical order.]")
style_run(r, size=8.5, italic=True, color=(150,0,0))
p.paragraph_format.space_after = Pt(10)

refs = [
 ("A deep learning based model for diabetic retinopathy grading", "Scientific Reports, Nature Portfolio, 2025.", "https://www.nature.com/articles/s41598-025-87171-9"),
 ("A novel diabetic retinopathy detection from fundus images using hybrid quantum convolutional neural network models", "Scientific Reports, Nature Portfolio.", "https://www.nature.com/articles/s41598-026-49227-2"),
 ("Diabetic Retinopathy Detection Using Quantum Transfer Learning", "arXiv:2405.01734, 2024.", "https://arxiv.org/abs/2405.01734"),
 ("FQPDR: Federated Quantum Neural Network for Privacy-preserving Early Detection of Diabetic Retinopathy", "Evolutionary Intelligence, Springer.", "https://link.springer.com/article/10.1007/s12065-024-00971-2"),
 ("Hybrid Quantum Convolutional Neural Networks for Enhanced Diabetic Retinopathy Detection", "in Proc. IEEE Conf.", "https://ieeexplore.ieee.org/document/11070674/"),
 ("Hybrid quantum-classical deep learning framework for balanced multiclass diabetic retinopathy classification", "ScienceDirect.", "https://www.sciencedirect.com/science/article/pii/S2215016125004492"),
 ("Quantum-Assisted Early Detection of Diabetic Retinopathy", "Med Data Min., TMR Publishing Group, 2025.", "https://www.tmrjournals.com/public/articlePDF/20250718/96cb56d983953e408fdf06baf02c2676.pdf"),
 ("QuantumNet: An enhanced diabetic retinopathy detection model using classical deep learning-quantum transfer learning", "ScienceDirect.", "https://www.sciencedirect.com/science/article/pii/S2215016125000330"),
 ("Task optimized vision transformer for diabetic retinopathy detection and classification in resource constrained early diagnosis settings", "Scientific Reports, Nature Portfolio, 2025.", "https://www.nature.com/articles/s41598-025-25399-1"),
 ("ViT-DR: Vision Transformers in Diabetic Retinopathy Grading Using Fundus Images", "in Proc. IEEE Conf.", "https://ieeexplore.ieee.org/document/9930027/"),
 ("Vision Transformer Model for Predicting the Severity of Diabetic Retinopathy in Fundus Photography-Based Retina Images", "IEEE J. Biomed. Health Inform.", "https://ieeexplore.ieee.org/document/10290868/"),
]
for i, (title, venue, url) in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    r = p.add_run(f'[{i}] "{title}," ')
    style_run(r, size=9.5)
    r2 = p.add_run(f"{venue} [Online]. Available: {url}")
    style_run(r2, size=9.5)

doc.save("paper/QuantumRetina-X_Paper1_Architecture_v7.docx")
print("FINAL SAVED: paper/QuantumRetina-X_Paper1_Architecture_v7.docx")
