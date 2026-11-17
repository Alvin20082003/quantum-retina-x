"""Renders the key equations as clean images for insertion into the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render(lines, filename, figsize=(6.0, None), fontsize=13, row_heights=None):
    n = len(lines)
    row_heights = row_heights or [1.0] * n
    total = sum(row_heights)
    h = figsize[1] or (0.42 * total + 0.25)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")
    y = 0.94
    step = 0.90 / total
    cursor = y
    for i, (eq, label) in enumerate(lines):
        ax.text(0.03, cursor, eq, fontsize=fontsize, va="top", ha="left")
        if label:
            ax.text(0.97, cursor, label, fontsize=fontsize - 2, va="top", ha="right", color="#555555")
        cursor -= step * row_heights[i]
    fig.tight_layout(pad=0.3)
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", filename)

# --- VQC equations ---
render([
    (r"$|\psi_i\rangle = R_Y(x_i)\,|0\rangle,\quad i=1,\dots,4$", "(1)"),
    (r"$U_l(\mathbf{\theta}) = \left(\prod_{i=1}^{3} \mathrm{CNOT}_{i,i+1}\right)"
     r"\prod_{i=1}^{4} R_Z(\theta_{l,i,3}) R_Y(\theta_{l,i,2}) R_X(\theta_{l,i,1})$", "(2)"),
    (r"$\langle Z_i \rangle = \langle \psi_2 | Z_i | \psi_2 \rangle,\quad "
     r"|\psi_2\rangle = U_2 U_1 \bigotimes_i |\psi_i\rangle$", "(3)"),
], "paper/fig_eq_vqc.png", figsize=(6.6, None), fontsize=12, row_heights=[1.0, 2.0, 1.0])

# --- Grad-CAM equations ---
render([
    (r"$\alpha_k^c = \dfrac{1}{Z}\sum_{i}\sum_{j}\dfrac{\partial y^c}{\partial A^k_{ij}}$", "(4)"),
    (r"$L^c_{\mathrm{GradCAM}} = \mathrm{ReLU}\left(\sum_k \alpha_k^c\, A^k\right)$", "(5)"),
], "paper/fig_eq_gradcam.png", figsize=(6.0, None), fontsize=13, row_heights=[1.8, 1.0])

# --- TCAV / probe equations ---
render([
    (r"$P(\mathrm{concept}\mid h) = \sigma(\mathbf{w}^{\top} h + b)$", "(6)"),
    (r"$\mathbf{v}_C = \mathbf{w} \,/\, \|\mathbf{w}\|$", "(7)"),
], "paper/fig_eq_tcav.png", figsize=(5.4, None), fontsize=13)

# --- Loss functions ---
render([
    (r"$\mathcal{L}_{\mathrm{DR}} = -\sum_{c=0}^{4} w_c\, y_c \log(\hat{y}_c)$", "(8)"),
    (r"$\mathcal{L}_{\mathrm{DME}} = -\left[p\, y \log\sigma(\hat{z}) + (1-y)\log(1-\sigma(\hat{z}))\right]$", "(9)"),
], "paper/fig_eq_loss.png", figsize=(6.2, None), fontsize=13, row_heights=[1.7, 1.0])
