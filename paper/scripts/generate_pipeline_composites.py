"""Builds composite figures from the real per-stage PNGs already saved by
generate_pipeline_figures.py (same real prediction run, no new data)."""
from PIL import Image, ImageDraw, ImageFont

OUT = "paper"

def label(draw, xy, text, font):
    draw.text(xy, text, fill=(20, 20, 20), font=font)

def load_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def triptych():
    names = ["fig_stage_input", "fig_stage_heatmap", "fig_stage_multicolor_merge"]
    titles = ["Original retinal image", "Grad-CAM heatmap (AI focus)", "Lesion-type map (TCAV)"]
    imgs = [Image.open(f"{OUT}/{n}.png").convert("RGB") for n in names]
    w, h = imgs[0].size
    pad, label_h = 10, 28
    canvas = Image.new("RGB", (w * 3 + pad * 4, h + label_h + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (w + pad)
        canvas.paste(img, (x, label_h + pad))
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, 6), title, fill=(20, 20, 20), font=font)
    canvas.save(f"{OUT}/fig_stage_triptych.png")
    print(f"saved {OUT}/fig_stage_triptych.png")

def concept_grid():
    concepts = ["microaneurysm", "hemorrhage", "exudate", "edema"]
    imgs = [Image.open(f"{OUT}/fig_stage_concept_{c}.png").convert("RGB") for c in concepts]
    w, h = imgs[0].size
    pad, label_h = 8, 24
    canvas = Image.new("RGB", (w * 2 + pad * 3, (h + label_h) * 2 + pad * 3), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    for i, (img, name) in enumerate(zip(imgs, concepts)):
        row, col = divmod(i, 2)
        x = pad + col * (w + pad)
        y = pad + row * (h + label_h + pad)
        canvas.paste(img, (x, y + label_h))
        title = name.capitalize()
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, y + 2), title, fill=(20, 20, 20), font=font)
    canvas.save(f"{OUT}/fig_stage_concept_grid.png")
    print(f"saved {OUT}/fig_stage_concept_grid.png")

def mask_before_after():
    names = ["fig_stage_input", "fig_stage_fundus_mask"]
    titles = ["Before: raw resized image", "After: fundus mask applied (Sec. III-G)"]
    imgs = [Image.open(f"{OUT}/{n}.png").convert("RGB") for n in names]
    w, h = imgs[0].size
    pad, label_h = 10, 26
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (w + pad)
        canvas.paste(img, (x, label_h + pad))
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, 6), title, fill=(20, 20, 20), font=font)
    canvas.save(f"{OUT}/fig_stage_mask_before_after.png")
    print(f"saved {OUT}/fig_stage_mask_before_after.png")

def redfree_before_after():
    names = ["fig_stage_input", "fig_stage_redfree"]
    titles = ["Standard color fundus photo", "Red-free view (green channel + CLAHE)"]
    imgs = [Image.open(f"{OUT}/{n}.png").convert("RGB") for n in names]
    w, h = imgs[0].size
    pad, label_h = 10, 26
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = pad + i * (w + pad)
        canvas.paste(img, (x, label_h + pad))
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, 6), title, fill=(20, 20, 20), font=font)
    canvas.save(f"{OUT}/fig_stage_redfree_before_after.png")
    print(f"saved {OUT}/fig_stage_redfree_before_after.png")

if __name__ == "__main__":
    triptych()
    concept_grid()
    mask_before_after()
    redfree_before_after()
    print("Composites done.")
