"""Real visual before/after for feature extraction: the same example image,
plain, next to the same image with the real 14x14 patch grid (16x16-pixel
cells) drawn directly on it - the actual patchification ViT-Base/16 applies,
not an illustrative diagram."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

img = Image.open("paper/fig_stage_input.png").convert("RGB")
w, h = img.size  # 224, 224

gridded = img.copy()
draw = ImageDraw.Draw(gridded)
patch = 16
for x in range(0, w + 1, patch):
    draw.line([(x, 0), (x, h)], fill=(59, 130, 246), width=1)
for y in range(0, h + 1, patch):
    draw.line([(0, y), (w, y)], fill=(59, 130, 246), width=1)

def load_font(size=12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

pad, label_h = 14, 40
canvas = Image.new("RGB", (w * 2 + pad * 3, h + label_h + pad * 2), (255, 255, 255))
d = ImageDraw.Draw(canvas)
font = load_font(12)
titles = ["Before: plain resized\nimage (224x224x3)", "After: real 14x14 patch grid\n(196 patches of 16x16 px)"]
for i, (im, title) in enumerate(zip([img, gridded], titles)):
    x = pad + i * (w + pad)
    canvas.paste(im, (x, label_h + pad))
    d.multiline_text((x + w / 2, 6), title, fill=(20, 20, 20), font=font, anchor="ma", align="center")
canvas.save("paper/fig_stage_patchgrid_visual.png")
print("saved paper/fig_stage_patchgrid_visual.png")
