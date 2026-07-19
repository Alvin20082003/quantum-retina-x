from pathlib import Path
DATA = Path("data")
IMG_EXT = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}
try: import pandas as pd
except ImportError: pd = None
try: from PIL import Image
except ImportError: Image = None

def audit_labels(path):
    if pd is None: print("    (pandas missing)"); return
    try: df = pd.read_csv(path) if path.suffix.lower()==".csv" else pd.read_excel(path)
    except Exception as e: print(f"    could not read {path.name}: {e}"); return
    print(f"    label file: {path.name} -> {df.shape[0]} rows, columns: {list(df.columns)}")
    for c in df.columns:
        if str(c).lower() in ("diagnosis","level","grade","retinopathy grade","dr_grade"):
            print(f"    grade column '{c}' distribution:")
            for k,v in df[c].value_counts().sort_index().items(): print(f"       grade {k}: {v}")
            return
    print("    (no obvious grade column - check columns above)")

def check_images(imgs, n=30):
    if Image is None: print("    (Pillow missing)"); return
    import random; random.seed(0)
    sample = imgs if len(imgs)<=n else random.sample(imgs,n)
    bad,zero,sizes = [],[],[]
    for p in sample:
        if p.stat().st_size==0: zero.append(p.name); continue
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im: sizes.append(im.size)
        except Exception: bad.append(p.name)
    if sizes:
        ws=[s[0] for s in sizes]; hs=[s[1] for s in sizes]
        print(f"    sampled {len(sample)} imgs | size {min(ws)}x{min(hs)} to {max(ws)}x{max(hs)}")
    if zero: print(f"    !! {len(zero)} ZERO-BYTE files: {zero[:5]}")
    if bad:  print(f"    !! {len(bad)} UNREADABLE files: {bad[:5]}")
    if not zero and not bad: print("    all sampled images opened cleanly")

def main():
    if not DATA.exists(): print("No data/ folder. Run from repo root."); return
    print("="*60); print("DATA AUDIT"); print("="*60)
    for d in sorted([x for x in DATA.iterdir() if x.is_dir()]):
        imgs = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT]
        print(f"\n[{d.name}]  images found: {len(imgs)}")
        labs = [p for p in d.rglob("*") if p.suffix.lower() in {".csv",".xls",".xlsx"}]
        if not labs: print("    !! no CSV/Excel label file found")
        for lp in labs: audit_labels(lp)
        if imgs: check_images(imgs)
    print("\n"+"="*60+"\nDone. Paste this whole report back.\n"+"="*60)

if __name__=="__main__": main()
