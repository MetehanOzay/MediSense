# check_images.py
import os
from PIL import Image, UnidentifiedImageError

ROOT = "assets/foods"

bad = []

for dirpath, _, files in os.walk(ROOT):
    for f in files:
        p = os.path.join(dirpath, f)
        size = os.path.getsize(p)
        try:
            with Image.open(p) as im:
                im.verify()  # sadece doğrulama, açmıyor
        except (UnidentifiedImageError, OSError) as e:
            bad.append((p, size, str(e)))

print("Problemli dosya sayısı:", len(bad))
for p, size, err in bad:
    print(f"- {p} | {size} bytes | {err}")
