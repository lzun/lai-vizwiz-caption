"""¿Cuántas imágenes de val con texto tienen prefijo ADE no vacío en la caché?
Clave para saber si el modelo ADE corrió SIN prefijo en val completo (confound)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wizviz_captions import load_wizviz_captions
from baseline_blip_ocr import clean_ocr

root = Path(__file__).resolve().parent
val = load_wizviz_captions(root=str(root), splits=("val",), format="grouped", with_images=False)["val"]
ade = json.loads((root / "ocr_cache_landingai.json").read_text(encoding="utf-8"))
easy = json.loads((root / "ocr_cache_easyocr.json").read_text(encoding="utf-8"))

N = len(val["image_id"])
for label, lo, hi in [("primeras 2000", 0, 2000), ("val completo", 0, N)]:
    text = adef = adenonempty = easynonempty = 0
    for i in range(lo, min(hi, N)):
        fn = val["file_name"][i]
        if val["text_detected"][i]:
            text += 1
            if clean_ocr(ade.get(fn, ""), "ade_full").strip():
                adenonempty += 1
            if (easy.get(fn, "") or "").strip():
                easynonempty += 1
    print(f"{label}: imgs-con-texto={text} | con prefijo ADE no vacío={adenonempty} "
          f"({100*adenonempty/max(text,1):.1f}%) | con EasyOCR no vacío={easynonempty}")
print(f"\nEntradas totales en caché ADE: {len(ade)} | EasyOCR: {len(easy)}")
# ¿cuántas claves de la caché ADE son de val?
valfiles = set(val["file_name"])
ade_val = sum(1 for k in ade if k in valfiles and clean_ocr(ade[k], "ade_full").strip())
print(f"Claves ADE que son de val y no vacías: {ade_val}")
