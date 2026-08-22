"""
Loader con limpieza para el dataset VizWiz-Captions v1.0, en formato Hugging Face `datasets`.

El dataset (Gurari et al., ECCV 2020) contiene fotos tomadas por personas ciegas,
con 5 captions por imagen recogidas por crowdsourcing. Este módulo:

  * lee las anotaciones estilo COCO (`annotations/annotations/{train,val,test}.json`),
  * filtra las captions "precanned" (texto fijo de baja calidad) y las "rejected",
  * empareja cada imagen con su lista de captions limpias,
  * adjunta la imagen como columna `datasets.Image` (carga perezosa desde disco),
  * ignora automáticamente los .jpg de disco que no están en el JSON.

Uso básico:

    from wizviz_captions import load_wizviz_captions
    ds = load_wizviz_captions()                 # DatasetDict con train/val/test
    ds = load_wizviz_captions(format="exploded") # una fila por (imagen, caption)

Ejecuta `python wizviz_captions.py` para ver estadísticas y un ejemplo.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from datasets import Dataset, DatasetDict, Features, Image, Sequence as HFSequence, Value

# Texto fijo que los anotadores insertaban cuando la imagen era irreconocible.
PRECANNED_TEXT = "Quality issues are too severe to recognize visual content."


def _split_paths(root: Path, split: str) -> tuple[Path, Path]:
    """Devuelve (ruta del json, carpeta de imágenes) para un split."""
    ann = root / "annotations" / "annotations" / f"{split}.json"
    img_dir = root / split / split  # las imágenes están anidadas: train/train/*.jpg
    return ann, img_dir


def _clean_split(
    root: Path,
    split: str,
    *,
    drop_precanned: bool,
    drop_rejected: bool,
    min_captions: int,
) -> list[dict]:
    """Lee un split, filtra captions y agrupa por imagen. Devuelve lista de dicts."""
    ann_path, img_dir = _split_paths(root, split)
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    # Metadatos por imagen.
    images = {img["id"]: img for img in data["images"]}

    # Agrupar captions limpias por image_id (test.json no trae "annotations").
    caps_by_img: dict[int, list[str]] = defaultdict(list)
    for a in data.get("annotations", []):
        if drop_precanned and a["is_precanned"]:
            continue
        if drop_rejected and a["is_rejected"]:
            continue
        caps_by_img[a["image_id"]].append(a["caption"])

    is_test = "annotations" not in data
    rows: list[dict] = []
    for img_id, img in images.items():
        captions = caps_by_img.get(img_id, [])
        # En test no hay captions; en train/val exigimos un mínimo tras limpiar.
        if not is_test and len(captions) < min_captions:
            continue
        rows.append(
            {
                "image_id": img_id,
                "file_name": img["file_name"],
                "image_path": str(img_dir / img["file_name"]),
                "captions": captions,
                "text_detected": bool(img["text_detected"]),
                "vizwiz_url": img["vizwiz_url"],
            }
        )
    return rows


def _to_dataset(rows: list[dict], *, with_images: bool, exploded: bool) -> Dataset:
    """Convierte filas agrupadas en un `datasets.Dataset` (agrupado o explotado)."""
    if exploded:
        flat: list[dict] = []
        for r in rows:
            caps = r["captions"] or [None]  # test: una fila con caption=None
            for cap in caps:
                flat.append(
                    {
                        "image_id": r["image_id"],
                        "file_name": r["file_name"],
                        "image_path": r["image_path"],
                        "caption": cap,
                        "text_detected": r["text_detected"],
                        "vizwiz_url": r["vizwiz_url"],
                    }
                )
        rows = flat
        caption_feature = {"caption": Value("string")}
    else:
        caption_feature = {"captions": HFSequence(Value("string"))}

    features = Features(
        {
            "image_id": Value("int64"),
            "file_name": Value("string"),
            "image_path": Value("string"),
            **caption_feature,
            "text_detected": Value("bool"),
            "vizwiz_url": Value("string"),
        }
    )

    # from_list preserva el orden; construimos columnas explícitas para fijar el esquema.
    cols = {k: [r[k] for r in rows] for k in features}
    ds = Dataset.from_dict(cols, features=features)

    if with_images:
        # Columna Image de carga perezosa: apunta a la ruta local, decodifica al acceder.
        ds = ds.add_column("image", ds["image_path"]).cast_column("image", Image())
    return ds


def load_wizviz_captions(
    root: str | os.PathLike = ".",
    splits: Sequence[str] = ("train", "val", "test"),
    *,
    drop_precanned: bool = True,
    drop_rejected: bool = True,
    min_captions: int = 1,
    format: str = "grouped",
    with_images: bool = True,
) -> DatasetDict:
    """Carga VizWiz-Captions como DatasetDict de Hugging Face, con limpieza.

    Args:
        root: raíz del repo (donde están las carpetas train/ val/ test/ annotations/).
        splits: qué particiones cargar.
        drop_precanned: descarta captions "precanned" (texto fijo de imagen irreconocible).
        drop_rejected: descarta captions marcadas como rechazadas por control de calidad.
        min_captions: en train/val, descarta imágenes que queden con menos captions que esto.
        format: "grouped" (una fila por imagen, columna `captions`: list[str]) o
                "exploded" (una fila por caption, columna `caption`: str) para entrenar.
        with_images: si True, añade la columna `image` (datasets.Image, carga perezosa).

    Returns:
        DatasetDict con las particiones solicitadas.
    """
    root = Path(root)
    if format not in ("grouped", "exploded"):
        raise ValueError(f"format debe ser 'grouped' o 'exploded', no {format!r}")

    out: dict[str, Dataset] = {}
    for split in splits:
        rows = _clean_split(
            root,
            split,
            drop_precanned=drop_precanned,
            drop_rejected=drop_rejected,
            min_captions=min_captions,
        )
        out[split] = _to_dataset(
            rows, with_images=with_images, exploded=(format == "exploded")
        )
    return DatasetDict(out)


def _demo() -> None:
    """Carga los tres splits, imprime estadísticas y un ejemplo."""
    root = Path(__file__).resolve().parent

    print(">> Formato agrupado (una fila por imagen):")
    ds = load_wizviz_captions(root=root)
    for split, d in ds.items():
        n_caps = sum(len(c) for c in d["captions"]) if len(d) else 0
        n_txt = sum(d["text_detected"]) if len(d) else 0
        print(
            f"  {split:5s}: {len(d):>6d} imágenes | {n_caps:>7d} captions limpias "
            f"| {n_txt:>6d} con texto"
        )

    ex = ds["train"][0]
    print("\n>> Ejemplo (train[0]):")
    print("   file_name    :", ex["file_name"])
    print("   text_detected:", ex["text_detected"])
    print("   captions     :")
    for c in ex["captions"]:
        print("      -", c)
    print("   image        :", ex["image"].size, ex["image"].mode)

    print("\n>> Formato explotado (una fila por caption), solo train, sin imágenes:")
    de = load_wizviz_captions(
        root=root, splits=("train",), format="exploded", with_images=False
    )
    print(f"   train: {len(de['train'])} filas (imagen, caption)")
    print("   columnas:", de["train"].column_names)


if __name__ == "__main__":
    _demo()
