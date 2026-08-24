# VizWiz-Captions — Experimentos de Image Captioning

Bitácora del proyecto: dataset, entorno, baseline con fine-tuning (BLIP + LoRA),
resultados y lecciones aprendidas. Última actualización: **2026-08-24**.

> **TL;DR** — Fine-tuning de BLIP-base con LoRA sobre VizWiz-Captions, entrenado en
> una laptop RTX 5060 (8 GB) **dentro de WSL2**. Mejor resultado: **CIDEr 60.3** en val
> (el modelo base sin adaptar da 43.5; el paper original reporta ~59–61).

---

## 1. Estructura del repositorio

```
wizviz-caption/
├── train/train/                 23 954 .jpg  (VizWiz_train_*.jpg)
├── val/val/                      7 750 .jpg
├── test/test/                    8 000 .jpg
├── annotations/annotations/
│   ├── train.json                info + images + annotations  (estilo COCO)
│   ├── val.json
│   └── test.json                 SIN annotations (hold-out)
├── VizWiz_captions_project_all_metadata.csv   metadata crudo de crowdsourcing (~197k filas, latin-1)
├── 2002.08565v2.pdf              paper del dataset (Gurari et al., ECCV 2020)
├── 2020_CSCW_HopeThisHelps.pdf   paper del proceso de crowdsourcing (CSCW 2020)
│
│   # --- creados en este proyecto ---
├── wizviz_captions.py            loader HF datasets con limpieza
├── 01_exploratory_analysis.ipynb análisis exploratorio (EDA) con gráficos
├── baseline_blip_lora.py         fine-tuning LoRA + evaluación
├── infer_caption.py              inferencia / demo + galería HTML
├── ocr_compare.py                comparación OCR: EasyOCR vs Landing AI (Fase 1)
├── ocr_precompute.py             precómputo/caché de OCR EasyOCR por imagen (Fase 2)
├── ocr_landingai_precompute.py   precómputo async de OCR Landing AI ADE (Fase 3)
├── baseline_blip_ocr.py          captioning con prefijo OCR + eval por subconjunto (Fases 2-3)
├── reporte_baseline.html         reporte visual de resultados
└── README.md                     este archivo
```

---

## 2. El dataset

**VizWiz-Captions v1.0**: fotos reales tomadas por personas ciegas, con **5 captions por
imagen** recogidas por crowdsourcing. A diferencia de COCO, hay problemas de calidad genuinos
(desenfoque, iluminación, encuadre) y mucho texto en imagen (etiquetas, pantallas, envases).

**Formato de anotaciones (COCO-style):**
```jsonc
"images":      { "file_name", "vizwiz_url", "id", "text_detected" }
"annotations": { "caption", "image_id", "is_precanned", "is_rejected", "id", "text_detected" }
```

**Limpieza aplicada por el loader** (descarta captions basura antes de entrenar):

| Partición | Imágenes (JSON) | Captions crudas | Captions limpias | Imágenes usadas |
|-----------|----------------:|----------------:|-----------------:|----------------:|
| train     | 23 431          | 117 155         | 100 575          | 22 866          |
| val       | 7 750           | 38 750          | 33 145           | 7 542           |
| test      | 8 000           | — sin captions —| —                | 8 000           |

- **`is_precanned`**: texto fijo *“Quality issues are too severe to recognize visual content.”*
  cuando la imagen es irreconocible → se filtra.
- **`is_rejected`**: captions marcadas de baja calidad por QA → se filtra.
- **`text_detected`**: ~63 % de imágenes tienen texto (el reto característico del dominio → OCR).

**Consideraciones del dataset:**
- ⚠️ **`test.json` no tiene captions** (hold-out del reto oficial) → toda la evaluación se hace en **val**.
- ⚠️ Hay **523 .jpg extra en `train/train`** que no están en `train.json`. El loader los ignora
  automáticamente porque itera los `file_name` del JSON, no el disco.
- ⚠️ El **CSV de metadata es `latin-1`**, no UTF-8 (`pd.read_csv(..., encoding="latin-1")`).

---

## 3. Entorno de ejecución (lo más importante)

### El problema que definió el setup

PyTorch **no cargaba en ningún Python de Windows** con el error
`OSError [WinError 4551]`. Causa: **Smart App Control** (Windows 11) está **activo en enforcement**
y bloquea la carga de DLLs sin firmar como `c10.dll` de PyTorch.

> ⚠️ **Desactivar Smart App Control es irreversible** (no se puede reactivar sin resetear Windows).
> Por eso se optó por **entrenar dentro de WSL2**, donde SAC no aplica a binarios Linux.

### Reglas clave aprendidas

- **CUDA Toolkit del sistema ≠ PyTorch.** Torch trae su propio runtime CUDA en la rueda (wheel);
  no necesita coincidir con el `nvcc` del sistema (aquí 13.2).
- **RTX 50xx es Blackwell (sm_120)** → requiere torch build **cu128** (`--index-url .../whl/cu128`).
- **Python 3.14 no tiene wheels de torch** (ni Windows ni el Python de sistema de Ubuntu 26.04).
  Se usa **`uv`** para gestionar un **Python 3.12** dedicado.
- **El venv va en el filesystem de Linux** (`~/wizviz/.venv`), no en `/mnt/c` (mucho más rápido).
  El dataset se **lee** desde `/mnt/c/...` sin copiarlo.

### Mapa de entornos

| Entorno | Python | Uso | torch |
|---------|--------|-----|-------|
| WSL2 `~/wizviz/.venv` (uv) | 3.12.14 | **entrenamiento / inferencia** | 2.11.0+cu128 ✅ |
| Anaconda `base` (Windows) | 3.13.9 | **solo el notebook EDA** (Jupyter) | — |
| Anaconda `pytorch-env` | 3.13.14 | (no usado) torch cu132 pero **bloqueado por SAC** | ❌ |
| pythoncore-3.14 (CMD) | 3.14.6 | (no usado) sin wheels de torch | ❌ |

### Hardware

`NVIDIA GeForce RTX 5060 Laptop GPU · 8 GB · sm_120 · driver 610.74`. La GPU funciona dentro de
WSL2 por passthrough (verificable con `nvidia-smi` en Ubuntu; **no** hay que instalar driver en Linux).

### Reconstruir el entorno WSL desde cero

```bash
# 1) Instalar WSL2 + Ubuntu (PowerShell/CMD como ADMIN, reinicia después)
wsl --install

# 2) Dentro de Ubuntu: instalar uv + Python 3.12 + venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
mkdir -p ~/wizviz && cd ~/wizviz
uv venv --python 3.12 .venv

# 3) Instalar torch (Blackwell → cu128) y el stack de ML
uv pip install --python ~/wizviz/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python ~/wizviz/.venv/bin/python transformers datasets peft accelerate evaluate sacrebleu rouge_score pycocoevalcap pandas pillow tqdm

# 4) Verificar GPU
~/wizviz/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# esperado: 2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU
```

Para el **notebook EDA** (corre en Anaconda `base`, no en WSL): `pip install datasets` en ese
entorno. Si un notebook falla con `ModuleNotFoundError`, instala en el kernel exacto con
`%pip install <paquete>` dentro de una celda.

---

## 4. Cómo ejecutar

Los scripts viven en `/mnt/c/Users/green/Documents/GitHub/wizviz-caption` y corren con el Python
del venv de WSL. Patrón de invocación **desde Windows** (PowerShell):

```powershell
wsl -e bash -lc '$HOME/wizviz/.venv/bin/python /mnt/c/Users/green/Documents/GitHub/wizviz-caption/<script>.py <args>'
```

> 💡 **Quoting PowerShell → bash es frágil**: los `|`, `&&`, `$(...)` y comillas internas rompen
> los comandos multilínea. Para algo no trivial, escribe un `.sh` (con saltos de línea LF) y ejecútalo
> con `wsl -e bash /mnt/c/.../script.sh`.

### Loader

```python
from wizviz_captions import load_wizviz_captions
ds = load_wizviz_captions()                      # grouped: 1 fila/imagen, captions: list[str]
ds = load_wizviz_captions(format="exploded")     # 1 fila por (imagen, caption)  → entrenamiento
# opciones: root, splits, drop_precanned, drop_rejected, min_captions, with_images
```

### Análisis exploratorio (EDA)

Abrir `01_exploratory_analysis.ipynb` en Jupyter (kernel Anaconda `base`) y *Run All*.
Cubre distribución de longitudes, vocabulario, texto en imagen, problemas de calidad y ejemplos.

### Entrenamiento del baseline

```bash
# Baseline completo (BLIP-base + LoRA, ~76 min en la RTX 5060)
python baseline_blip_lora.py --epochs 3 --batch-size 16 --max-eval-samples 2000

# Referencia del modelo base, sin fine-tuning (~4 min)
python baseline_blip_lora.py --no-train --max-eval-samples 2000

# Corrida rápida de validación del pipeline
python baseline_blip_lora.py --max-train-samples 500 --max-eval-samples 100 --epochs 1
```

Argumentos útiles: `--lora-r`, `--lr`, `--num-beams`, `--grad-accum`, `--grad-checkpointing`,
`--output-dir`. El adaptador LoRA se guarda en `~/wizviz/runs/`.

### Inferencia / demo

```bash
# 8 imágenes aleatorias del test set, base vs fine-tuned, a galería HTML
python infer_caption.py --split test --num 8 --random --compare --html demo.html

# Imágenes propias
python infer_caption.py foto1.jpg foto2.jpg

# Modelo base sin adaptador
python infer_caption.py --split val --num 4 --base-only
```

---

## 5. Resultados

Evaluación sobre **2 000 imágenes de val**, beam = 3.

| Modelo | CIDEr | BLEU-4 | BLEU-1 | ROUGE-L |
|--------|------:|-------:|-------:|--------:|
| BLIP base (sin fine-tuning) | 43.49 | 20.87 | 63.27 | 43.09 |
| LoRA · 20k pares · 2 épocas | 54.40 | 22.59 | 65.57 | 45.99 |
| **LoRA · train completo · 3 épocas** | **60.28** | **24.26** | **67.18** | **46.83** |

- El fine-tuning aporta **+16.8 CIDEr** sobre el base; escalar de 20k → 100k pares suma otros +5.9.
- **CIDEr 60.3 cae en el rango del paper original (~59–61)**, logrado con LoRA (0.52 % de params) en 8 GB.
- Entrenamiento full: 3 épocas / 100 575 pares en **4 559 s (~76 min)** a 66 pares/s, loss 3.69 → 2.98.

**Ejemplo cualitativo** (mejora de dominio):
`base: "a person holding a bottle of liquid"` → `fine-tuned: "a person is holding a bottle of medicine"`.

### Detalles técnicos del baseline

- Modelo: `Salesforce/blip-image-captioning-base` (225M params), fp16.
- LoRA: r=16, α=32, sobre `query`/`value` de la atención → **1.18M params entrenables (0.52 %)**.
- **batch 16 cabe en 8 GB** en fp16 sin OOM.
- Métricas vía `pycocoevalcap` con **tokenización propia** (lowercase + strip de puntuación) para
  **evitar la dependencia de Java** de PTBTokenizer.

---

## 6. Exploración de OCR (Fases 1, 2 y 3)

El 63 % de imágenes tienen texto, así que se exploró añadir OCR. Se comparan dos motores sobre
**30 imágenes de val con `text_detected`** (`ocr_compare.py` → galería `ocr_compare.html`).
Como VizWiz-Captions **no trae ground-truth de OCR**, la comparación es cualitativa.

| Motor | Coste | Fallos / vacías | Texto extraído | Comportamiento |
|-------|-------|-----------------|----------------|----------------|
| **EasyOCR** (tradicional, GPU) | gratis | 2 fallos + 4 vacías (6/30) | 6 557 chars | OCR crudo; se degrada mucho en fotos difíciles |
| **Landing AI ADE** (de paga) | ~30 créditos (de 1 000 gratis) → $0 | 0 | 21 988 chars (**3.4×**) | Nunca falla; transcribe documentos limpio y **describe** las fotos |

**Hallazgos:**
- **Landing AI es mucho más robusto.** Nunca falla ni devuelve vacío; extrae 3.4× más texto.
  En texto tipo documento lo transcribe casi perfecto (p. ej. `User account | GeoAccess - Internet
  Explorer http://geoaccess.org/...`) donde EasyOCR devuelve ruido (`ccount Gcoaccess... optin htp`).
- **EasyOCR se hunde en fotos degradadas** (blur, ángulo, poca luz): produce cadenas como
  `4 3 I 3 3 Mm 8 0 S 8 B...` o directamente no procesa la imagen.
- **Ojo — no es una comparación pareja.** Landing AI ADE **no es OCR puro**: es *document
  understanding* agéntico. En 20/30 imágenes devolvió una **descripción de escena**
  (`<figure><description>…`), no solo texto verbatim — a menudo leyendo la marca del producto
  (`Verbatim`, `McDonald's`) y quedando **más cerca de una caption** que del OCR. EasyOCR da tokens
  crudos; ADE da texto + descripción (más útil, pero no literal).

**Implicación para la Fase 2:** para inyectar OCR como señal al captioner, EasyOCR aporta tokens
crudos baratos; Landing AI aporta texto+descripción de más calidad pero con coste por imagen.

### Fase 2 — Inyectar OCR en el captioner (resultado negativo)

Se inyectó el texto de **EasyOCR** (precomputado con `ocr_precompute.py`) como **prefijo
condicional del decoder** de BLIP: en entrenamiento se enmascara el prefijo en la loss; en
inferencia se genera a partir de `[CLS]+<OCR>` y se recorta el prefijo. Experimento A/B controlado
sobre el **mismo subset de 20k pares / 2 épocas** (`baseline_blip_ocr.py`), evaluando en 2 000 val
con métricas **desglosadas por `text_detected`**.

| Subconjunto (CIDEr) | Control (sin OCR) | Con OCR (EasyOCR) | Δ |
|---------------------|------------------:|------------------:|----:|
| Todas (n=2000)      | 54.49 | 51.14 | **−3.35** |
| **Con texto** (n=1559) | 51.68 | 47.37 | **−4.31** |
| Sin texto (n=441)   | 62.18 | 62.43 | +0.25 |

**Conclusión: la inyección naíf de OCR con EasyOCR NO ayuda — empeora, y justo en las imágenes
con texto.** El experimento es fiable: el subconjunto *sin texto* (donde no se inyecta OCR) queda
idéntico, así que el daño viene del **contenido ruidoso** del OCR, no de la mecánica. EasyOCR es
demasiado ruidoso en estas fotos degradadas (ver Fase 1). Hay ganancias puntuales (p. ej. una
botella pasó de *"bottle of cough"* a *"bottle of medicine"*), pero el ruido domina en promedio.

### Fase 3 — Prefijo con Landing AI ADE (dos variantes)

Se repitió el A/B usando **Landing AI ADE** como fuente del prefijo, precomputado en concurrencia
(`ocr_landingai_precompute.py`, solo imágenes con `text_detected` → ~4 500 créditos, 0 fallos).
Dos limpiezas del texto (`--ocr-clean`): **solo-texto** (transcripción literal, se quitan los
bloques `<description>`) y **ADE completo** (descripción de escena + texto). Mismo subset de 20k / 2 épocas.

| Variante (CIDEr) | Global | Con texto (n=1559) | Sin texto (n=441) |
|------------------|-------:|-------------------:|------------------:|
| Control (sin OCR)            | 54.49 | 51.68 | 62.18 |
| EasyOCR                      | 51.14 | 47.37 | 62.43 |
| Landing AI — solo texto      | 52.38 | 49.22 | 62.23 |
| **Landing AI — ADE completo**| **56.99** | **54.84** | 63.26 |

**Progresión monótona y clara:** EasyOCR (ruidoso) < LA solo-texto (limpio pero literal) < LA ADE
completo — y **solo la última supera al control** (+2.5 global, **+3.2 en imágenes con texto**).

**Interpretación honesta:** lo que ayuda **no es "leer el texto"** (el solo-texto literal sigue por
debajo del control), sino la **descripción de escena** de ADE, que actúa como señal de un captioner
externo fuerte — es esencialmente **destilación de un VLM**, no OCR. Coste de despliegue: la variante
ganadora requiere una llamada a Landing AI (~1 crédito/img) también en inferencia.

**Conclusión de las 3 fases:** el OCR *literal* (barato o caro) no mejora el captioning en VizWiz
—las captions humanas describen escenas, no transcriben—; pero inyectar la **descripción
multimodal** de una herramienta fuerte sí ayuda (+3.2 CIDEr en imágenes con texto).

**Notas de entorno OCR:**
- EasyOCR 1.7.2 **requiere OpenCV 4.x** (`opencv-python-headless<5`); con OpenCV 5 crashea en `resize`.
- Algunas imágenes degradadas rompen EasyOCR (recorte vacío) → `run_easyocr` lo captura por imagen.
- Clave de Landing AI en `~/wizviz/.env` (fuera del repo) como `VISION_AGENT_API_KEY=...` (gitignored).

---

## 7. Lecciones aprendidas / consideraciones

1. **Smart App Control bloquea PyTorch en Windows** (`WinError 4551`). Diagnóstico:
   `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`.
   Solución sin sacrificar seguridad: **WSL2**.
2. **CUDA Toolkit del sistema ≠ PyTorch.** No confundir tener `nvcc` con tener torch funcional.
3. **Blackwell (sm_120) → wheels cu128.** Las builds cu121/cpu no sirven.
4. **Evita Python 3.14 para ML** (aún sin wheels de torch/muchas libs). `uv` facilita fijar 3.12.
5. **Quoting PowerShell → bash/WSL**: usa scripts `.sh` para cualquier comando con `|`, `$()`, comillas.
6. **Mismatch de entorno en Jupyter**: `%pip install` instala en el kernel correcto.
7. **CSV de metadata en latin-1**, no UTF-8.
8. **Filtrar precanned/rejected** es imprescindible; sin ello se entrena con ruido/frases fijas.
9. **test sin captions** → evaluar en val; **523 imágenes extra** en train se ignoran vía JSON.
10. **venv en Linux fs, datos en /mnt/c**: buen equilibrio velocidad/espacio (no copiar 30 GB).
11. **EasyOCR necesita OpenCV 4.x** (`opencv-python-headless<5`) y hay que envolver `readtext`
    en try/except (algunas fotos degradadas lo crashean).
12. **WSL se cae si la laptop suspende** (`Wsl/Service/CreateInstance/0x800705b4` al reanudar):
    tumba los jobs largos. Mitigación: mantener la laptop despierta, y hacer los precómputos
    **reanudables** (el caché OCR se guarda cada 100 imgs). Recuperación: `wsl --shutdown` y reintentar.
13. **OCR-prefix en BLIP**: forzar `min_new_tokens` en generación; si no, un prefijo OCR ruidoso
    colapsa la salida a vacío. BOS del decoder = `[DEC]` (30522), EOS = `[SEP]` (102).

---

## 8. Próximos pasos

- [x] **OCR — Fase 1**: comparar EasyOCR vs Landing AI ADE (ver §6). Landing AI 3.4× más texto y más robusto.
- [x] **OCR — Fase 2**: prefijo OCR (EasyOCR) en BLIP → **resultado negativo** (CIDEr −3.35, −4.31 en imágenes con texto). El OCR ruidoso perjudica.
- [x] **OCR — Fase 3**: prefijo Landing AI ADE. Solo-texto sigue bajo el control; **ADE completo (descripción) SÍ mejora** (+2.5 global, **+3.2 en imágenes con texto**). Ayuda la descripción del VLM, no el OCR literal.
- [ ] **OCR — Fase 4 (idea)**: destilación explícita desde ADE, o gatear la descripción solo cuando aporta; medir coste/beneficio en inferencia.
- [ ] **Submission al test oficial** de VizWiz para una cifra comparable públicamente.
- [ ] Escalar el fine-tuning: más épocas, LoRA de mayor rango, o afinar el encoder de visión.
- [ ] Augmentations que imiten blur/encuadre para las imágenes más difíciles.

---

## 9. Referencias

- D. Gurari, Y. Zhao, M. Zhang, N. Bhattacharya. *Captioning Images Taken by People Who Are Blind.*
  ECCV 2020. (`2002.08565v2.pdf`) · <https://vizwiz.org>
- R. N. Simons, D. Gurari, K. R. Fleischmann. *"I Hope This Is Helpful": Understanding Crowdworkers'
  Challenges and Motivations for an Image Description Task.* CSCW 2020. (`2020_CSCW_HopeThisHelps.pdf`)
- Licencia del dataset: CC BY 4.0.
