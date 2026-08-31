# VizWiz-Captions — Experimentos de Image Captioning

Bitácora del proyecto: dataset, entorno, fine-tuning (BLIP + LoRA), OCR, escalado y
alternativas de VLM. Última actualización: **2026-08-29**.

> **TL;DR** — Image captioning sobre VizWiz-Captions, todo con LoRA en una laptop RTX 5060
> (8 GB) **dentro de WSL2**. Partiendo de BLIP-base (CIDEr 43.5 sin adaptar):
> - 🥇 **Mejor absoluto: CIDEr 78.7** (val) — BLIP + fine-tuning de visión + prefijo de descripción
>   de Landing AI ADE. Requiere pagar ADE por imagen en inferencia.
> - 🥈 **Mejor autónomo: CIDEr 74.6** (val) — BLIP + fine-tuning de visión (LoRA texto+visión, r=32),
>   gratis y local.
>
> **Hallazgos clave:** la palanca decisiva es adaptar el **encoder de visión** (+12.4); la **calidad
> del OCR es decisiva** (EasyOCR resta, ADE suma; simetría ~±4); ADE y visión se **complementan** (no
> son redundantes); en VizWiz gana la **descripción**, no la transcripción literal. GIT-large como
> backbone alternativo cuasi-empata a BLIP a escala. Ver §5–§9.

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
├── baseline_blip_distill.py      destilación: pseudo-captions como target (Fases 4 y 4b)
├── distill_teacher_gen.py        genera captions-profesor estilo-VizWiz (Fase 4b)
├── run_ade_vision_phaseA.sh      experimento ADE+visión Fase A — ejecutado (§9)
├── run_ade_vision_phaseB.sh      Fase B (full train, turnkey; requiere créditos) (§9)
├── baseline_git.py               prototipo GIT-large (VLM alternativo a BLIP)
├── baseline_qwen.py              prototipo Qwen2.5-VL-3B (QLoRA 4-bit)
├── eval_qwen.py                  re-eval Qwen con inglés forzado / bloqueo CJK
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
| LoRA texto · train completo · 3 épocas | 60.28 | 24.26 | 67.18 | 46.83 |
| LoRA texto+visión (r=32) · full · 2 épocas | 72.67 | 27.25 | 69.93 | 48.97 |
| **LoRA texto+visión (r=32) · full · 4 épocas** | **74.60** | **27.51** | **70.38** | **49.02** |

- El fine-tuning aporta **+16.8 CIDEr** sobre el base; escalar de 20k → 100k pares suma otros +5.9.
- **Escalar al encoder de visión es el mayor salto: +12.4 CIDEr** (60.3 → 72.7) al aplicar LoRA
  también al ViT (`qkv`) con r=32. Adaptar las *features visuales* al dominio (fotos borrosas,
  ángulos raros) pesa mucho más que solo adaptar el decoder de texto.
- Entrenamiento texto full: 3 épocas en ~76 min. Escalado (texto+visión): 2 épocas en **~104 min**
  (32 pares/s, VRAM 7.0/8.1 GB sin OOM, batch 12, grad-checkpointing no-reentrante disponible).
- **74.60 es el mejor modelo AUTÓNOMO** (gratis en inferencia). Sumar el prefijo de descripción de
  Landing AI ADE lo lleva a **78.68** (mejor absoluto, pero de pago en inferencia) — ver §6.
- Ejes ya explorados y saturados: rango (r=32 óptimo, r=64 no ayuda), épocas (4, rendimientos
  decrecientes), lr (1e-4), dropout (0.05). Ver checklist en §8.

**Ejemplo cualitativo** (mejora de dominio):
`base: "a person holding a bottle of liquid"` → `fine-tuned: "a person is holding a bottle of medicine"`.

### Detalles técnicos del baseline

- Modelo: `Salesforce/blip-image-captioning-base` (225M params), fp16.
- LoRA: r=16, α=32, sobre `query`/`value` de la atención → **1.18M params entrenables (0.52 %)**.
- **batch 16 cabe en 8 GB** en fp16 sin OOM.
- Métricas vía `pycocoevalcap` con **tokenización propia** (lowercase + strip de puntuación) para
  **evitar la dependencia de Java** de PTBTokenizer.

---

## 6. Exploración de OCR (Fases 1–4b)

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

### Fase 4 — Destilación desde ADE (resultado negativo)

Como la ganancia de la Fase 3 exige llamar a Landing AI en inferencia, se intentó **destilar**:
usar la primera oración de la descripción de ADE como **pseudo-caption extra** en entrenamiento
(`baseline_blip_distill.py`), sin prefijo en inferencia → modelo autónomo y gratis. Solo se usaron
1 981 descripciones limpias (filtradas de transcripciones ruidosas). A/B vs control, mismo subset.

| Modelo (CIDEr) | Global | Con texto | Sin texto | Dep. inferencia |
|----------------|-------:|----------:|----------:|-----------------|
| Control                  | 55.14 | 52.28 | 62.86 | ninguna |
| Fase 3 (ADE prefijo)     | 56.99 | 54.84 | 63.26 | Landing AI/img |
| **Destilado (Fase 4)**   | **33.54** | 29.53 | 46.67 | ninguna |

**Empeoró mucho (−21.6 CIDEr).** La destilación **contamina el estilo**: BLIP aprende la prosa
verbosa de ADE (*"a close-up, slightly blurry photograph of a red, white and blue tube of
toothpaste…"*) y la aplica en todas partes, incluso en imágenes sin texto (62.86→46.67, donde no
había pseudo-captions). Como CIDEr/BLEU premian coincidir con las captions humanas **cortas**, el
desajuste de estilo domina. Bastó un ~10% de ejemplos verbosos. El **conocimiento** sí se transfiere
(*"bottle of cough"* → *"medicine bottle"*), pero el estilo lo arruina.

**Lección:** destilar de un profesor con estilo distinto al de las referencias daña la métrica.

### Fase 4b — Destilación estilo-matched

Se corrigió el estilo: el profesor es el modelo de la Fase 3 (que ya produce captions cortas
estilo-VizWiz). Se generan sus captions CON prefijo ADE (`distill_teacher_gen.py` →
`teacher_captions.json`, 1 981 targets, ~11.5 palabras) y el alumno se entrena con ellas SIN prefijo
(`baseline_blip_distill.py --teacher-cache`). Mismas imágenes que la Fase 4; única variable: el target.

| Modelo (CIDEr) | Global | Con texto | Sin texto | Dep. inferencia |
|----------------|-------:|----------:|----------:|-----------------|
| Control                  | 55.14 | 52.28 | 62.86 | ninguna |
| Fase 3 (ADE prefijo)     | **56.99** | **54.84** | 63.26 | Landing AI/img 💰 |
| Fase 4 (prosa ADE)       | 33.54 | 29.53 | 46.67 | ninguna |
| **Fase 4b (profesor)**   | 54.19 | 50.89 | 63.31 | ninguna |

**El estilo era el problema** (33.54 → 54.19; el subconjunto sin texto se recupera del todo). **Pero
la destilación no aporta beneficio neto:** la 4b queda a la par del control (−0.9, dentro del ruido)
y por debajo de la Fase 3. Se **neutraliza el daño, pero no se captura la ganancia.**

**Conclusión del hilo de OCR (4 fases):** la ganancia de la descripción de ADE (+3.2 CIDEr en
imágenes con texto) **solo se materializa con la señal fresca por-imagen en inferencia** (Fase 3, que
paga ADE). Ni el OCR barato (Fase 2), ni destilar la prosa (Fase 4), ni destilar de un profesor
estilo-matched (Fase 4b) la meten "gratis" en un modelo autónomo. El valor del VLM fuerte reside en
su inferencia por-imagen, que un alumno pequeño no interioriza desde ~2 000 pseudo-etiquetas.

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
14. **El test set no es evaluable localmente.** Captions de test oficialmente ocultas; el CSV usa
    nombres originales que no mapean a los archivos renombrados. **Val es el held-out.** Lección:
    validar un mapeo con 2 muestras no basta — probar también índices tardíos (0 y 100 alineaban
    por casualidad, pero 5000/7000 no). `eval_test.py` queda como referencia, pero su GT es inválido.
15. **Vision LoRA + grad-checkpointing**: usar `use_reentrant=False`; el `enable_input_require_grads`
    choca con el `+=` in-place de los embeddings (pasa en BLIP y en GIT). O entrenar sin checkpointing
    (batch 12 BLIP / batch 8 GIT caben en 8 GB).
16. **La calidad del OCR es decisiva** (no basta "tener OCR"): EasyOCR resta ~−4, ADE suma ~+4, en
    ambos regímenes. Y en VizWiz gana la **descripción de escena**, no la transcripción literal
    (por eso EasyOCR/solo-texto/ADE-textonly restan; ADE-descripción suma).
17. **Fijar SIEMPRE todos los hiperparámetros en comparativas.** Un `--grad-accum` por defecto metió
    un confound (loss ×2 en logging + batch efectivo distinto); un `--lora-r` dispar sesgó GIT-vs-BLIP.
18. **Un prototipo de bajo dato puede engañar.** GIT superaba a BLIP en 20k (+4.5) pero cuasi-empataba
    a escala completa. Validar a escala Y con hiperparámetros pareados antes de concluir.

---

## 8. Próximos pasos

- [x] **OCR — Fase 1**: comparar EasyOCR vs Landing AI ADE (ver §6). Landing AI 3.4× más texto y más robusto.
- [x] **OCR — Fase 2**: prefijo OCR (EasyOCR) en BLIP → **resultado negativo** (CIDEr −3.35, −4.31 en imágenes con texto). El OCR ruidoso perjudica.
- [x] **OCR — Fase 3**: prefijo Landing AI ADE. Solo-texto sigue bajo el control; **ADE completo (descripción) SÍ mejora** (+2.5 global, **+3.2 en imágenes con texto**). Ayuda la descripción del VLM, no el OCR literal.
- [x] **OCR — Fase 4**: destilación (pseudo-captions de ADE) → **negativo** (CIDEr 55→34): contamina el estilo hacia la prosa verbosa de ADE. El conocimiento se transfiere pero el estilo arruina la métrica.
- [x] **OCR — Fase 4b**: destilación estilo-matched (profesor = modelo Fase 3). Arregla el colapso de la Fase 4 (34→54) pero **queda a la par del control** — neutraliza el daño, no captura la ganancia. La ventaja de ADE solo es realizable pagándolo en inferencia (Fase 3).
- [x] **Test set**: NO evaluable localmente. Las captions de test son oficialmente **ocultas** (VizWiz publica solo train/val). El CSV tiene captions TEST pero con nombres originales que **no mapean** a los archivos renombrados (se intentó un offset `20000+i` que falla; `test.json` no tiene captions con qué anclar por contenido). **Val es el conjunto de evaluación válido.**
- [x] Escalar el fine-tuning: LoRA texto+visión (r=32) → **CIDEr 60.3 → 72.7 (+12.4)**. El encoder de
  visión era el eje clave (`--target-modules query,value,qkv --lora-r 32 --save-steps 800`, reanudable).
- [x] Escalar más: 2→4 épocas → CIDEr 72.7 → **74.6 (+1.9)**. Rendimientos decrecientes sin overfitting;
  la palanca era la visión, no las épocas. (Una corrida crasheó con `CUDA error: unknown error` y se
  **reanudó desde checkpoint** sin re-entrenar — el diseño reanudable funcionó.)
- [x] Escalar r=64 sobre texto+visión → CIDEr 74.02 (vs 74.60 con r=32): **no ayuda, saturado**. r=32 es
  el punto dulce. Fin de los experimentos de encoder de visión. **Mejor config: texto+visión, r=32, 4 épocas.**
- [x] Afinar hiperparámetros (r=32 fijo): **ningún cambio supera 74.60**. dropout 0.10 perjudica (70.6);
  lr=2e-4 gana a 2 épocas (73.5 vs 72.7) pero **pierde a 4 épocas (73.7 vs 74.6)** — el ranking se
  invierte, por eso se confirmó a 4 ép. **Óptimo: lr=1e-4, dropout=0.05, r=32, texto+visión, 4 épocas.**
- [x] **Avenida ADE + visión — Fase A** (ver §9): **las señales se suman.** ADE aporta +4.5 CIDEr (texto)
  sobre el modelo con visión — más que en solo-texto (+3.2). No es redundante. **Luz verde a la Fase B.**
- [x] **ADE + visión — Fase B** (train completo): **el delta se mantiene → CIDEr 78.68** (+4.08 sobre
  74.60 solo-visión). Tesis validada a escala. Mejor absoluto del proyecto (de pago en inferencia);
  mejor autónomo sigue siendo 74.60 (gratis).
- [x] **VLM alternativo — GIT-large** (`baseline_git.py`): en 20k GIT (69.26) **superaba** a BLIP (64.73),
  pero **no escaló**. A full train con r=32 matched: GIT-coco **73.91** vs BLIP solo-visión **74.60** →
  cuasi-empate, BLIP marginalmente adelante (−0.69). El confound de rango era parcial (r=16→32 dio +1.1)
  pero no flipea el resultado. **GIT es más *data-efficient* (gana con poco dato) pero no supera a BLIP a
  escala.** Podio sin cambios; BLIP-base fue elección razonable. Lección: un prototipo de bajo dato engaña
  — validar a escala Y con hiperparámetros pareados.
- [x] **VLM alternativo — Qwen2.5-VL-3B** (`baseline_qwen.py`, QLoRA 4-bit): viable en 8 GB (bitsandbytes
  funciona en Blackwell) pero **~14× más lento** (~1 par/s, 20k/2ep = ~11h). Trampa: el modelo bilingüe
  **fuga chino** en 26% de captions → hundía el CIDEr (56.4). Bloqueando tokens CJK en generación
  (`eval_qwen.py --block-cjk`) sube a **63.57** (global) / 64.90 (texto) / 55.79 (sin texto): mejor que
  BLIP en imágenes con texto, pero peor sin texto (captions verbosas). **No supera a BLIP (64.7) ni GIT
  (69.3) a 20k**, y es mucho más caro. Forzar inglés en el prompt de **entrenamiento** (re-run de 11h)
  NO lo arregla: empeora (49.36) y degenera en bucles de chino. Solo el bloqueo CJK en generación da
  salida limpia. Lección: el prior bilingüe no es suprimible por prompt/LoRA a esta escala.
- [x] **Conclusión exploración VLMs**: ningún alternativo (GIT, Qwen) destrona a BLIP para este task/8 GB.
- [ ] Augmentations que imiten blur/encuadre para las imágenes más difíciles.

---

## 9. Avenida ADE + visión (ejecutada — mejor resultado del proyecto)

**Pregunta abierta:** la Fase 3 mostró que el prefijo ADE aporta +3.2 CIDEr, pero sobre el setup
**débil** (20k, LoRA solo-texto, r=16). El escalado a 74.6 nunca recibió ADE. **¿Sigue aportando ADE
cuando el modelo ya tiene fine-tuning del encoder de visión, o el vision-scaling ya capturó su valor
(marcas/objetos), haciéndolo redundante?** Nunca se combinaron. Herramienta lista y verificada:
`baseline_blip_ocr.py` ahora acepta `--target-modules query,value,qkv` (LoRA texto+visión),
`--save-steps` (reanudable) y `--grad-checkpointing`.

### Fase A — EJECUTADA (2026-08-27): las señales se SUMAN 🟢

A/B sobre el subset de 20k (ADE ya cacheado → 0 créditos), ambos con LoRA texto+visión (r=32), 4 épocas:

| CIDEr (20k, texto+visión, 4 ép) | Global | Con texto | Sin texto | Δ texto vs control |
|---------------------------------|-------:|----------:|----------:|-------------------:|
| easyocr-visión (OCR ruidoso)    | 62.51  | 59.99     | 69.22     | −3.04 |
| control-visión (sin OCR)        | 64.73  | 63.03     | 68.59     | — |
| **ade-visión (Landing AI)**     | **68.86** | **67.55** | 71.38  | **+4.52** |

**Veredicto: la CALIDAD del OCR es decisiva.** Orden monótono en ambos regímenes (solo-texto y visión):
EasyOCR < control < Landing AI ADE. EasyOCR sigue perjudicando con visión (−3.04), aunque el daño se
**amortigua** vs solo-texto (−4.31 en Fase 2) — mejores features visuales ignoran mejor el ruido, pero
no lo vuelven útil. **ADE y el vision-scaling se COMPLEMENTAN, no son redundantes.** El delta de ADE en imágenes
con texto **creció** de +3.2 (Fase 3, solo-texto) a +4.52 (con visión) — el vision-scaling no capturó lo
de ADE. Bonus: hasta el subconjunto *sin texto* mejoró (+2.79, aunque ahí no se inyecta ADE), porque
entrenar con los prefijos mejoró los pesos compartidos. Reproducir: `run_ade_vision_phaseA.sh`.

**Luz verde a la Fase B**: si el delta (~+4) se mantiene al escalar, full+visión+ADE podría rondar
~78-79 CIDEr (vs 74.60 solo-visión).

### Fase B — EJECUTADA (2026-08-27): el delta se mantiene 🏆

Precómputo ADE del train completo (11 729 imágenes nuevas, 0 fallos, ~11.7k créditos) + entrenamiento
full+visión+ADE (r=32, 4 épocas). **CIDEr val: all 78.68 · con texto 78.36 · sin texto 77.14.**

| Modelo full-train (texto+visión, 4 ép) | CIDEr all | Con texto | Δ vs control | Inferencia |
|----------------------------------------|----------:|----------:|-------------:|------------|
| EasyOCR + visión (OCR ruidoso)         | 70.33     | 68.35     | −4.27        | gratis |
| Solo-visión (mejor autónomo)           | 74.60     | —         | —            | gratis |
| **Visión + ADE (mejor absoluto)**      | **78.68** | **78.36** | +4.08        | ADE/img (pago) |

**Simetría casi perfecta:** OCR malo resta ~−4.3, OCR bueno suma ~+4.1 — espejo alrededor del control.
El orden EasyOCR < control < ADE se replica en **ambos regímenes** (20k y full). Matiz que emerge a
escala: el daño de EasyOCR **crece** (−2.2 en 20k → −4.3 en full) porque el prefijo ruidoso limita
cuánto exprime el modelo de más datos (control gana +9.9 de 20k→full; EasyOCR solo +7.8).

**Tesis validada a escala: +4.08 CIDEr sobre solo-visión**, dentro del ~78-79 proyectado en Fase A. ADE
y el fine-tuning de visión se complementan también con train completo. **Matiz de despliegue:** este
modelo requiere llamar a Landing AI por imagen en inferencia (coste + latencia + dependencia). Dos
"mejores" según criterio: **absoluto 78.68** (de pago) vs **autónomo 74.60** (gratis, local).

### Fase B — escalar ADE al train completo (requiere créditos) — LISTA, sin ejecutar

Fase A dio luz verde. La Fase B está **turnkey** en `run_ade_vision_phaseB.sh` (precómputo full +
entrenamiento full+visión+ADE). Decisión del usuario (2026-08-27): **esperar a tener créditos para el
full** antes de ejecutar.

```bash
wsl -e bash /mnt/c/Users/green/Documents/GitHub/wizviz-caption/run_ade_vision_phaseB.sh
```

**Cuenta de créditos** (≈1 crédito/imagen-con-texto): el train completo tiene ~14 700 imágenes con
texto; ya hay ~2 900 cacheadas → faltan **~11 800**. Con ~5 500 disponibles **no alcanza para el full**
→ hace falta comprar ~6 500 más. (El precómputo es reanudable, guarda cada 50 imágenes.)

**Control de comparación** = el modelo solo-visión de 74.60 (`~/wizviz/runs/blip_scaled4`); para su
desglose por texto/sin-texto, re-evaluarlo con `baseline_blip_ocr.py --no-train`. **Objetivo:** si el
delta (~+4.5 en texto) se mantiene, full+visión+ADE ≈ **78-79 CIDEr**. Matiz: la variante ganadora
**seguiría pagando ADE en inferencia** (trade-off de despliegue de la Fase 3).

---

## 10. Referencias

- D. Gurari, Y. Zhao, M. Zhang, N. Bhattacharya. *Captioning Images Taken by People Who Are Blind.*
  ECCV 2020. (`2002.08565v2.pdf`) · <https://vizwiz.org>
- R. N. Simons, D. Gurari, K. R. Fleischmann. *"I Hope This Is Helpful": Understanding Crowdworkers'
  Challenges and Motivations for an Image Description Task.* CSCW 2020. (`2020_CSCW_HopeThisHelps.pdf`)
- Licencia del dataset: CC BY 4.0.
