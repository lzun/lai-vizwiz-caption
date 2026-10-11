# VizWiz-Captions — Experimentos de Image Captioning

Bitácora del proyecto: dataset, entorno, fine-tuning (BLIP + LoRA), OCR, escalado y
alternativas de VLM. Última actualización: **2026-10-10** (evaluación sobre val completo, 7 542).

> **TL;DR** — Image captioning sobre VizWiz-Captions, todo con LoRA en una laptop RTX 5060
> (8 GB) **dentro de WSL2**. Partiendo de BLIP-base (CIDEr 47.3 sin adaptar). Métricas sobre el
> **conjunto de validación completo (7 542 imágenes)**:
> - 🥇 **Mejor absoluto: CIDEr 94.8** (val) — BLIP + fine-tuning de visión + prefijo de descripción
>   de Landing AI ADE. Requiere pagar ADE por imagen en inferencia.
> - 🥈 **Mejor autónomo: CIDEr 86.9** (val) — BLIP + fine-tuning de visión (LoRA texto+visión, r=32),
>   gratis y local.
>
> **Hallazgos clave:** la palanca decisiva es adaptar el **encoder de visión** (+15.5); la **calidad
> del OCR es decisiva** (EasyOCR resta, ADE suma +7.8); ADE y visión se **complementan** (no
> son redundantes); en VizWiz gana la **descripción**, no la transcripción literal. GIT-large como
> backbone alternativo queda por debajo de BLIP a escala. Ver §5–§9.
> *(Nota: CIDEr no es comparable entre conjuntos de evaluación; todas las cifras son sobre val completo.)*

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
├── baseline_llava.py             prototipo LLaVA-1.5-7B (QLoRA 4-bit)
├── eval_llava_extra.py           CIDEr + CLIP/BERT/SBERT de LLaVA sobre 2000 val
├── compare_models.py             galería HTML comparando captions por modelo
├── eval_metrics_extra.py         CLIPScore + BERTScore + SBERT por modelo
├── reporte_baseline.html         reporte visual de resultados
├── space/                        servicio web Gradio (prototipo para HF Spaces, §10)
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

Evaluación sobre el **conjunto de validación completo (7 542 imágenes)**, beam = 3.

| Modelo | CIDEr | BLEU-4 | BLEU-1 | ROUGE-L |
|--------|------:|-------:|-------:|--------:|
| BLIP base (sin fine-tuning) | 47.27 | 19.54 | 61.29 | 42.75 |
| LoRA · 20k pares · 2 épocas | 61.91 | 22.08 | 64.47 | 45.93 |
| LoRA texto · train completo · 3 épocas | 69.10 | 24.20 | 66.40 | 47.28 |
| LoRA texto+visión (r=32) · full · 2 épocas | 84.63 | 27.60 | 69.60 | 49.64 |
| **LoRA texto+visión (r=32) · full · 4 épocas** | **86.93** | **27.93** | **70.15** | **49.83** |

- El fine-tuning aporta **+14.6 CIDEr** sobre el base; escalar de 20k → 100k pares suma otros +7.2.
- **Escalar al encoder de visión es el mayor salto: +15.5 CIDEr** (69.1 → 84.6) al aplicar LoRA
  también al ViT (`qkv`) con r=32. Adaptar las *features visuales* al dominio (fotos borrosas,
  ángulos raros) pesa mucho más que solo adaptar el decoder de texto.
- Entrenamiento texto full: 3 épocas en ~76 min. Escalado (texto+visión): 2 épocas en **~104 min**
  (32 pares/s, VRAM 7.0/8.1 GB sin OOM, batch 12, grad-checkpointing no-reentrante disponible).
- **86.93 es el mejor modelo AUTÓNOMO** (gratis en inferencia). Sumar el prefijo de descripción de
  Landing AI ADE lo lleva a **94.77** (mejor absoluto, pero de pago en inferencia) — ver §6.
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
sobre el **mismo subset de 20k pares / 2 épocas** (`baseline_blip_ocr.py`), evaluando en **val
completo (7 542)** con métricas **desglosadas por `text_detected`**.

| Subconjunto (CIDEr) | Control (sin OCR) | Con OCR (EasyOCR) | Δ |
|---------------------|------------------:|------------------:|----:|
| Todas (n=7542)      | 61.87 | 60.67 | **−1.20** |
| **Con texto** (n=5010) | 61.63 | 60.02 | **−1.61** |
| Sin texto (n=2532)   | 59.60 | 59.26 | −0.34 |

**Conclusión: la inyección naíf de OCR con EasyOCR NO ayuda — empeora, y justo en las imágenes
con texto.** El experimento es fiable: el subconjunto *sin texto* (donde casi no se inyecta OCR) queda
prácticamente igual, así que el daño viene del **contenido ruidoso** del OCR, no de la mecánica. EasyOCR es
demasiado ruidoso en estas fotos degradadas (ver Fase 1). Hay ganancias puntuales (p. ej. una
botella pasó de *"bottle of cough"* a *"bottle of medicine"*), pero el ruido domina en promedio.

### Fase 3 — Prefijo con Landing AI ADE (dos variantes)

Se repitió el A/B usando **Landing AI ADE** como fuente del prefijo, precomputado en concurrencia
(`ocr_landingai_precompute.py`, solo imágenes con `text_detected`). Dos limpiezas del texto
(`--ocr-clean`): **solo-texto** (transcripción literal, se quitan los bloques `<description>`) y
**ADE completo** (descripción de escena + texto). Mismo subset de 20k / 2 épocas, eval en val completo.

| Variante (CIDEr) | Global | Con texto (n=5010) | Sin texto (n=2532) |
|------------------|-------:|-------------------:|------------------:|
| Control (sin OCR)            | 61.87 | 61.63 | 59.60 |
| EasyOCR                      | 60.67 | 60.02 | 59.26 |
| Landing AI — solo texto      | 61.73 | 61.70 | 59.46 |
| **Landing AI — ADE completo**| **69.99** | **73.96** | 59.56 |

**Progresión monótona y clara:** EasyOCR (ruidoso) < LA solo-texto (limpio pero literal) < LA ADE
completo — y **solo la última supera al control** (+8.1 global, **+12.3 en imágenes con texto**).

**Interpretación honesta:** lo que ayuda **no es "leer el texto"** (el solo-texto literal sigue por
debajo del control), sino la **descripción de escena** de ADE, que actúa como señal de un captioner
externo fuerte — es esencialmente **destilación de un VLM**, no OCR. Coste de despliegue: la variante
ganadora requiere una llamada a Landing AI (~1 crédito/img) también en inferencia.

**Conclusión de las 3 fases:** el OCR *literal* (barato o caro) no mejora el captioning en VizWiz
—las captions humanas describen escenas, no transcriben—; pero inyectar la **descripción
multimodal** de una herramienta fuerte sí ayuda (+12.3 CIDEr en imágenes con texto).

### Fase 4 — Destilación desde ADE (resultado negativo)

Como la ganancia de la Fase 3 exige llamar a Landing AI en inferencia, se intentó **destilar**:
usar la primera oración de la descripción de ADE como **pseudo-caption extra** en entrenamiento
(`baseline_blip_distill.py`), sin prefijo en inferencia → modelo autónomo y gratis. Solo se usaron
1 981 descripciones limpias (filtradas de transcripciones ruidosas). A/B vs control, mismo subset.

| Modelo (CIDEr) | Global | Con texto | Sin texto | Dep. inferencia |
|----------------|-------:|----------:|----------:|-----------------|
| Control                  | 61.52 | 61.27 | 59.29 | ninguna |
| Fase 3 (ADE prefijo)     | 69.99 | 73.96 | 59.56 | Landing AI/img |
| **Destilado (Fase 4)**   | **35.96** | 31.33 | 43.49 | ninguna |

**Empeoró mucho (−25.6 CIDEr).** La destilación **contamina el estilo**: BLIP aprende la prosa
verbosa de ADE (*"a close-up, slightly blurry photograph of a red, white and blue tube of
toothpaste…"*) y la aplica en todas partes, incluso en imágenes sin texto (59.3→43.5, donde no
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
| Control                  | 61.52 | 61.27 | 59.29 | ninguna |
| Fase 3 (ADE prefijo)     | **69.99** | **73.96** | 59.56 | Landing AI/img 💰 |
| Fase 4 (prosa ADE)       | 35.96 | 31.33 | 43.49 | ninguna |
| **Fase 4b (profesor)**   | 60.92 | 60.54 | 58.92 | ninguna |

**El estilo era el problema** (35.96 → 60.92; el subconjunto sin texto se recupera del todo). **Pero
la destilación no aporta beneficio neto:** la 4b queda a la par del control (−0.6, dentro del ruido)
y por debajo de la Fase 3. Se **neutraliza el daño, pero no se captura la ganancia.**

**Conclusión del hilo de OCR (4 fases):** la ganancia de la descripción de ADE (+12.3 CIDEr en
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
- [x] **OCR — Fase 2**: prefijo OCR (EasyOCR) en BLIP → **resultado negativo** (CIDEr −1.20, −1.61 en imágenes con texto). El OCR ruidoso perjudica.
- [x] **OCR — Fase 3**: prefijo Landing AI ADE. Solo-texto sigue bajo el control; **ADE completo (descripción) SÍ mejora** (+8.1 global, **+12.3 en imágenes con texto**). Ayuda la descripción del VLM, no el OCR literal.
- [x] **OCR — Fase 4**: destilación (pseudo-captions de ADE) → **negativo** (CIDEr 62→36): contamina el estilo hacia la prosa verbosa de ADE. El conocimiento se transfiere pero el estilo arruina la métrica.
- [x] **OCR — Fase 4b**: destilación estilo-matched (profesor = modelo Fase 3). Arregla el colapso de la Fase 4 (36→61) pero **queda a la par del control** — neutraliza el daño, no captura la ganancia. La ventaja de ADE solo es realizable pagándolo en inferencia (Fase 3).
- [x] **Test set**: NO evaluable localmente. Las captions de test son oficialmente **ocultas** (VizWiz publica solo train/val). El CSV tiene captions TEST pero con nombres originales que **no mapean** a los archivos renombrados (se intentó un offset `20000+i` que falla; `test.json` no tiene captions con qué anclar por contenido). **Val es el conjunto de evaluación válido.**
- [x] Escalar el fine-tuning: LoRA texto+visión (r=32) → **CIDEr 69.1 → 84.6 (+15.5)**. El encoder de
  visión era el eje clave (`--target-modules query,value,qkv --lora-r 32 --save-steps 800`, reanudable).
- [x] Escalar más: 2→4 épocas → CIDEr 84.6 → **86.9 (+2.3)**. Rendimientos decrecientes sin overfitting;
  la palanca era la visión, no las épocas. (Una corrida crasheó con `CUDA error: unknown error` y se
  **reanudó desde checkpoint** sin re-entrenar — el diseño reanudable funcionó.)
- [x] Escalar r=64 sobre texto+visión → **no ayuda, saturado**; r=32 es el punto dulce. Fin de los
  experimentos de encoder de visión. **Mejor config: texto+visión, r=32, 4 épocas.** *(ablations de rango/lr/dropout evaluados en el subset de 2000; el ranking no depende del conjunto.)*
- [x] Afinar hiperparámetros (r=32 fijo): **ningún cambio supera la config base**. dropout 0.10 perjudica;
  lr=2e-4 gana a 2 épocas pero **pierde a 4 épocas** — el ranking se invierte, por eso se confirmó a 4 ép.
  **Óptimo: lr=1e-4, dropout=0.05, r=32, texto+visión, 4 épocas.**
- [x] **Avenida ADE + visión — Fase A** (ver §9): **las señales se suman.** ADE aporta +10.7 CIDEr (texto)
  sobre el modelo con visión — comparable a su efecto en solo-texto (+12.3). No es redundante. **Luz verde a la Fase B.**
- [x] **ADE + visión — Fase B** (train completo): **el delta se mantiene → CIDEr 94.77** (+7.84 global,
  +11.63 en texto, sobre 86.93 solo-visión). Tesis validada a escala. Mejor absoluto del proyecto (de pago
  en inferencia); mejor autónomo sigue siendo 86.93 (gratis).
- [x] **VLM alternativo — GIT-large** (`baseline_git.py`): en el prototipo de 20k GIT **superaba** a BLIP,
  pero **no escaló**. A full train con r=32 matched (eval val completo): GIT-coco **84.54** vs BLIP
  solo-visión **86.93** → BLIP adelante (−2.39). **GIT es más *data-efficient* (gana con poco dato) pero no
  supera a BLIP a escala.** Podio sin cambios; BLIP-base fue elección razonable. Lección: un prototipo de
  bajo dato engaña — validar a escala Y con hiperparámetros pareados.
- [x] **VLM alternativo — Qwen2.5-VL-3B** (`baseline_qwen.py`, QLoRA 4-bit): viable en 8 GB (bitsandbytes
  funciona en Blackwell) pero **~14× más lento** (~1 par/s, 20k/2ep = ~11h). Trampa: el modelo bilingüe
  **fuga chino** en 26% de captions → hundía el CIDEr. Bloqueando tokens CJK en generación
  (`eval_qwen.py --block-cjk`) da salida limpia; en **val completo** llega a **70.54** (global) / 77.16
  (texto) / 54.13 (sin texto). **No supera a BLIP (86.9) ni GIT (84.5)**, y es mucho más caro. Forzar
  inglés en el prompt de **entrenamiento** (re-run de 11h)
  NO lo arregla: empeora (49.36) y degenera en bucles de chino. Solo el bloqueo CJK en generación da
  salida limpia. Lección: el prior bilingüe no es suprimible por prompt/LoRA a esta escala.
- [x] **VLM alternativo — Florence-2-large**: intentado, **bloqueado**. Su código remoto (`trust_remote_code`)
  es incompatible con transformers 5.15 (escrito para ~4.4x): tras 4 workarounds (parches config/processor,
  einops+timm, eager attn) el checkpoint carga mal (`decoder.embed_tokens` MISSING) y exige imágenes cuadradas.
  Requeriría un venv aislado con transformers antiguo. Exploración cerrada aquí.
- [x] **VLM alternativo — LLaVA-1.5-7B** (`baseline_llava.py`, QLoRA 4-bit nf4): **cabe en 8 GB** (smoke
  confirmó antes de comprometer el run largo) y completó 20k/2ep en **~21 h sin crashes**. English-only,
  **cero fuga de idioma** (a diferencia de Qwen). **Lee el texto mejor que ningún otro modelo** (transcribe
  "50 mg", "100 tablets", fechas de vencimiento). En crudo es **extremadamente verboso** (23.7 palabras) y
  tras una primera oración impecable **degenera en repetición patológica** ("…medicine. 50 mg. 100 tablets.
  100% natural…"), lo que hunde TODAS las métricas: CIDEr **15.20**, BERTScore **31.21**, SBERT **60.89**.
  **Pero truncar al primer enunciado** (`first_sentence()` en `eval_llava_extra.py`, post-proceso trivial y
  determinista) **lo transforma en un modelo de primer nivel en calidad**: 10.6 palabras/caption, CIDEr
  **90.15** — el **mejor CIDEr autónomo** del proyecto (supera a solo-visión 86.93, ~4.6 pts por debajo del ADE de pago 94.77) — BERTScore
  **56.10** (#1 de todos), SBERT **69.76** (#1 de todos). Las 4 métricas se computan sobre **val completo
  (7542)**; captions crudas cacheadas en `llava_caps.json`/`fullval_caps/`.
  **Hallazgo clave (el más fuerte sobre evaluación):** el CIDEr 15.2 no medía mala comprensión sino un
  *defecto de formato podable en una línea*; ninguna métrica lo detectó, solo mirar las captions y truncar.
  Capacidad ≠ presentación, y las métricas agregadas los confunden. Caveats honestos: LLaVA es 7B (~30× BLIP)
  y ~2 s/imagen; LLaVA usó greedy vs beam=3 de BLIP/GIT (decoding no 100% pareado).
- [x] **Conclusión exploración VLMs (cerrada)**: en **eficiencia**, BLIP sigue siendo la elección correcta
  (225M, casi instantáneo; ningún VLM lo destrona a coste comparable). En **calidad absoluta**, el podio
  **sí cambia**: **LLaVA-1.5-7B truncado al primer enunciado da el mejor CIDEr autónomo (90.15, > 86.93
  solo-visión) y es #1 en BERTScore/SBERT** de todos los modelos — aunque el mejor CIDEr absoluto sigue
  siendo el ADE de pago (94.77). Todo al coste de 30× params y ~2 s/imagen. GIT y Qwen no superan a BLIP; Florence-2 quedó
  bloqueado. Patrón: los VLM-LLM grandes traen un prior verboso/bilingüe difícil de alinear vía LoRA — pero
  con LLaVA un simple post-proceso (truncar) separó un modelo excelente de sus métricas hundidas.
- [x] **Métricas complementarias** (`eval_metrics_extra.py`): CLIPScore + BERTScore + SBERT sobre 2000 val.
  Hallazgos: **CLIPScore inútil** aquí (plano ~0.73, CLIP no lee texto); **CIDEr y BERTScore se contradicen**
  (ADE alto-CIDEr/bajo-BERTScore; Qwen bajo-CIDEr/alto-BERTScore) → cada métrica premia algo distinto
  (texto exacto vs fluidez semántica); la utilidad de ADE es **específica** (contenido de texto), CIDEr era
  correcta para ella; las semánticas **rehabilitan a Qwen** (subestimado por CIDEr). Con **LLaVA** ninguna
  métrica bastó: su CIDEr 14.2 escondía un defecto de formato (verbosidad+repetición) que solo *mirar y
  truncar* reveló → truncado es el #1 en las 3 métricas. Ver reporte §7.
- [ ] Métricas de alucinación (CHAIR/POPE): necesitan anotaciones de objetos o un detector open-vocab.
- [ ] Augmentations que imiten blur/encuadre para las imágenes más difíciles.

---

## 9. Avenida ADE + visión (ejecutada — mejor resultado del proyecto)

**Pregunta abierta:** la Fase 3 mostró que el prefijo ADE aporta +12.3 CIDEr (texto), pero sobre el setup
**débil** (20k, LoRA solo-texto, r=16). El escalado a 86.9 nunca recibió ADE. **¿Sigue aportando ADE
cuando el modelo ya tiene fine-tuning del encoder de visión, o el vision-scaling ya capturó su valor
(marcas/objetos), haciéndolo redundante?** Nunca se combinaron. Herramienta lista y verificada:
`baseline_blip_ocr.py` ahora acepta `--target-modules query,value,qkv` (LoRA texto+visión),
`--save-steps` (reanudable) y `--grad-checkpointing`.

### Fase A — EJECUTADA (2026-08-27): las señales se SUMAN 🟢

A/B sobre el subset de 20k (ADE ya cacheado → 0 créditos), ambos con LoRA texto+visión (r=32), 4 épocas:

| CIDEr (20k, texto+visión, 4 ép; eval val completo) | Global | Con texto | Sin texto | Δ texto vs control |
|---------------------------------|-------:|----------:|----------:|-------------------:|
| easyocr-visión (OCR ruidoso)    | 75.28  | 77.64     | 67.63     | −1.72 |
| control-visión (sin OCR)        | 76.06  | 79.36     | 66.69     | — |
| **ade-visión (Landing AI)**     | **83.41** | **90.07** | 67.17  | **+10.71** |

**Veredicto: la CALIDAD del OCR es decisiva.** El orden es monótono y claro en todos los regímenes:
EasyOCR < control < Landing AI ADE. EasyOCR con visión sigue restando en texto (−1.72), aunque menos que
en solo-texto — mejores features visuales amortiguan el ruido pero no lo vuelven útil. **ADE y el
vision-scaling se COMPLEMENTAN, no son redundantes:** el delta de ADE en texto es **+10.7 con visión**,
comparable a los **+12.3 en solo-texto** (Fase 3) — el vision-scaling no capturó el valor de ADE. Bonus:
hasta el subconjunto *sin texto* sube un poco (+0.5, aunque ahí casi no se inyecta ADE). Reproducir:
`run_ade_vision_phaseA.sh`.

**Luz verde a la Fase B**: si el delta en texto se mantiene al escalar, full+visión+ADE debería superar
holgadamente al solo-visión (86.9).

### Fase B — EJECUTADA (2026-08-27): el delta se mantiene 🏆

Precómputo ADE del train completo (11 729 imágenes nuevas, 0 fallos, ~11.7k créditos) + entrenamiento
full+visión+ADE (r=32, 4 épocas). **CIDEr val completo: all 94.77 · con texto 103.38 · sin texto 74.48.**

| Modelo full-train (texto+visión, 4 ép) | CIDEr all | Con texto | Δ vs s/visión | Inferencia |
|----------------------------------------|----------:|----------:|-------------:|------------|
| EasyOCR + visión (OCR ruidoso)         | 85.10     | 89.33     | −1.83        | gratis |
| Solo-visión (mejor autónomo)           | 86.93     | 91.75     | —            | gratis |
| **Visión + ADE (mejor absoluto)**      | **94.77** | **103.38** | +7.84        | ADE/img (pago) |

**Asimetría reveladora:** el OCR malo resta poco (~−1.8), el OCR bueno suma mucho (~+7.8) — no es un
espejo: ADE aporta bastante más de lo que EasyOCR daña. El orden EasyOCR < control < ADE se replica en
**ambos regímenes** (20k y full), y el daño de EasyOCR **crece** al escalar (el prefijo ruidoso limita
cuánto exprime el modelo de más datos).

**Tesis validada a escala: +7.84 CIDEr global (+11.63 en texto) sobre solo-visión.** ADE
y el fine-tuning de visión se complementan también con train completo. **Matiz de despliegue:** este
modelo requiere llamar a Landing AI por imagen en inferencia (coste + latencia + dependencia). Dos
"mejores" según criterio: **absoluto 94.77** (de pago) vs **autónomo 86.93** (gratis, local).

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

**Control de comparación** = el modelo solo-visión de 86.93 (`~/wizviz/runs/blip_scaled4`); para su
desglose por texto/sin-texto, re-evaluarlo con `baseline_blip_ocr.py --no-train`. **Objetivo:** si el
delta (~+4.5 en texto) se mantiene, full+visión+ADE ≈ **78-79 CIDEr**. Matiz: la variante ganadora
**seguiría pagando ADE en inferencia** (trade-off de despliegue de la Fase 3).

---

## 10. Servicio web (prototipo, 2026-10-06)

`space/` empaqueta el **mejor modelo autónomo** (`blip_scaled4`, CIDEr 86.9) como servicio web con
Gradio: imagen → descripción en inglés, con UI y API HTTP (`/describe`). Listo para Hugging Face
Spaces (`space/README.md` lleva el front-matter del Space). Ejecución, API y despliegue en
[`space/README.md`](space/README.md).

- Local: `wsl -e bash /mnt/c/Users/green/Documents/GitHub/wizviz-caption/space/run_local.sh` → <http://localhost:7860>.
- Latencia medida: ~0.5 s/imagen en la RTX 5060, ~1.5–2 s en CPU.
- En HF, las cuentas gratuitas ya no pueden alojar Gradio en `cpu-basic` (402, requiere PRO);
  la opción gratuita es **ZeroGPU**, y `app.py` ya es compatible (`@spaces.GPU`, torch 2.11.0).
- Modo opcional **BLIP + ADE** (CIDEr 94.8): solo se habilita si hay `VISION_AGENT_API_KEY`;
  ~7–10 s y ~1 crédito por imagen. En un Space público no conviene poner la clave.
- Los adaptadores (`space/adapters/`) no se versionan en git; se copian desde `~/wizviz/runs/`.

**Bug encontrado al construirlo (corregido y re-evaluado, 2026-10-06):**
`BlipForConditionalGeneration.generate` descarta el **último token** de `input_ids` (asume que termina
en `[SEP]`). En `baseline_blip_ocr.py` (`evaluate`) el prompt `[CLS] + prefijo` se pasaba sin `[SEP]`,
así que **se perdía el último token del prefijo y el recorte `1 + len(ocr_ids)` se comía la primera
palabra generada** (típicamente "a"). **Fix:** añadir un `[SEP]` de relleno al prompt (aplicado en
`baseline_blip_ocr.py` y `compare_models.py`; verificado en `verify_blip_prefix.py`). Todas las cifras
con prefijo OCR/ADE se **re-evaluaron** con el caché existente (sin re-entrenar) vía `reeval_ocr.py`.

**Segundo cambio metodológico (2026-10-10): evaluación sobre val COMPLETO (7 542).** Todas las corridas
previas medían sobre las *primeras 2 000* imágenes de val (el default `--max-eval-samples 2000`). Dos
hallazgos al pasar a val completo: (i) **CIDEr no es comparable entre conjuntos** —su IDF depende del
corpus—, por eso los absolutos suben (p. ej. solo-visión 74.6→**86.9**); (ii) la caché de ADE solo cubría
las ~1 536 imágenes de val de las primeras 2 000, así que una eval ingenua sobre val completo corría los
modelos ADE **sin prefijo** en la mayoría de imágenes con texto. Tras recargar créditos y **completar ADE
al 98.9 %** de las imágenes con texto (`reeval_full.py`, `vlm_full.py`, `fullval_extra.py`), la comparación
justa confirma y **refuerza la tesis**: visión+ADE **94.77** vs solo-visión **86.93** (+7.84 global,
**+11.63 en texto**). El subconjunto *sin texto* queda ~idéntico (sin prefijo), confirmando la corrección.

---

## 11. Referencias

- D. Gurari, Y. Zhao, M. Zhang, N. Bhattacharya. *Captioning Images Taken by People Who Are Blind.*
  ECCV 2020. (`2002.08565v2.pdf`) · <https://vizwiz.org>
- R. N. Simons, D. Gurari, K. R. Fleischmann. *"I Hope This Is Helpful": Understanding Crowdworkers'
  Challenges and Motivations for an Image Description Task.* CSCW 2020. (`2020_CSCW_HopeThisHelps.pdf`)
- Licencia del dataset: CC BY 4.0.
