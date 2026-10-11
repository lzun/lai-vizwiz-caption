# 4. Experimental Setup

## 4.1. Dataset and Preprocessing

We conduct all experiments on **VizWiz-Captions v1.0** [Gurari et al., ECCV 2020],
a captioning dataset of photographs taken by people who are blind, with five
crowdsourced captions per image. Unlike curated benchmarks such as MS-COCO,
VizWiz images exhibit genuine real-world quality issues — motion blur, poor
illumination, and partial framing — and a high incidence of embedded text
(product labels, screens, packaging): approximately **63%** of images carry a
`text_detected` flag, which makes the dataset a natural testbed for studying the
interaction between captioning and optical character recognition (OCR).

The dataset is distributed in a COCO-style format with `images` and
`annotations` records. Because the official `test` split is released without
captions (it is the hidden hold-out of the public challenge), **all evaluation in
this work is performed on the `val` split**, which therefore serves as our
held-out set; `train` is used exclusively for fine-tuning.

We apply a cleaning pass before training to remove low-quality supervision. Two
flags in the annotations are used to discard captions: `is_precanned`, which
marks a fixed placeholder string ("Quality issues are too severe to recognize
visual content.") emitted when an image is unrecognizable, and `is_rejected`,
which marks captions flagged by the dataset's quality-assurance process. Images
left with no valid captions after filtering are dropped. Table 1 summarizes the
resulting counts.

**Table 1. VizWiz-Captions splits after cleaning.**

| Split | Images (JSON) | Raw captions | Clean captions | Images used |
|-------|--------------:|-------------:|---------------:|------------:|
| train | 23,431        | 117,155      | 100,575        | 22,866      |
| val   | 7,750         | 38,750       | 33,145         | 7,542       |
| test  | 8,000         | — (no captions) | —           | 8,000       |

The loader iterates over the `file_name` entries declared in each JSON file
rather than scanning the image directory on disk; this automatically ignores 523
extra images present in `train/` that are absent from `train.json`. The raw
crowdsourcing metadata CSV is read with `latin-1` encoding. For training we use
an *exploded* view with one (image, caption) pair per row, and for analysis a
*grouped* view with one row per image.

## 4.2. Model and Parameter-Efficient Fine-tuning

Our primary captioner is **BLIP-base**
(`Salesforce/blip-image-captioning-base`, 225M parameters), fine-tuned in half
precision (fp16). To keep training feasible on a single consumer GPU, we adopt
**Low-Rank Adaptation (LoRA)** rather than full fine-tuning.

The baseline configuration applies LoRA with rank *r* = 16 and α = 32 to the
`query` and `value` projections of the attention layers, yielding **1.18M
trainable parameters (0.52% of the model)**. A central axis of our study is
*where* adaptation is applied: our best configuration additionally adapts the
**vision encoder** by extending the LoRA targets to the ViT attention
projections (`qkv`) and raising the rank to *r* = 32 (target modules
`query, value, qkv`). We refer to these two regimes as **text-only LoRA** and
**text+vision LoRA** throughout.

## 4.3. Training Configuration and Hardware

All models are trained in WSL2 on a single **NVIDIA GeForce RTX 5060 Laptop GPU
(8 GB, Blackwell / sm_120)**, using PyTorch 2.11 with the CUDA 12.8 build (the
Blackwell architecture requires `cu128` wheels) under a dedicated Python 3.12
environment. The 8 GB memory budget is the binding constraint on all design
choices.

Unless otherwise noted, we train with learning rate 1e-4, LoRA dropout 0.05, and
beam search with beam width 3 at inference. The text-only regime fits batch size
16 in fp16 without out-of-memory errors; the text+vision regime uses batch size
12 with non-reentrant gradient checkpointing (`use_reentrant=False`), required
because the in-place embedding update in BLIP conflicts with the default
reentrant path. Representative wall-clock costs on the RTX 5060 are ~76 minutes
for 3 epochs of full-train text-only fine-tuning and ~104 minutes for 2 epochs of
text+vision fine-tuning (≈32 pairs/s, 7.0–8.1 GB VRAM). Long precomputation and
training jobs are made **resumable** (checkpoints saved periodically) so they can
recover from WSL interruptions; one run that crashed with a transient CUDA error
was resumed from checkpoint without retraining.

## 4.4. Evaluation Protocol

We report **CIDEr, BLEU-4, BLEU-1, and ROUGE-L**, computed with
`pycocoevalcap`. To remove the external Java dependency of the standard
PTBTokenizer, we substitute a lightweight in-house tokenizer (lowercasing and
punctuation stripping); the same tokenization is applied to candidates and
references across all conditions, so comparisons remain internally consistent.
Unless stated otherwise, metrics are computed on a fixed sample of **2,000 `val`
images** with beam width 3. Our headline metric is CIDEr, which is the standard
primary metric on VizWiz-Captions.

A key methodological feature of our evaluation is **stratification by
`text_detected`**: for every OCR-related experiment we report CIDEr separately
on the *with-text* and *no-text* subsets of `val`, in addition to the global
score. Because OCR signal is injected only for with-text images, the no-text
subset acts as an internal control — if it moves, the effect cannot be attributed
to OCR content alone.

## 4.5. Experimental Design

Our experiments are organized as **controlled A/B comparisons** in which a single
factor is varied while every other hyperparameter — training subset, number of
epochs, rank, learning rate, dropout, gradient accumulation, and beam width — is
held fixed. We adopt this discipline deliberately: in preliminary runs an
unintended default `--grad-accum` and a mismatched `--lora-r` each introduced
confounds that would have biased cross-model conclusions. Many comparisons are
run on a fixed **20k-pair / 2-epoch** subset for fast, cheap iteration, with the
decisive findings re-validated at **full-train** scale.

**(a) Where to adapt: scaling the LoRA target.** We first study the effect of
scaling fine-tuning along two axes — data (20k pairs → full train) and adapted
modules (text-only → text+vision). This isolates the contribution of adapting the
vision encoder versus the text decoder, and establishes the strongest *autonomous*
model (no inference-time external dependency) used as the control in subsequent
OCR experiments.

**(b) OCR integration (Phases 1–4b).** We then investigate whether OCR signal
improves captioning on this text-heavy domain. Because VizWiz-Captions provides
no OCR ground truth, Phase 1 is a **qualitative** comparison of two OCR engines —
**EasyOCR** (a traditional GPU recognizer) and **Landing AI ADE** (an agentic
document-understanding service) — over 30 with-text `val` images, characterizing
their robustness and output style. Phases 2–3 inject the extracted text as a
**conditional decoder prefix**: the prefix is masked out of the training loss and
stripped from the generated output at inference, and `min_new_tokens` is enforced
to prevent a noisy prefix from collapsing generation to an empty string. Phase 2
uses EasyOCR; Phase 3 uses Landing AI ADE under two text-cleaning variants —
*text-only* (literal transcription) and *full ADE* (scene description plus text).
All prefix experiments are run as matched A/B trials on the 20k/2-epoch subset
with the `text_detected` stratification of §4.4.

Phases 4 and 4b test whether the Phase-3 gain can be internalized into an
autonomous model via **distillation**, removing the inference-time dependency on
ADE. Phase 4 uses ADE descriptions directly as extra pseudo-caption targets;
Phase 4b corrects for caption-style mismatch by using as teacher the Phase-3
model itself (which already produces short, VizWiz-style captions), training the
student on the teacher's outputs without any prefix at inference. Both are A/B
against the same control, varying only the supervision target.

**(c) Combining ADE with vision adaptation.** The prefix gains of Phase 3 were
measured on the *weak* backbone (20k, text-only, *r* = 16), so it remains open
whether ADE still contributes once the vision encoder has been adapted, or whether
vision scaling already captures the same object/brand information. We test this
directly with a two-stage A/B (Phase A on the 20k subset, Phase B at full-train
scale), each comparing control-vision, EasyOCR-vision, and ADE-vision under an
identical text+vision (*r* = 32, 4-epoch) configuration. The ADE prefixes are
precomputed for all with-text images with a resumable concurrent client.

**(d) Alternative vision-language backbones.** Finally, to verify that BLIP-base
is a reasonable backbone choice, we benchmark three alternatives under
hardware-matched settings: **GIT-large**, evaluated both at 20k and at full-train
scale with a rank-matched (*r* = 32) configuration to avoid a data-efficiency
confound; **Qwen2.5-VL-3B** via 4-bit QLoRA, including an inference-time CJK-token
block to neutralize bilingual-leakage into non-English captions; and
**Florence-2-large**, whose custom remote code proved incompatible with the
installed Transformers release and is reported as a blocked exploration. All
backbones are compared on the same `val` evaluation protocol of §4.4.
