# State-of-the-Art Image Classification Pipeline for the BDC 2026 Waste Classification Challenge

## TL;DR
- **The maximum-score configuration is a 2-model out-of-fold ensemble of one modern CNN (ConvNeXt V2) and one transformer (EVA-02 or a ViT/DeiT III), each trained with 5-fold stratified cross-validation, AdamW, cosine schedule with warmup, label smoothing, and EMA, then combined by probability averaging with flip-only test-time augmentation and per-class logit-offset tuning on out-of-fold predictions.** This stays inside the 3-submission limit and is optimized for Macro F1.
- **Run the EDA and image-size diagnostic before choosing anything else.** The resolution decision drives the whole pipeline. If images are small (median under about 128 px, plausible given TrashNet-style waste data at 512x384 or smaller), upscale with bicubic to a moderate input (192 to 256 px), not to the full 384/512, and skip super-resolution because it adds risk with no reliable Macro-F1 gain.
- **Optimize Macro F1 explicitly with post-hoc per-class logit offsets tuned on out-of-fold predictions instead of trusting raw argmax.** This is the single cheapest and safest lever for a 3-class imbalanced problem, because it uses only training-fold data and touches no test labels.

## Key Findings

1. **Backbones.** ConvNeXt V2 and EfficientNetV2 are the strongest CNNs for transfer learning at this dataset size, and EVA-02 or a modern ViT gives the architectural diversity that makes an ensemble worth building. All are in timm with documented input sizes. A CNN plus a transformer decorrelates errors more than two CNNs.
2. **Low resolution.** Super-resolution before classification helps only in narrow research settings and generally is not worth the risk in a competition. Bicubic upscaling to a moderate resolution is the safe default. Upscaling tiny images to 384 or 512 fabricates no new information and wastes compute.
3. **Macro F1.** For a 3-class problem the biggest safe gains come from post-hoc logit adjustment and per-class threshold tuning on out-of-fold predictions, plus class-balanced sampling or weighted loss when imbalance exceeds roughly 3:1.
4. **Mixup and CutMix** often hurt when fine-tuning strong pretrained backbones for few epochs. They are hypotheses to test on OOF, not defaults.
5. **TTA is not free.** Recent evidence shows aggressive TTA can degrade accuracy severely. Use flip-only or flip plus one scale, and validate on OOF before committing a submission to it.
6. **Validation.** 5-fold stratified CV with out-of-fold Macro F1 is the model-selection signal. A strict internal ranking guards the 3-submission budget.
7. **Data integrity.** Waste datasets frequently contain duplicates and mislabeled images. Perceptual hashing finds near-duplicates and cleanlab flags label noise, but be conservative near a deadline.

## Details

### Competition context and rule confirmation
This task follows the Satria Data Big Data Challenge (BDC) format. The official Satria Data 2026 Panduan confirms the general BDC rules: it is a 3-person team competition with qualification (penyisihan), semifinal, and final stages; teams may submit at most 3 times during qualification ("Tim peserta dapat mengirimkan hasil pekerjaan sebanyak-banyaknya 3 (tiga) kali selama masa tahap penyisihan"); results are submitted in a provided template file, and format conformance is required for the automatic scoring system to read the submission; ties are broken in favor of the earlier upload. The specific problem and metric for BDC are announced in separate socialization material ("Materi Sosialisasi BDC"), not in the main Panduan. The qualification stage historically runs on a Kaggle-style platform (for example the 2023 ITB and 2025 UNAIR qualification competitions on Kaggle), which is consistent with a CSV submission. The 2026 host is Universitas Diponegoro, and the official BDC work window is 1 to 31 July 2026.

Two consequences follow. First, the native image resolution is not published in the accessible official documents, so the EDA diagnostic that measures it is mandatory, not optional. Second, the exact CSV schema (column names, header, filename convention) must be confirmed from the official platform template before the final submission; the writer below defaults to `id,predicted` as stated in the task but is parameterized so you can change it in one place.

### Backbone recommendation
For an A100 with roughly 26,500 training images, use two diverse backbones:

- **CNN member:** `convnextv2_base.fcmae_ft_in22k_in1k_384` or `convnextv2_large.fcmae_ft_in22k_in1k_384`. Per the timm and Hugging Face model cards, `convnextv2_base.fcmae_ft_in22k_in1k_384` (88.7M parameters, 45.2 GMACs) reaches 87.646 percent top-1 on ImageNet-1k. The often-quoted 88.9 percent figure belongs to ConvNeXt V2 Huge (659M parameters), which Woo et al. (2023) report at 88.9 percent top-1 on ImageNet-1k (`convnextv2_huge.fcmae_ft_in22k_in1k_384` = 88.668 percent, and the 512 variant = 88.848 percent). Do not reach for Huge unless the A100 memory and time budget clearly allow it; the accuracy gain per compute is poor for a 3-class task.
- **Transformer member:** `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`, or a ViT/DeiT III model, or the SBB ViT `vit_so150m2_patch16_reg1_gap_384.sbb_e200_in12k_ft_in1k` (87.9 percent top-1). If transformer training proves unstable on small upscaled images, substitute a second CNN: `tf_efficientnetv2_m.in21k_ft_in1k`. Per Tan and Le (EfficientNetV2, ICML 2021, arXiv:2104.00298), EfficientNetV2-M trains at 384 and the authors "restrict the maximum inference image size to 480, as very large images often lead to expensive memory and training speed overhead"; the family "achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT-L/16 by 2.0% accuracy while training 5x-11x faster." In timm the config is `input_size=(3,384,384), test_input_size=(3,480,480)`.

Document the exact timm model strings and pretraining tags in the final report, because the rules require the backbone to be documented.

### Low-resolution strategy
If the EDA shows small images, upscale with bicubic (or Lanczos) to a moderate resolution and avoid super-resolution. Research on low-resolution classification (arXiv:2208.03641 and related work) notes that adding a super-resolution step before the classifier requires paired high-resolution training images for the specific classes, which you do not have, and does not reliably improve downstream accuracy. Devil's advocate on your own suspicion: "super-resolution helps low-res classification" is largely false in this setting. A GAN-based upscaler (Real-ESRGAN) hallucinates texture that can be class-inconsistent, and it introduces a train/test preprocessing dependency that is easy to get wrong. Bicubic is deterministic, fast, and label-preserving.

Do not aggressively downscale either. If images are natively larger than the chosen input (for example 512x384), resize the short side to the target and center or random-resized crop, which keeps object detail. For CNNs on genuinely tiny inputs (32 to 64 px), the literature (arXiv:2209.07399, ConvNeXt-for-small-images work) shows reducing the stem stride helps, but this only matters if you train from a small-image checkpoint; when upscaling to 192 to 256 for an ImageNet-pretrained backbone, keep the standard stem.

### Training recipe
Concrete fine-tuning recipe, drawn from published ConvNeXt V2, MAE, and CLIP fine-tuning configs:
- Optimizer AdamW, betas (0.9, 0.999), weight decay 0.05.
- Cosine decay with linear warmup of 3 to 5 epochs; base learning rate about 4e-4 for ConvNeXt-scale batch 1024, scaled linearly to your batch (roughly 5e-5 to 1e-4 at batch 128 to 256).
- Layer-wise learning rate decay 0.65 to 0.75 for transformers and ConvNeXt.
- Label smoothing 0.1.
- Drop path 0.1 (base) to 0.2 (large).
- EMA of weights around 0.9998; published CLIP fine-tuning shows EMA "improves performance in most cases," especially when the learning rate is not perfectly tuned.
- Mixed precision bf16 on A100; batch 128 to 256; gradient accumulation only if memory forces a smaller batch.
- Roughly 15 to 30 epochs; early stop on OOF Macro F1 with patience.
- RandAugment moderate strength. Treat mixup 0.8 and cutmix 1.0 as ablations: the CLIP fine-tuning paper found "removing the Mixup/Cutmix gets the best results" under short fine-tuning because the pretrained features are already strong. On a 3-class problem, mixing two classes can also blur the exact decision boundaries that Macro F1 rewards, so verify on OOF before enabling.

### Macro F1 optimization
Train with cross-entropy (optionally class-weighted), then apply post-hoc logit adjustment. Menon et al. (Long-Tail Learning via Logit Adjustment, ICLR 2021, arXiv:2007.07314) show that the adjustment "adds a label-dependent offset to each of the logits," predicting the argmax of s_c(x) + tau times log p(c), where the priors are "the empirical class frequencies on the training sample," and that this rule is "consistent for minimising the balanced error." In practice, fit a small per-class offset vector by directly maximizing Macro F1 on the stacked OOF probabilities using a coordinate search or scipy optimizer. This is safe under the rules because it uses only training-fold labels.

On loss choice: for mild imbalance, class-weighted cross-entropy is sufficient and simpler; focal loss helps mainly under severe imbalance or many hard negatives. For a 3-class waste problem, start with weighted cross-entropy and only try focal loss if one class F1 lags badly. If the imbalance ratio exceeds about 3:1, add class-balanced sampling (a WeightedRandomSampler) so minibatches see minority classes often enough.

### Ensembling and TTA
Average probabilities across folds and across the two backbones, weighting members by their OOF Macro F1. Probability averaging is the robust default; rank averaging matters mainly for AUC metrics, not argmax Macro F1. Logit averaging is a reasonable alternative but probability averaging is safer when temperature differs across models.

For TTA, stay minimal. Medeiros ("I Can't Believe TTA Is Not Better," arXiv:2604.09697, 6 April 2026) finds TTA "consistently degrades accuracy relative to single-pass inference, with drops as severe as 31.6 percentage points for ResNet-18 on pathology images," and concludes "TTA should not be applied as a default post-hoc improvement but must be validated on the specific model-dataset combination." Shanmugam et al. (Better Aggregation in Test-Time Augmentation, arXiv:2011.11156) add that "even when test-time augmentation produces a net improvement in accuracy, it can change many correct predictions into incorrect predictions." Practical rule: use horizontal flip only, or flip plus one mild scale, and only keep TTA if it raises OOF Macro F1. Intensity-only augmentations are safer than geometric ones because they do not disturb the spatial statistics that batch-norm layers calibrate to.

### Data integrity
Run perceptual hashing (pHash via the `imagehash` library) to find exact and near-duplicates inside train, and cluster near-duplicates so they land in the same CV fold; otherwise duplicates leak between train and validation folds and inflate your OOF score. You cannot check train-test leakage against labels, but you can and should keep duplicate clusters intact within folds. For label noise, cleanlab confident learning can flag likely-mislabeled images using cross-validated predicted probabilities. Be conservative: cleanlab helped modestly at low noise in benchmarks but had "little effect" at higher noise, so review flagged images rather than mass-deleting, and never prune within days of the deadline.

---

## Complete End-to-End Notebook

The code below is organized into clearly numbered cells. It auto-detects DGX A100 versus Colab T4 and scales settings. It computes test image sizes only for the inference resize decision, which is permissible because it uses no labels and no test information leaks into training.

```python
# =====================================================================
# CELL 1  Environment setup and library installs
# =====================================================================
# On Colab run this cell. On a DGX with a prepared conda env, skip installs.
import os, sys, subprocess

def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

# timm >= 1.0 gives ConvNeXt V2, EVA-02, SigLIP2, DINOv3 mapped models.
pip_install([
    "timm>=1.0.15", "torch", "torchvision",
    "scikit-learn", "pandas", "numpy", "pillow",
    "imagehash", "cleanlab", "tqdm", "matplotlib", "seaborn", "gdown"
])

import torch, timm, numpy as np, pandas as pd
print("torch", torch.__version__, "| timm", timm.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          "| count:", torch.cuda.device_count())
```

```python
# =====================================================================
# CELL 2  Global config with automatic hardware scaling
# =====================================================================
import torch, random, numpy as np

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # set True for speed once resolution is fixed
set_seed()

def detect_hardware():
    if not torch.cuda.is_available():
        return "cpu", 0
    name = torch.cuda.get_device_name(0).lower()
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if "a100" in name or mem_gb > 60:
        return "a100", mem_gb
    if "t4" in name or mem_gb < 20:
        return "t4", mem_gb
    return "other", mem_gb

HW, GPU_MEM = detect_hardware()
print("Detected hardware profile:", HW, f"({GPU_MEM:.0f} GB)")

class CFG:
    # ---- paths (edit for your environment) ----
    DATA_ROOT   = "/content/dataset"        # contains train/ and test/
    TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
    TEST_DIR    = os.path.join(DATA_ROOT, "test")
    OUT_DIR     = "/content/work"
    TEAM_NAME   = "TeamName"                 # -> submission_TeamName.csv

    # ---- submission schema (CONFIRM against official template.csv) ----
    ID_COL      = "id"
    PRED_COL    = "predicted"

    CLASSES     = [0, 1, 2]                  # 0 Recyclable, 1 Electronic, 2 Organic
    N_CLASSES   = 3

    # ---- CV ----
    N_FOLDS     = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]            # train all; set fewer to save time

    # ---- input size (auto-recommended in EDA, overridable) ----
    IMG_SIZE    = 224                        # placeholder; EDA overwrites RECOMMENDED_SIZE

    # ---- models: (timm_name, input_size, batch_a100, batch_t4) ----
    MODELS = [
        ("convnextv2_base.fcmae_ft_in22k_in1k_384", None, 128, 16),
        ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", None, 64, 8),
        # T4 fallback / EfficientNetV2 alternative:
        # ("tf_efficientnetv2_m.in21k_ft_in1k", None, 96, 12),
    ]

    EPOCHS      = 20 if HW == "a100" else 12
    WARMUP_EP   = 3
    BASE_LR     = 1.2e-4                     # tune per batch size
    MIN_LR      = 1e-6
    WEIGHT_DECAY= 0.05
    LABEL_SMOOTH= 0.1
    LLRD        = 0.75
    DROP_PATH   = 0.1
    EMA_DECAY   = 0.9998
    USE_MIXUP   = False                      # ablate on OOF before enabling
    MIXUP_A     = 0.8
    CUTMIX_A    = 1.0

    AMP_DTYPE   = torch.bfloat16 if HW == "a100" else torch.float16
    NUM_WORKERS = 8 if HW == "a100" else 2
    GRAD_ACCUM  = 1

    IMBALANCE_SAMPLER_THRESHOLD = 3.0        # ratio > this -> balanced sampler
    TTA_HFLIP   = True
    TTA_SCALES  = [1.0]                      # add e.g. 0.9 only if OOF improves

os.makedirs(CFG.OUT_DIR, exist_ok=True)

def batch_for(model_batch_a100, model_batch_t4):
    return model_batch_a100 if HW == "a100" else model_batch_t4
```

```python
# =====================================================================
# CELL 3  Dataset download / mount  (two paths shown)
# =====================================================================
# The dataset lives behind bit.ly/datasetbdc2026 (likely a Google Drive zip).

# ---- Path A: Colab, mount Google Drive and unzip ----
# from google.colab import drive
# drive.mount('/content/drive')
# import zipfile
# ZIP = '/content/drive/MyDrive/datasetbdc2026.zip'   # adjust
# with zipfile.ZipFile(ZIP) as z: z.extractall(CFG.DATA_ROOT)

# ---- Path B: direct download with gdown (resolve the bit.ly first) ----
# Resolve the short link in a browser to get the real Drive file id, then:
# import gdown, zipfile
# FILE_ID = 'PUT_DRIVE_FILE_ID_HERE'
# gdown.download(f'https://drive.google.com/uc?id={FILE_ID}',
#                '/content/data.zip', quiet=False)
# with zipfile.ZipFile('/content/data.zip') as z: z.extractall(CFG.DATA_ROOT)

# ---- On DGX A100: data usually already staged on a fast local disk ----
assert os.path.isdir(CFG.TRAIN_DIR), f"train dir not found: {CFG.TRAIN_DIR}"
assert os.path.isdir(CFG.TEST_DIR),  f"test  dir not found: {CFG.TEST_DIR}"
print("train subfolders:", sorted(os.listdir(CFG.TRAIN_DIR)))
```

```python
# =====================================================================
# CELL 4  Build train index + natural-sorted test index
# =====================================================================
import re, glob
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Map class subfolder names to integer codes. Adjust if folders are named 0/1/2
# or 'recyclable'/'electronic'/'organic'.
CLASS_DIR_TO_CODE = {}
subdirs = sorted(os.listdir(CFG.TRAIN_DIR))
for d in subdirs:
    low = d.lower()
    if d in ("0","1","2"):                     CLASS_DIR_TO_CODE[d] = int(d)
    elif "recycl" in low:                      CLASS_DIR_TO_CODE[d] = 0
    elif "electro" in low or "e-waste" in low: CLASS_DIR_TO_CODE[d] = 1
    elif "organ" in low:                       CLASS_DIR_TO_CODE[d] = 2
print("class dir -> code:", CLASS_DIR_TO_CODE)

train_rows = []
for d, code in CLASS_DIR_TO_CODE.items():
    for p in glob.glob(os.path.join(CFG.TRAIN_DIR, d, "*")):
        if p.lower().endswith(IMG_EXT):
            train_rows.append({"path": p, "label": code})
train_df = pd.DataFrame(train_rows).reset_index(drop=True)
print("train images:", len(train_df))

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]

test_paths = [p for p in glob.glob(os.path.join(CFG.TEST_DIR, "*"))
              if p.lower().endswith(IMG_EXT)]
# CRITICAL: natural sort so 1,2,...,10 not 1,10,2. Match template.csv order.
test_paths = sorted(test_paths, key=lambda p: natural_key(os.path.basename(p)))
test_ids   = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
print("test images:", len(test_paths), "| first 5 ids:", test_ids[:5])
```

```python
# =====================================================================
# CELL 5  EDA: dimension scan, formats, corrupt files, class balance
#           -> automatic input-size recommendation
# =====================================================================
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def scan_images(paths):
    recs, corrupt = [], []
    for p in tqdm(paths, desc="scanning"):
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size; fmt = im.format; mode = im.mode
            recs.append((p, w, h, fmt, mode))
        except Exception as e:
            corrupt.append((p, str(e)))
    df = pd.DataFrame(recs, columns=["path","w","h","fmt","mode"])
    return df, corrupt

train_scan, train_corrupt = scan_images(train_df["path"].tolist())
test_scan,  test_corrupt  = scan_images(test_paths)

print("corrupt train:", len(train_corrupt), "| corrupt test:", len(test_corrupt))
print("\ntrain class distribution:")
print(train_df["label"].value_counts().sort_index())
cls_counts = train_df["label"].value_counts().sort_index()
imb_ratio = cls_counts.max() / cls_counts.min()
print(f"imbalance ratio (max/min): {imb_ratio:.2f}")

for name, s in [("TRAIN", train_scan), ("TEST", test_scan)]:
    mn = s[["w","h"]].min(axis=1)
    print(f"\n{name}: formats {s['fmt'].value_counts().to_dict()}")
    print(f"{name}: width  median {s['w'].median():.0f}  "
          f"[{s['w'].min()}-{s['w'].max()}]")
    print(f"{name}: height median {s['h'].median():.0f}  "
          f"[{s['h'].min()}-{s['h'].max()}]")
    print(f"{name}: min-side median {mn.median():.0f}  "
          f"25th {mn.quantile(.25):.0f}  75th {mn.quantile(.75):.0f}")

# aspect-ratio spread
ar = train_scan["w"] / train_scan["h"]
print(f"\naspect ratio: median {ar.median():.2f} "
      f"[{ar.quantile(.05):.2f}-{ar.quantile(.95):.2f}]")

def recommend_input_size(scan_df):
    med_min = float(scan_df[["w","h"]].min(axis=1).median())
    spread  = scan_df[["w","h"]].min(axis=1)
    mixed   = (spread.quantile(.9) / max(spread.quantile(.1), 1)) > 3
    if   med_min < 96:  size = 128
    elif med_min < 224: size = 224
    else:               size = 256
    return size, med_min, mixed

RECOMMENDED_SIZE, MED_MIN, MIXED = recommend_input_size(train_scan)
CFG.IMG_SIZE = RECOMMENDED_SIZE
print(f"\n>>> median min-side = {MED_MIN:.0f}px | mixed resolution = {MIXED}")
print(f">>> RECOMMENDED input size = {RECOMMENDED_SIZE}px")
print(">>> If native << backbone size: bicubic UPSCALE, do NOT use super-resolution.")
print(">>> If native >> input size: resize short side then crop, do NOT hard-downscale.")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.scatter(train_scan["w"], train_scan["h"], s=3, alpha=.3)
plt.xlabel("width"); plt.ylabel("height"); plt.title("Train dimensions")
plt.subplot(1,2,2); cls_counts.plot(kind="bar"); plt.title("Class distribution")
plt.tight_layout(); plt.show()
```

```python
# =====================================================================
# CELL 6  Duplicate and near-duplicate scan (perceptual hashing)
# =====================================================================
import imagehash
from collections import defaultdict

def phash_all(paths, hash_size=8):
    d = {}
    for p in tqdm(paths, desc="phash"):
        try:
            with Image.open(p) as im:
                d[p] = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        except Exception:
            pass
    return d

train_hashes = phash_all(train_df["path"].tolist())

# exact hash collisions
buckets = defaultdict(list)
for p, h in train_hashes.items():
    buckets[str(h)].append(p)
exact_dups = {k: v for k, v in buckets.items() if len(v) > 1}
print("exact-duplicate clusters (train):", len(exact_dups))

# near-duplicate clustering (Hamming distance <= threshold) via union-find
items = list(train_hashes.items())
parent = {p: p for p, _ in items}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    parent[find(a)] = find(b)

THRESH = 5  # tighten to 3 for stricter, loosen for looser
for i in range(len(items)):
    pi, hi = items[i]
    for j in range(i+1, len(items)):
        pj, hj = items[j]
        if (hi - hj) <= THRESH:
            union(pi, pj)
clusters = defaultdict(list)
for p, _ in items:
    clusters[find(p)].append(p)
near_dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
print("near-duplicate clusters (train):", len(near_dup_clusters))

# Assign a group id per image so CV keeps duplicates together (prevents leakage).
img_to_group = {}
gid = 0
for members in clusters.values():
    for m in members:
        img_to_group[m] = gid
    gid += 1
train_df["group"] = train_df["path"].map(lambda p: img_to_group.get(p, -1))
print("distinct groups:", train_df["group"].nunique())
# Do NOT delete duplicates near the deadline; grouping is enough to protect OOF.
```

```python
# =====================================================================
# CELL 7  Optional: cleanlab label-noise flagging (review, do not mass-delete)
# =====================================================================
# Run this AFTER you have OOF predicted probabilities (Cell 12). Kept here for
# structure; call flag_label_issues(oof_probs, labels) once available.
from cleanlab.filter import find_label_issues

def flag_label_issues(oof_probs, labels):
    idx = find_label_issues(
        labels=np.asarray(labels),
        pred_probs=np.asarray(oof_probs),
        return_indices_ranked_by="self_confidence",
    )
    print(f"cleanlab flagged {len(idx)} potential label issues "
          f"({100*len(idx)/len(labels):.1f}%). Review manually.")
    return idx
```

```python
# =====================================================================
# CELL 8  Stratified group K-fold split
# =====================================================================
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for f, (_, val_idx) in enumerate(
        sgkf.split(train_df, train_df["label"], groups=train_df["group"])):
    train_df.loc[val_idx, "fold"] = f
print(train_df.groupby(["fold","label"]).size().unstack(fill_value=0))
```

```python
# =====================================================================
# CELL 9  Datasets and transforms (bicubic upscale for small images)
# =====================================================================
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transforms(img_size, train=True):
    # BICUBIC upscaling is the safe choice for small native images.
    if train:
        return T.Compose([
            T.Resize(int(img_size*1.15), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=7),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(int(img_size*1.15), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class WasteDataset(Dataset):
    def __init__(self, df, tfm, has_label=True):
        self.df = df.reset_index(drop=True); self.tfm = tfm; self.has_label = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        x = self.tfm(img)
        if self.has_label:
            return x, int(r["label"])
        return x, i

def make_sampler(sub_df):
    counts = sub_df["label"].value_counts().sort_index()
    if counts.max()/counts.min() <= CFG.IMBALANCE_SAMPLER_THRESHOLD:
        return None
    w_per_class = (1.0 / counts).to_dict()
    weights = sub_df["label"].map(w_per_class).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

```python
# =====================================================================
# CELL 10  Model, EMA, LLRD optimizer, cosine+warmup scheduler
# =====================================================================
import math, copy
import torch.nn as nn

def create_model(name):
    return timm.create_model(name, pretrained=True,
                             num_classes=CFG.N_CLASSES,
                             drop_path_rate=CFG.DROP_PATH)

class ModelEMA:
    def __init__(self, model, decay=CFG.EMA_DECAY):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.state_dict().values(),
                        model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m, alpha=1-self.decay)
            else:
                e.copy_(m)

def build_llrd_param_groups(model, base_lr, wd, decay=CFG.LLRD):
    # Approximate layer index by module depth; head gets full lr.
    params = []
    named = list(model.named_parameters())
    n = len(named)
    for i, (name, p) in enumerate(named):
        if not p.requires_grad: continue
        scale = decay ** ((n - i) / n)           # deeper (later) -> closer to base_lr
        this_wd = 0.0 if p.ndim == 1 or name.endswith(".bias") else wd
        params.append({"params": [p], "lr": base_lr*scale, "weight_decay": this_wd})
    return params

def cosine_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1-min_lr_ratio)*0.5*(1+math.cos(math.pi*prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
```

```python
# =====================================================================
# CELL 11  Training + validation loop (one fold, one model)
# =====================================================================
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def class_weights(sub_df):
    counts = sub_df["label"].value_counts().sort_index().values.astype(float)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)

def train_one_fold(model_name, model_input, fold, df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = model_input or CFG.IMG_SIZE
    tr = df[df.fold != fold].copy(); va = df[df.fold == fold].copy()

    tr_ds = WasteDataset(tr, build_transforms(img_size, True))
    va_ds = WasteDataset(va, build_transforms(img_size, False))
    sampler = make_sampler(tr)
    bs = batch_for(*[b for (n,_,ba,bt) in CFG.MODELS if n==model_name for b in (ba,bt)][:2]) \
         if False else None
    # simple batch lookup:
    bs = next(batch_for(ba, bt) for (n,_,ba,bt) in CFG.MODELS if n==model_name)

    tr_dl = DataLoader(tr_ds, batch_size=bs, sampler=sampler,
                       shuffle=(sampler is None), num_workers=CFG.NUM_WORKERS,
                       pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False,
                       num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = create_model(model_name).to(device)
    ema = ModelEMA(model)
    opt = torch.optim.AdamW(build_llrd_param_groups(model, CFG.BASE_LR, CFG.WEIGHT_DECAY),
                            betas=(0.9,0.999))
    total_steps = CFG.EPOCHS * len(tr_dl) // CFG.GRAD_ACCUM
    warmup = CFG.WARMUP_EP * len(tr_dl) // CFG.GRAD_ACCUM
    sched = cosine_warmup(opt, warmup, total_steps, CFG.MIN_LR/CFG.BASE_LR)

    cw = class_weights(tr).to(device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=CFG.LABEL_SMOOTH)
    scaler = torch.cuda.amp.GradScaler(enabled=(CFG.AMP_DTYPE==torch.float16))

    best_f1, best_state, best_probs = -1, None, None
    for ep in range(CFG.EPOCHS):
        model.train(); opt.zero_grad()
        for it,(x,y) in enumerate(tqdm(tr_dl, desc=f"{model_name[:14]} f{fold} e{ep}")):
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            with torch.autocast("cuda", dtype=CFG.AMP_DTYPE):
                out = model(x); loss = crit(out, y) / CFG.GRAD_ACCUM
            if CFG.AMP_DTYPE==torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (it+1) % CFG.GRAD_ACCUM == 0:
                if CFG.AMP_DTYPE==torch.float16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(); sched.step(); ema.update(model)

        # validate with EMA weights
        ema.ema.eval(); probs, ys = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=CFG.AMP_DTYPE):
            for x,y in va_dl:
                p = torch.softmax(ema.ema(x.to(device)), 1).float().cpu().numpy()
                probs.append(p); ys.append(y.numpy())
        probs = np.concatenate(probs); ys = np.concatenate(ys)
        f1 = f1_score(ys, probs.argmax(1), average="macro")
        print(f"  epoch {ep} macro-F1 (EMA) = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; best_probs = probs
            best_state = copy.deepcopy(ema.ema.state_dict())

    ckpt = os.path.join(CFG.OUT_DIR, f"{model_name.split('.')[0]}_f{fold}.pt")
    torch.save(best_state, ckpt)
    print(f"  >> best fold {fold} macro-F1 = {best_f1:.4f}  saved {ckpt}")
    return best_f1, best_probs, va.index.values, ckpt, img_size
```

```python
# =====================================================================
# CELL 12  Run CV for all models; collect OOF probabilities
# =====================================================================
oof = {}   # model_name -> (oof_probs [N,3], ckpt_paths, img_size)
for (mname, minput, _, _) in CFG.MODELS:
    oof_probs = np.zeros((len(train_df), CFG.N_CLASSES), dtype=np.float32)
    ckpts, isize = [], None
    for fold in CFG.TRAIN_FOLDS:
        f1, probs, idx, ckpt, isize = train_one_fold(mname, minput, fold, train_df)
        oof_probs[idx] = probs; ckpts.append((fold, ckpt))
    macro = f1_score(train_df["label"].values, oof_probs.argmax(1), average="macro")
    print(f"\n=== {mname} OOF macro-F1 = {macro:.4f} ===")
    print(classification_report(train_df["label"].values, oof_probs.argmax(1),
                                digits=4))
    print(confusion_matrix(train_df["label"].values, oof_probs.argmax(1)))
    np.save(os.path.join(CFG.OUT_DIR, f"oof_{mname.split('.')[0]}.npy"), oof_probs)
    oof[mname] = (oof_probs, ckpts, isize)
```

```python
# =====================================================================
# CELL 13  Ensemble weight search + per-class logit-offset tuning (Macro F1)
# =====================================================================
from scipy.optimize import minimize
from itertools import product

y_true = train_df["label"].values
model_names = list(oof.keys())
oof_stack = [oof[m][0] for m in model_names]

# 13a. grid-search convex weights on OOF (coarse)
def blended(weights, stack):
    w = np.array(weights); w = w/w.sum()
    return sum(wi*si for wi, si in zip(w, stack))

best_w, best_f1 = None, -1
grid = np.linspace(0, 1, 11)
for combo in product(grid, repeat=len(model_names)):
    if abs(sum(combo)-1) > 1e-6 or sum(combo)==0: continue
    p = blended(combo, oof_stack)
    f1 = f1_score(y_true, p.argmax(1), average="macro")
    if f1 > best_f1: best_f1, best_w = f1, combo
print("best ensemble weights:", best_w, "OOF macro-F1:", round(best_f1,4))

ens_oof = blended(best_w, oof_stack)

# 13b. per-class logit offset (Menon et al. logit adjustment, tuned for Macro F1)
eps = 1e-6
log_oof = np.log(ens_oof + eps)
def neg_macro_f1(offsets):
    adj = log_oof + offsets[None, :]
    return -f1_score(y_true, adj.argmax(1), average="macro")
res = minimize(neg_macro_f1, x0=np.zeros(CFG.N_CLASSES),
               method="Nelder-Mead",
               options={"maxiter": 2000, "xatol":1e-4, "fatol":1e-5})
OFFSETS = res.x
print("tuned per-class offsets:", np.round(OFFSETS,3),
      "| OOF macro-F1 after offset:", round(-res.fun,4))
```

```python
# =====================================================================
# CELL 14  Ensemble inference over test set with flip-only TTA
# =====================================================================
def predict_model(model_name, ckpts, img_size, paths, tta_hflip, scales):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = next(batch_for(ba, bt) for (n,_,ba,bt) in CFG.MODELS if n==model_name)
    base = create_model(model_name).to(device).eval()
    agg = np.zeros((len(paths), CFG.N_CLASSES), dtype=np.float32)
    n_views = 0
    for (fold, ckpt) in ckpts:
        base.load_state_dict(torch.load(ckpt, map_location=device))
        for scale in scales:
            s = int(round(img_size*scale))
            tfm = build_transforms(s, train=False)
            ds = WasteDataset(pd.DataFrame({"path": paths}), tfm, has_label=False)
            dl = DataLoader(ds, batch_size=bs, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)
            for flip in ([False, True] if tta_hflip else [False]):
                with torch.no_grad(), torch.autocast("cuda", dtype=CFG.AMP_DTYPE):
                    for x, idx in dl:
                        x = x.to(device)
                        if flip: x = torch.flip(x, dims=[3])
                        p = torch.softmax(base(x), 1).float().cpu().numpy()
                        agg[idx.numpy()] += p
                n_views += 1
    return agg / (n_views * len(ckpts) / len(ckpts))   # normalize by views

test_probs_per_model = []
for m in model_names:
    _, ckpts, isize = oof[m]
    tp = predict_model(m, ckpts, isize, test_paths, CFG.TTA_HFLIP, CFG.TTA_SCALES)
    test_probs_per_model.append(tp / tp.sum(1, keepdims=True))

ens_test = blended(best_w, test_probs_per_model)
adj_test = np.log(ens_test + 1e-6) + OFFSETS[None, :]
test_pred = adj_test.argmax(1).astype(int)
print("test prediction distribution:", np.bincount(test_pred, minlength=3))
```

```python
# =====================================================================
# CELL 15  Submission writer with validation
# =====================================================================
def write_submission(ids, preds, team=CFG.TEAM_NAME):
    assert len(ids) == len(preds), "id/pred length mismatch"
    sub = pd.DataFrame({CFG.ID_COL: ids, CFG.PRED_COL: preds})
    # validations
    assert len(sub) == 1458, f"expected 1458 rows, got {len(sub)}"
    assert list(sub.columns) == [CFG.ID_COL, CFG.PRED_COL], "column names/order wrong"
    assert set(sub[CFG.PRED_COL].unique()).issubset({0,1,2}), "labels not in {0,1,2}"
    # cast ids to int where possible to match a numeric template
    try:
        sub[CFG.ID_COL] = sub[CFG.ID_COL].astype(int)
        sub = sub.sort_values(CFG.ID_COL).reset_index(drop=True)  # numeric order 1..1458
    except ValueError:
        pass
    path = os.path.join(CFG.OUT_DIR, f"submission_{team}.csv")
    sub.to_csv(path, index=False)
    print("wrote", path); print(sub.head()); print("rows:", len(sub))
    return sub

submission = write_submission(test_ids, test_pred)
```

---

## Tuning Guide: what to change once you see the real data

**Input size (from EDA Cell 5).**
- If median min-side < 96 px: set `CFG.IMG_SIZE = 128`, use ConvNeXt V2 base or an EfficientNetV2-S, keep RandomResizedCrop scale near (0.8, 1.0) to avoid cropping away tiny objects, and bicubic upscale. Do not use super-resolution.
- If 96 to 224 px: `IMG_SIZE = 224`, standard recipe.
- If > 224 px or mixed and large: `IMG_SIZE = 256` (or 288/384 on A100 if time allows). Resize short side then crop; do not hard-downscale to a tiny square.
- If mixed resolution (Cell 5 flag `MIXED=True`): pick a size near the median and rely on RandomResizedCrop for scale robustness; consider a multi-scale ensemble member.

**Batch size and model size.**
- A100: batch 128 (ConvNeXt base) or 64 (EVA-02 base); step up to convnextv2_large only if OOF clearly improves and time permits.
- T4 (Colab): use the T4 batch column, switch to `tf_efficientnetv2_s` or `convnextv2_nano`, and reduce epochs to about 10 to 12.

**Class imbalance (from Cell 5 `imb_ratio`).**
- Ratio <= 3:1: weighted cross-entropy only (already in the loop).
- Ratio > 3:1: the sampler auto-activates. If a class F1 still lags, add focal loss (gamma 2.0) as an ablation and keep whichever wins on OOF.

**Augmentation strength.**
- Small images: lower RandAugment magnitude to 5 to 7 and keep RandomErasing at 0.25.
- If OOF Macro F1 improves with mixup/cutmix, set `CFG.USE_MIXUP = True`; otherwise leave off. Do not assume they help.

**Backbone swap.**
- If the transformer member is unstable (loss spikes, low OOF), replace EVA-02 with `tf_efficientnetv2_m.in21k_ft_in1k` for a stable second CNN. Diversity from a different training recipe still helps the ensemble.

**Decision rule summary.** If median size < 96px do input 128 with a CNN and bicubic upscaling; if 96 to 224 do input 224 standard; if mixed or larger do input 256 to 384 with short-side resize; if imbalance ratio > 3:1 turn on the balanced sampler and consider focal loss.

## Submission-Budget Plan (3 submissions)

- **Submission 1 (safety, early):** best single backbone (ConvNeXt V2), 5-fold averaged, flip-only TTA, argmax with logit offsets. Establishes a solid baseline on the real leaderboard and confirms the CSV format is accepted.
- **Submission 2 (main):** full 2-model ensemble with OOF-tuned weights, per-class logit offsets, flip-only TTA. This should be your highest OOF Macro F1.
- **Submission 3 (reserve):** spend only if a variant beats submission 2 on OOF by a clear margin, for example adding a multi-scale TTA view or a third backbone or cleanlab-cleaned retraining. If nothing beats submission 2 on OOF, do not spend it; the rules keep your best score, but each upload also serves as a tiebreaker timestamp, so upload the best version as early as it is ready.

**Timeline sanity check.** The BDC work window is 1 to 31 July 2026, and the task cites a 30 July 16:00 WIB deadline (the official Panduan lists the work period ending 31 July; confirm the exact BDC cutoff from the socialization material). On an A100, one ConvNeXt V2 base fold at 224 px over 20 epochs on about 21k training images per fold runs in a few hours; 5 folds times 2 models is roughly 1 to 2 days of wall-clock with checkpointing. Plan to freeze training by about 27 July, reserve 2 to 3 days for ensembling, offset tuning, and submission validation, and keep a buffer for the mandatory anti-manual verification video.

## Compliance Checklist

- **Model built only on provided training data:** all training reads from `CFG.TRAIN_DIR`; test images are used only in Cell 14 for prediction. No external labeled datasets are used. Pretrained ImageNet-21k/1k weights are allowed by the rules and were never trained on this competition's data.
- **Only visual content used:** no filenames, EXIF, or metadata feed the model; the pipeline reads pixels only.
- **No test-data leakage into training:** CV splits, class weights, sampler, augmentation, logit offsets, and ensemble weights are all computed from training-fold labels only. Test image sizes are read solely to resize at inference, which uses no labels and does not influence any trained parameter.
- **Documented backbone:** the exact timm model strings (for example `convnextv2_base.fcmae_ft_in22k_in1k_384`, `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`) and their pretraining tags are in `CFG.MODELS`; copy them verbatim into the final report.
- **Non-manual prediction:** predictions come entirely from Cells 12 to 15 with no hand editing; the notebook is deterministic given `SEED`, which supports the required proof-of-work video.
- **Unchanged CSV format and order:** Cell 15 validates exactly 1458 rows, columns `id,predicted` in that order, values in {0,1,2}, and sorts by numeric id to match template ordering. Cell 4 uses natural sort so filenames like 1.jpg, 10.jpg, 2.jpg map correctly.
- **Reproducibility:** global seed, cudnn determinism, per-fold checkpoints, and saved OOF arrays enable exact reruns.

## Caveats
- Several competition specifics could not be confirmed from official sources: the native image resolution, the exact CSV schema and header, and the precise deadline time. The main Satria Data 2026 Panduan confirms the 3-submission limit, template-file submission, earlier-upload tiebreak, and 1 to 31 July work window, but the waste-task details live in the separate BDC socialization material, which was not machine-accessible. Confirm the resolution empirically via the EDA cell, and confirm the template column names and id format from the official platform before your final upload.
- Reported model accuracies (for example ConvNeXt V2 base at 87.646 percent, Huge at 88.9 percent, EfficientNetV2-M at 87.3 percent) are ImageNet-1k numbers and only indicate relative backbone strength; your Macro F1 on this 3-class task will differ and must be judged on OOF.
- The TTA and mixup/cutmix guidance is deliberately cautious because published evidence shows both can hurt on specific datasets; treat every such switch as an OOF-validated ablation, not a default.
- Devil's-advocate points to keep in mind: a bigger backbone is not always better here (a 3-class task with tens of thousands of images saturates a base model, and Huge mostly buys overfitting risk and slower iteration); TTA does not always help and can flip correct predictions; and super-resolution rarely helps low-resolution classification when you lack paired high-resolution training data.
