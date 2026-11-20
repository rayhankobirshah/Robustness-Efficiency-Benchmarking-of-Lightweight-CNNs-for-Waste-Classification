"""
Robustness-Efficiency Benchmark of Lightweight CNNs for Waste Classification
Run on Kaggle (GPU T4, Internet ON) after adding the dataset:
  https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
All outputs are saved to /kaggle/working/results
"""
import os, glob, io, time, random, copy, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# ----------------------------- CONFIG -----------------------------
SEED = 42
IMG = 224
BATCH = 32
CV_FOLDS = 3          # use 5 if you have more time
CV_EPOCHS = 4
FINAL_EPOCHS = 8
LR = 5e-4
MODELS = ["mobilenet_v3_small", "shufflenet_v2_x1_0", "efficientnet_b0", "squeezenet1_1"]
OUT = "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"
print("Device:", DEVICE)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


set_seed(SEED)

# ----------------------------- DATA -----------------------------
def find_root():
    for p in glob.glob("/kaggle/input/**/cardboard", recursive=True):
        if os.path.isdir(p):
            return os.path.dirname(p)
    raise FileNotFoundError("Dataset not found. Add it via 'Add Input' in Kaggle.")


ROOT = find_root()
CLASSES = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
print("Root:", ROOT, "\nClasses:", CLASSES)

paths, labels = [], []
for ci, c in enumerate(CLASSES):
    for f in glob.glob(os.path.join(ROOT, c, "*")):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(f); labels.append(ci)
labels = np.array(labels)
print("Total images:", len(paths))

print("Pre-loading images into RAM ...")
IMGS = [Image.open(p).convert("RGB").resize((IMG, IMG)) for p in paths]

idx_all = np.arange(len(IMGS))
trainval_idx, test_idx = train_test_split(idx_all, test_size=0.2, stratify=labels, random_state=SEED)
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=labels[trainval_idx], random_state=SEED)
print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

# ----------------------------- CORRUPTIONS -----------------------------
def c_blur(img, s):   return img.filter(ImageFilter.GaussianBlur([1, 2, 4][s - 1]))
def c_noise(img, s):
    a = np.asarray(img).astype(np.float32)
    a = a + np.random.randn(*a.shape) * [10, 25, 50][s - 1]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
def c_dark(img, s):   return ImageEnhance.Brightness(img).enhance([0.6, 0.35, 0.15][s - 1])
def c_jpeg(img, s):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=[30, 15, 5][s - 1]); buf.seek(0)
    return Image.open(buf).convert("RGB")
def c_lowres(img, s):
    f = [2, 4, 8][s - 1]; w = img.size[0]
    return img.resize((w // f, w // f)).resize((w, w))


CORRUPTIONS = {"gaussian_blur": c_blur, "gaussian_noise": c_noise, "low_light": c_dark,
               "jpeg_compression": c_jpeg, "low_resolution": c_lowres}


class RandCorrupt:
    """Corruption-aware augmentation: random corruption + severity with probability p."""
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        fn = random.choice(list(CORRUPTIONS.values()))
        return fn(img, random.randint(1, 3))


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(15),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
tf_eval = T.Compose([T.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, idx, tf, corrupt=None):
        self.idx, self.tf, self.corrupt = idx, tf, corrupt
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; img = IMGS[j]
        if self.corrupt is not None: img = self.corrupt(img)
        return self.tf(img), int(labels[j])


def loader(idx, train=False, corrupt=None):
    return DataLoader(DS(idx, tf_train if train else tf_eval, corrupt), batch_size=BATCH,
                      shuffle=train, num_workers=4, pin_memory=USE_CUDA, drop_last=False)

# ----------------------------- MODELS -----------------------------
def build(name, nc):
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="DEFAULT")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    elif name == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
    elif name == "squeezenet1_1":
        m = models.squeezenet1_1(weights="DEFAULT")
        m.classifier[1] = nn.Conv2d(512, nc, kernel_size=1)
    return m.to(DEVICE)


def get_layer(name, m):
    return {"mobilenet_v3_small": lambda: m.features[-1],
            "shufflenet_v2_x1_0": lambda: m.conv5,
            "efficientnet_b0": lambda: m.features[-1],
            "squeezenet1_1": lambda: m.features[-1]}[name]()


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); crit = nn.CrossEntropyLoss(); tl, preds, ys = 0.0, [], []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x); tl += crit(out, y).item() * len(y)
        preds += out.argmax(1).cpu().tolist(); ys += y.cpu().tolist()
    return tl / len(ys), np.array(preds), np.array(ys)


def train_model(name, tr_idx, va_idx, epochs, corrupt_aug=False, verbose=True):
    model = build(name, len(CLASSES))
    dl_tr = loader(tr_idx, train=True, corrupt=RandCorrupt(0.5) if corrupt_aug else None)
    dl_va = loader(va_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_acc = None, -1
    for ep in range(epochs):
        model.train(); tl, correct, n = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=USE_CUDA):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        vl, vp, vy = evaluate(model, dl_va); va = accuracy_score(vy, vp)
        hist["train_loss"].append(tl / n); hist["train_acc"].append(correct / n)
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if va > best_acc: best_acc, best = va, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  [{name}{' +aug' if corrupt_aug else ''}] ep {ep+1}/{epochs} "
                  f"loss {tl/n:.3f} acc {correct/n:.3f} | val_loss {vl:.3f} val_acc {va:.3f}")
    model.load_state_dict(best)
    return model, hist

# ----------------------------- 1) CROSS-VALIDATION -----------------------------
print("\n=== STEP 1: Cross-validation ===")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_rows = []
for name in MODELS:
    for fold, (a, b) in enumerate(skf.split(trainval_idx, labels[trainval_idx]), 1):
        tr, va = trainval_idx[a], trainval_idx[b]
        model, _ = train_model(name, tr, va, CV_EPOCHS, verbose=False)
        _, p, y = evaluate(model, loader(va))
        cv_rows.append({"model": name, "fold": fold, "accuracy": accuracy_score(y, p),
                        "macro_f1": f1_score(y, p, average="macro")})
        print(f"{name} fold {fold}: acc {cv_rows[-1]['accuracy']:.4f} f1 {cv_rows[-1]['macro_f1']:.4f}")
        del model; torch.cuda.empty_cache()
cv_df = pd.DataFrame(cv_rows); cv_df.to_csv(f"{OUT}/cv_folds.csv", index=False)
cv_sum = cv_df.groupby("model").agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                                    f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std")).round(4)
cv_sum.to_csv(f"{OUT}/cv_summary.csv"); print(cv_sum)

# ----------------------------- 2) FINAL TRAINING -----------------------------
print("\n=== STEP 2: Final training (standard + corruption-aware) ===")
STD, AUG, HIST = {}, {}, {}
for name in MODELS:
    STD[name], HIST[(name, "standard")] = train_model(name, train_idx, val_idx, FINAL_EPOCHS)
    AUG[name], HIST[(name, "corruption_aug")] = train_model(name, train_idx, val_idx, FINAL_EPOCHS, corrupt_aug=True)

COLORS = dict(zip(MODELS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))

# --- accuracy & loss curves
fig, ax = plt.subplots(2, len(MODELS), figsize=(4.5 * len(MODELS), 7), squeeze=False)
for j, name in enumerate(MODELS):
    for v, ls in [("standard", "-"), ("corruption_aug", "--")]:
        h = HIST[(name, v)]; ep = range(1, len(h["train_loss"]) + 1)
        ax[0, j].plot(ep, h["train_loss"], ls, color="tab:blue", label=f"train ({v})")
        ax[0, j].plot(ep, h["val_loss"], ls, color="tab:red", label=f"val ({v})")
        ax[1, j].plot(ep, h["train_acc"], ls, color="tab:blue", label=f"train ({v})")
        ax[1, j].plot(ep, h["val_acc"], ls, color="tab:red", label=f"val ({v})")
    ax[0, j].set_title(name); ax[0, j].set_ylabel("Loss"); ax[1, j].set_ylabel("Accuracy")
    ax[1, j].set_xlabel("Epoch")
ax[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/curves_loss_accuracy.png", dpi=200); plt.close()
pd.concat({f"{k[0]}|{k[1]}": pd.DataFrame(v) for k, v in HIST.items()}).to_csv(f"{OUT}/training_history.csv")

# --- test evaluation + confusion matrices
print("\n=== STEP 3: Test evaluation & confusion matrices ===")
dl_test = loader(test_idx)
fig, ax = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5), squeeze=False)
final = {}
for j, name in enumerate(MODELS):
    _, p, y = evaluate(STD[name], dl_test)
    cm = confusion_matrix(y, p)
    ax[0, j].imshow(cm, cmap="Blues"); ax[0, j].set_title(f"{name}\nacc={accuracy_score(y, p):.3f}")
    ax[0, j].set_xticks(range(len(CLASSES))); ax[0, j].set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=8)
    ax[0, j].set_yticks(range(len(CLASSES))); ax[0, j].set_yticklabels(CLASSES, fontsize=8)
    for r in range(len(CLASSES)):
        for c in range(len(CLASSES)):
            ax[0, j].text(c, r, cm[r, c], ha="center", va="center",
                          color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=8)
    ax[0, j].set_xlabel("Predicted"); ax[0, j].set_ylabel("True")
    with open(f"{OUT}/classification_report_{name}.txt", "w") as f:
        f.write(classification_report(y, p, target_names=CLASSES, digits=4))
    final[name] = {"test_acc": accuracy_score(y, p), "test_macro_f1": f1_score(y, p, average="macro")}
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_matrices.png", dpi=200); plt.close()

# ----------------------------- 4) ROBUSTNESS -----------------------------
print("\n=== STEP 4: Robustness under corruptions ===")
rob_rows = []
for variant, store in [("standard", STD), ("corruption_aug", AUG)]:
    for name in MODELS:
        _, p, y = evaluate(store[name], dl_test)
        rob_rows.append({"model": name, "variant": variant, "corruption": "clean", "severity": 0,
                         "accuracy": accuracy_score(y, p)})
        for cname, fn in CORRUPTIONS.items():
            for s in (1, 2, 3):
                set_seed(SEED)
                dl = loader(test_idx, corrupt=lambda im, fn=fn, s=s: fn(im, s))
                _, p, y = evaluate(store[name], dl)
                rob_rows.append({"model": name, "variant": variant, "corruption": cname,
                                 "severity": s, "accuracy": accuracy_score(y, p)})
        print("done", variant, name)
rob = pd.DataFrame(rob_rows); rob.to_csv(f"{OUT}/robustness_results.csv", index=False)

fig, ax = plt.subplots(1, len(CORRUPTIONS), figsize=(4.2 * len(CORRUPTIONS), 4), sharey=True)
for k, cname in enumerate(CORRUPTIONS):
    for name in MODELS:
        for variant, ls in [("standard", "-"), ("corruption_aug", "--")]:
            d = rob[(rob.model == name) & (rob.variant == variant) &
                    ((rob.corruption == cname) | (rob.corruption == "clean"))].sort_values("severity")
            ax[k].plot(d.severity, d.accuracy, ls, marker="o", color=COLORS[name],
                       label=f"{name} ({variant})" if k == 0 else None)
    ax[k].set_title(cname); ax[k].set_xlabel("Severity"); ax[k].set_xticks([0, 1, 2, 3])
ax[0].set_ylabel("Test accuracy"); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{OUT}/robustness_curves.png", dpi=200); plt.close()

# ----------------------------- 5) EFFICIENCY -----------------------------
print("\n=== STEP 5: Efficiency ===")
def efficiency(name):
    m = copy.deepcopy(STD[name]).cpu().eval()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    size_mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6
    x = torch.randn(1, 3, IMG, IMG); torch.set_num_threads(2)
    with torch.no_grad():
        for _ in range(10): m(x)
        t = time.time()
        for _ in range(50): m(x)
    return params, size_mb, (time.time() - t) / 50 * 1000

rows = []
for name in MODELS:
    pm, mb, ms = efficiency(name)
    r_std = rob[(rob.model == name) & (rob.variant == "standard") & (rob.corruption != "clean")].accuracy.mean()
    r_aug = rob[(rob.model == name) & (rob.variant == "corruption_aug") & (rob.corruption != "clean")].accuracy.mean()
    c_aug = rob[(rob.model == name) & (rob.variant == "corruption_aug") & (rob.corruption == "clean")].accuracy.iloc[0]
    rows.append({"model": name, "params_M": round(pm, 2), "size_MB": round(mb, 2), "cpu_latency_ms": round(ms, 1),
                 "test_acc": round(final[name]["test_acc"], 4), "test_macro_f1": round(final[name]["test_macro_f1"], 4),
                 "cv_acc_mean": cv_sum.loc[name, "acc_mean"], "cv_acc_std": cv_sum.loc[name, "acc_std"],
                 "mean_corrupt_acc_standard": round(r_std, 4), "mean_corrupt_acc_aug": round(r_aug, 4),
                 "clean_acc_after_aug": round(c_aug, 4), "robust_gain": round(r_aug - r_std, 4),
                 "accuracy_per_MB": round(final[name]["test_acc"] / mb, 4)})
summary = pd.DataFrame(rows); summary.to_csv(f"{OUT}/final_summary_table.csv", index=False)
print(summary.to_string(index=False))

# ----------------------------- 6) INTERPRETABILITY (Grad-CAM) -----------------------------
print("\n=== STEP 6: Grad-CAM ===")
def cam_for(name, model, pil, corrupt=None):
    img = corrupt(pil) if corrupt else pil
    x = tf_eval(img).unsqueeze(0).to(DEVICE)
    model.eval(); layer = get_layer(name, model); acts, grads = [], []
    def fh(m, i, o):
        acts.append(o); o.register_hook(lambda g: grads.append(g))
    h = layer.register_forward_hook(fh)
    out = model(x); c = out.argmax(1).item()
    model.zero_grad(); out[0, c].backward(); h.remove()
    a, g = acts[0].detach(), grads[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return np.asarray(img), cam, c

# one test sample per class
samples = []
for ci in range(len(CLASSES)):
    samples.append([i for i in test_idx if labels[i] == ci][0])

def cam_grid(corrupt, title, fname):
    fig, ax = plt.subplots(len(MODELS), len(samples), figsize=(2.6 * len(samples), 2.6 * len(MODELS)))
    for r, name in enumerate(MODELS):
        for c, j in enumerate(samples):
            img, cam, pred = cam_for(name, STD[name], IMGS[j], corrupt)
            ax[r, c].imshow(img); ax[r, c].imshow(cam, cmap="jet", alpha=0.45); ax[r, c].axis("off")
            ax[r, c].set_title(f"T:{CLASSES[labels[j]]}\nP:{CLASSES[pred]}", fontsize=7,
                               color="green" if pred == labels[j] else "red")
            if c == 0: ax[r, c].text(-0.05, 0.5, name, transform=ax[r, c].transAxes, rotation=90,
                                     va="center", ha="right", fontsize=8)
    fig.suptitle(title); plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=200); plt.close()

cam_grid(None, "Grad-CAM on clean images", "gradcam_clean.png")
cam_grid(lambda im: c_blur(im, 3), "Grad-CAM under Gaussian blur (severity 3)", "gradcam_blur_sev3.png")
cam_grid(lambda im: c_dark(im, 3), "Grad-CAM under low light (severity 3)", "gradcam_lowlight_sev3.png")

print("\nALL DONE. Files in", OUT)
print(os.listdir(OUT))
