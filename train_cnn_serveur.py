import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm
import rasterio
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA       = Path('/home/barrage/challenge2026MIASHS')
PATCHES_TR = DATA / 'SatelitePatches/PA-train'
PATCHES_TE = DATA / 'SatelitePatches/PA-test'
WORK_DIR   = Path('/home/grp4')
WORK_DIR.mkdir(parents=True, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ─── LABELS ───────────────────────────────────────────────────────────────────
pa_train = pd.read_csv(DATA / 'GLC25_PA_metadata_train.csv')
pa_test  = pd.read_csv(DATA / 'GLC25_PA_metadata_test.csv')

train_labels = (
    pa_train.dropna(subset=['speciesId'])
    .groupby('surveyId')['speciesId']
    .apply(lambda x: list(x.astype(int).unique()))
    .reset_index()
    .rename(columns={'speciesId': 'species_list'})
)

all_species = sorted(pa_train['speciesId'].dropna().astype(int).unique())
mlb = MultiLabelBinarizer(classes=all_species)
mlb.fit(train_labels['species_list'])
NUM_CLASSES = len(all_species)

print(f'Train: {len(train_labels)} surveys | Classes: {NUM_CLASSES}')
print(f'Test:  {len(pa_test["surveyId"].unique())} surveys')

# ─── RAM PRELOADING ───────────────────────────────────────────────────────────
def get_patch_path(survey_id, patches_dir):
    s = str(int(survey_id))
    return patches_dir / s[-2:] / s[-4:-2] / f'{survey_id}.tiff'

def load_all_patches(survey_ids, patches_dir, desc=''):
    n = len(survey_ids)
    arr = np.zeros((n, 4, 64, 64), dtype=np.float32)
    t0 = time.time()
    missing = 0
    for i, sid in enumerate(survey_ids):
        path = get_patch_path(sid, patches_dir)
        if path.exists():
            try:
                with rasterio.open(path) as src:
                    img = src.read().astype(np.float32)
                for b in range(img.shape[0]):
                    mn, mx = img[b].min(), img[b].max()
                    if mx > mn:
                        img[b] = (img[b] - mn) / (mx - mn)
                arr[i] = img
            except:
                missing += 1
        else:
            missing += 1
        if (i+1) % 10000 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i+1) * (n - i - 1)
            print(f'  {desc} {i+1}/{n} | {elapsed/60:.1f}min | ~{remaining/60:.1f}min restantes | manquantes: {missing}')
    print(f'{desc}: {n-missing}/{n} images en {(time.time()-t0)/60:.1f}min')
    return arr

all_tr_ids = train_labels['surveyId'].values
test_ids   = pa_test['surveyId'].unique()

print('Chargement train en RAM...')
train_patches = load_all_patches(all_tr_ids, PATCHES_TR, 'Train')
print('Chargement test en RAM...')
test_patches  = load_all_patches(test_ids,   PATCHES_TE, 'Test')
print(f'RAM: train={train_patches.nbytes/1e9:.1f}GB | test={test_patches.nbytes/1e9:.1f}GB')

# ─── DATASET ──────────────────────────────────────────────────────────────────
class RAMDataset(Dataset):
    def __init__(self, patches_arr, survey_ids, labels_df, mlb, is_train=True):
        self.patches    = patches_arr
        self.ids        = survey_ids
        self.is_train   = is_train
        self.mlb        = mlb
        self.label_dict = dict(zip(labels_df['surveyId'], labels_df['species_list'])) if labels_df is not None else {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = torch.tensor(self.patches[idx])
        if self.is_train:
            if torch.rand(1) > 0.5: img = torch.flip(img, dims=[2])
            if torch.rand(1) > 0.5: img = torch.flip(img, dims=[1])
        sid = self.ids[idx]
        if sid in self.label_dict:
            label = self.mlb.transform([self.label_dict[sid]])[0].astype(np.float32)
        else:
            label = np.zeros(len(self.mlb.classes_), dtype=np.float32)
        return img, torch.tensor(label)

tr_idx, val_idx = train_test_split(np.arange(len(all_tr_ids)), test_size=0.2, random_state=42)

train_ds = RAMDataset(train_patches[tr_idx],  all_tr_ids[tr_idx],  train_labels, mlb, is_train=True)
val_ds   = RAMDataset(train_patches[val_idx], all_tr_ids[val_idx], train_labels, mlb, is_train=False)
test_ds  = RAMDataset(test_patches,           test_ids,            None,         mlb, is_train=False)

train_dl = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=8, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# ─── MODELE ───────────────────────────────────────────────────────────────────
class ResNet18_4ch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base    = timm.create_model('resnet18', pretrained=True, num_classes=0)
        old_conv     = self.base.conv1
        new_conv     = nn.Conv2d(4, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3]  = old_conv.weight.mean(dim=1)
        self.base.conv1 = new_conv
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base.num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.base(x))

model = ResNet18_4ch(NUM_CLASSES).to(device)
print(f'Parametres: {sum(p.numel() for p in model.parameters()):,}')

# ─── ENTRAINEMENT 50 EPOCHS ───────────────────────────────────────────────────
criterion  = nn.BCEWithLogitsLoss()
optimizer  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler     = GradScaler()
CHECKPOINT = WORK_DIR / 'best_cnn_model.pt'

NUM_EPOCHS   = 50
best_f_score = 0.0
t_start      = time.time()

for epoch in range(NUM_EPOCHS):
    t_epoch = time.time()

    model.train()
    train_loss = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()
            val_preds.append(torch.sigmoid(outputs).cpu().float().numpy())
            val_true.append(labels.cpu().numpy())

    val_preds = np.vstack(val_preds)
    val_true  = np.vstack(val_true)

    THRESHOLD, TOP_K = 0.2, 10
    f_scores = []
    for i in range(len(val_preds)):
        above = np.where(val_preds[i] >= THRESHOLD)[0]
        if len(above) < TOP_K:
            above = np.argsort(val_preds[i])[::-1][:TOP_K]
        pred_set = set(above)
        true_set = set(np.where(val_true[i] == 1)[0])
        tp = len(pred_set & true_set)
        if tp == 0: f_scores.append(0.0); continue
        p = tp / len(pred_set)
        r = tp / len(true_set)
        f_scores.append(2*p*r/(p+r))

    avg_f     = np.mean(f_scores)
    tl        = train_loss / len(train_dl)
    vl        = val_loss   / len(val_dl)
    ep_time   = time.time() - t_epoch
    elapsed   = (time.time() - t_start) / 60
    remaining = ep_time * (NUM_EPOCHS - epoch - 1) / 60

    print(f'Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train: {tl:.4f} | Val: {vl:.4f} | F: {avg_f:.4f} | {ep_time:.0f}s | Ecoule: {elapsed:.1f}min | Restant: ~{remaining:.1f}min')

    if avg_f > best_f_score:
        best_f_score = avg_f
        torch.save(model.state_dict(), CHECKPOINT)
        print(f'  Meilleur modele sauvegarde (F={avg_f:.4f})')

    scheduler.step()

print(f'Entrainement termine! Meilleur F-score: {best_f_score:.4f}')

# ─── PREDICTIONS TEST ─────────────────────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

test_probs_all = []
with torch.no_grad():
    for imgs, _ in test_dl:
        with autocast():
            outputs = model(imgs.to(device))
        test_probs_all.append(torch.sigmoid(outputs).cpu().float().numpy())

test_probs_all = np.vstack(test_probs_all)
np.save(WORK_DIR / 'cnn_test_probs.npy', test_probs_all)
print(f'Predictions test: {test_probs_all.shape} | Sauvegarde: {WORK_DIR}/cnn_test_probs.npy')

# ─── SOUMISSION CNN ───────────────────────────────────────────────────────────
predictions = []
for i in range(len(test_probs_all)):
    probs = test_probs_all[i]
    above = np.where(probs >= THRESHOLD)[0]
    if len(above) < TOP_K:
        pred_idx = np.union1d(above, np.argsort(probs)[::-1][:TOP_K])
    else:
        pred_idx = above
    predictions.append(' '.join(map(str, sorted([int(mlb.classes_[j]) for j in pred_idx]))))

sub_path = WORK_DIR / 'submission_cnn_serveur.csv'
pd.DataFrame({'surveyId': test_ids, 'predictions': predictions}).to_csv(sub_path, index=False)
print(f'Soumission CNN sauvegardee: {sub_path}')
